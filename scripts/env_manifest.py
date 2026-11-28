#!/usr/bin/env python3
"""Read-only environment and artifact manifest.

Writes one JSON object: UTC time, optional git commit and dirty flag, kernel, OS,
CPU, Mesa/RADV package, ROCm and amdclang++/hipcc, the AMD GPU's PCI identity and
the overdrive track, selected environment variables, and SHA-256 of each --artifact.

Every string is scrubbed before it is written. A path under /home/<user> or
/Users/<user> becomes <home>. --self-test checks the stock/overclock classifier,
that scrub, a deliberate scrub break, and the schema. It does not open the GPU
or sysfs.

  python3 -I scripts/env_manifest.py --self-test
  python3 -I scripts/env_manifest.py --out manifest.json [--repo DIR] [--device 0x7550] [--artifact FILE]

Exit 0 when the manifest is written, or when --self-test passes. Exit 1 on a
self-test failure or a leftover /home or /Users path. Exit 2 on usage.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True

# The OD_RANGE ceiling is 1500 MHz even when the request is the stock 1259.
# The track uses the configured OD_MCLK level 1, plus the two offsets.
STOCK_MCLK_MAX = 1259
PREFERRED_DEVICE = 0x7550

_HOME = re.compile(r"/(?:home|Users)/[^/\s\"']+")
LIVE = False  # set when a real sysfs or GPU read runs; --self-test requires it to stay false

SCHEMA_KEYS = (
    "generated_utc",
    "git",
    "host",
    "toolchain",
    "gpu",
    "env",
    "artifacts",
)


def scrub(text):
    """Replace a home-directory prefix. Not a no-op: --self-test fails if it is."""
    if not isinstance(text, str):
        return text
    return _HOME.sub("<home>", text)


def scrub_tree(value):
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {scrub(str(k)): scrub_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_tree(v) for v in value]
    return value


def leaks(text):
    return bool(re.search(r"/home/|/Users/", text))


def _section(text, name):
    """Body of a `NAME:` block, up to the next ALL-CAPS header. OD_RANGE is its own block."""
    lines = text.splitlines()
    body = []
    on = False
    for ln in lines:
        if re.match(r"^[A-Z][A-Z0-9_]*:\s*$", ln.strip()):
            if on:
                break
            on = ln.strip()[:-1] == name
            continue
        if on:
            body.append(ln)
    return body


def _first_int(lines):
    for ln in lines:
        m = re.search(r"-?\d+", ln)
        if m:
            return int(m.group(0))
    return None


def classify_od(text):
    """Return (track, parsed). track is stock, overclock, or unknown.

    stock: core offset 0, voltage offset 0, and the OD_MCLK level-1 request at or
    under STOCK_MCLK_MAX. overclock: any of those three says otherwise. unknown:
    the three fields are not all present. The OD_RANGE ceiling is not the request.
    """
    parsed = {"sclk_offset_mhz": None, "vddgfx_offset_mv": None, "mclk_max_mhz": None}
    if not text or not str(text).strip():
        return "unknown", parsed
    parsed["sclk_offset_mhz"] = _first_int(_section(text, "OD_SCLK_OFFSET"))
    parsed["vddgfx_offset_mv"] = _first_int(_section(text, "OD_VDDGFX_OFFSET"))
    # Level lines look like "1: 1259MHz". The first integer is the level, not the clock.
    parsed["mclk_max_mhz"] = None
    for ln in _section(text, "OD_MCLK"):
        if re.match(r"\s*1\s*:", ln):
            hit = re.search(r":\s*(-?\d+)", ln)
            if hit:
                parsed["mclk_max_mhz"] = int(hit.group(1))
            break
    if None in parsed.values():
        return "unknown", parsed
    stock = (
        parsed["sclk_offset_mhz"] == 0
        and parsed["vddgfx_offset_mv"] == 0
        and parsed["mclk_max_mhz"] <= STOCK_MCLK_MAX
    )
    return ("stock" if stock else "overclock"), parsed


def _run(argv, timeout=20):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    out = (p.stdout or "") + (p.stderr or "")
    return (out.strip() or None), None if p.returncode == 0 else "exit %s" % p.returncode


def _read(path):
    global LIVE
    LIVE = True
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip(), None
    except OSError as exc:
        return None, exc.strerror or str(exc)


def _cpu_model():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if ln.startswith("model name"):
                    return ln.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _os_release():
    try:
        with open("/etc/os-release", "r", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if ln.startswith("PRETTY_NAME="):
                    return ln.split("=", 1)[1].strip().strip('"')
    except OSError:
        return None
    return None


def _mesa():
    # rpm -q does not open the device. vulkaninfo does, so it is not used.
    for pkg in ("mesa-vulkan-drivers", "mesa-vulkan-drivers-git-freeworld"):
        out, err = _run(["rpm", "-q", pkg])
        if out and err is None and not out.startswith("package "):
            return out.splitlines()[0]
    return None


def _tool_version(names):
    path = None
    for name in names:
        path = shutil.which(name)
        if path:
            break
    rocm = os.environ.get("ROCM_PATH")
    if path is None and rocm:
        for name in names:
            cand = os.path.join(rocm, "bin", name)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                path = cand
                break
    if path is None:
        return {"path": None, "version": None}
    out, _err = _run([path, "--version"])
    first = out.splitlines()[0] if out else None
    return {"path": scrub(path), "version": first}


def _git(repo):
    if not repo:
        return {"commit": None, "dirty": None, "repo": None}
    commit, err = _run(["git", "-C", repo, "rev-parse", "HEAD"])
    status, status_err = _run(["git", "-C", repo, "status", "--porcelain"])
    # bool(None) is false, so a failed status must not be reported as a clean tree.
    dirty = bool(status) if err is None and status_err is None else None
    return {
        "commit": commit if err is None else None,
        "dirty": dirty,
        "repo": scrub(os.path.abspath(repo)),
        "error": err if err is not None else status_err,
    }


def _amd_cards():
    """PCI device dirs of AMD DRM cards. Read-only. Marks LIVE."""
    global LIVE
    LIVE = True
    root = "/sys/class/drm"
    found = []
    try:
        names = os.listdir(root)
    except OSError:
        return found
    for name in sorted(names):
        if not re.fullmatch(r"card\d+", name):
            continue
        dev = os.path.join(root, name, "device")
        vendor, _ = _read(os.path.join(dev, "vendor"))
        if vendor is None or vendor.lower() not in ("0x1002", "1002"):
            continue
        device, _ = _read(os.path.join(dev, "device"))
        try:
            devid = int(device, 16) if device else None
        except ValueError:
            devid = None
        found.append((devid, dev, name))
    return found


def _pick_card(device_id):
    cards = _amd_cards()
    if device_id is not None:
        for devid, dev, name in cards:
            if devid == device_id:
                return devid, dev, name
        return None
    for devid, dev, name in cards:
        if devid == PREFERRED_DEVICE:
            return devid, dev, name
    return cards[0] if cards else None


def _hwmon_cap(dev):
    base = os.path.join(dev, "hwmon")
    try:
        mons = os.listdir(base)
    except OSError:
        return None, None
    for mon in sorted(mons):
        text, err = _read(os.path.join(base, mon, "power1_cap"))
        if text and text.isdigit():
            microwatts = int(text)
            return microwatts, microwatts / 1_000_000
    return None, None


def _firmware(dev):
    """Version files under fw_version/. That directory is the driver's firmware list."""
    names = []
    base = os.path.join(dev, "fw_version")
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return names
    for name in entries:
        text, _ = _read(os.path.join(base, name))
        if text:
            names.append({"name": name, "value": text[:200]})
    return names


def _displays():
    root = "/sys/class/drm"
    out = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in sorted(names):
        if not re.search(r"card\d+-", name):
            continue
        status, _ = _read(os.path.join(root, name, "status"))
        if status:
            out.append({"name": name, "status": status})
    return out


def _profile_line(text):
    if not text:
        return None
    for ln in text.splitlines():
        if "*" in ln:
            return ln.strip()
    return None


def _env_selected():
    prefixes = ("HSA_", "RADV_", "AMD_", "VK_", "PROTON_")
    kept = {}
    for key, val in os.environ.items():
        if key == "ROCM_PATH" or key.startswith(prefixes):
            kept[key] = scrub(val)
    return dict(sorted(kept.items()))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _rocm_info():
    """Compiler versions, plus a version file under ROCM_PATH when one is set. No rocminfo."""
    root = os.environ.get("ROCM_PATH")
    version = None
    if root:
        for rel in (".info/version-dev", ".info/version"):
            path = os.path.join(root, rel)
            if os.path.isfile(path):
                # Reading a version file is not a GPU open. It may live under a home prefix;
                # the path is scrubbed by the caller.
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        version = fh.readline().strip() or None
                except OSError:
                    version = None
                if version:
                    break
    return {
        "version": version,
        "ROCM_PATH": scrub(root) if root else None,
        "amdclangpp": _tool_version(("amdclang++",)),
        "hipcc": _tool_version(("hipcc",)),
    }


def collect(repo=None, device_id=None, artifacts=()):
    """Live manifest. Read-only. Never writes sysfs and never starts a GPU job."""
    toolchain_mesa = _mesa()
    rocm = _rocm_info()
    picked = _pick_card(device_id)
    if picked is None:
        gpu = {
            "present": False,
            "requested_device": None if device_id is None else hex(device_id),
            "track": "unknown",
            "error": "no matching AMD DRM card",
        }
    else:
        devid, dev, name = picked
        vendor, _ = _read(os.path.join(dev, "vendor"))
        device, _ = _read(os.path.join(dev, "device"))
        revision, _ = _read(os.path.join(dev, "revision"))
        vbios, _ = _read(os.path.join(dev, "vbios_version"))
        vbios_build, _ = _read(os.path.join(dev, "vbios_build"))
        driver = None
        link = os.path.join(dev, "driver")
        if os.path.islink(link):
            global LIVE
            LIVE = True
            driver = os.path.basename(os.readlink(link))
        level, _ = _read(os.path.join(dev, "power_dpm_force_performance_level"))
        cap_uw, cap_w = _hwmon_cap(dev)
        od, od_err = _read(os.path.join(dev, "pp_od_clk_voltage"))
        track, parsed = classify_od(od or "")
        profile, _ = _read(os.path.join(dev, "pp_power_profile_mode"))
        gpu = {
            "present": True,
            "drm": name,
            "vendor": vendor,
            "device": device,
            "device_id": devid,
            "revision": revision,
            "vbios": vbios,
            "vbios_build": vbios_build,
            "driver": driver,
            "firmware": _firmware(dev),
            "performance_level": level,
            "power_cap_uw": cap_uw,
            "power_cap_w": cap_w,
            "od": parsed,
            "track": track,
            "od_error": od_err,
            "power_profile_active": _profile_line(profile),
            "displays": _displays(),
        }
    arts = []
    for path in artifacts:
        item = {"path": scrub(os.path.abspath(path)), "sha256": None, "bytes": None, "error": None}
        try:
            item["bytes"] = os.path.getsize(path)
            item["sha256"] = _sha256(path)
        except OSError as exc:
            item["error"] = exc.strerror or str(exc)
        arts.append(item)
    doc = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": _git(repo),
        "host": {
            "kernel": platform.release(),
            "os": _os_release(),
            "cpu": _cpu_model(),
            "machine": platform.machine(),
        },
        "toolchain": {
            "mesa_radv": toolchain_mesa,
            "rocm": rocm,
        },
        "gpu": gpu,
        "env": _env_selected(),
        "artifacts": arts,
    }
    return scrub_tree(doc)


def _expect(ok, label, fails):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        fails.append(label)
    return ok


def self_test():
    """Classifier, scrub, a deliberate break, and schema. No sysfs and no GPU."""
    global LIVE
    before = LIVE
    fails = []

    stock = "OD_SCLK_OFFSET:\n0Mhz\nOD_MCLK:\n0: 97Mhz\n1: 1259MHz\nOD_RANGE:\nSCLK_OFFSET: -500Mhz 1000Mhz\nMCLK: 97Mhz 1500Mhz\nVDDGFX_OFFSET: -200mv 0mv\nOD_VDDGFX_OFFSET:\n0mV\n"
    oc = "OD_SCLK_OFFSET:\n150Mhz\nOD_MCLK:\n0: 97Mhz\n1: 1409MHz\nOD_RANGE:\nSCLK_OFFSET: -500Mhz 1000Mhz\nMCLK: 97Mhz 1500Mhz\nVDDGFX_OFFSET: -200mv 0mv\nOD_VDDGFX_OFFSET:\n-50mV\n"
    mem_only = stock.replace("1: 1259MHz", "1: 1409MHz")
    core_only = stock.replace("0Mhz", "150Mhz", 1)
    volt_only = stock.replace("0mV", "-50mV", 1)
    missing = "OD_MCLK:\n0: 97Mhz\n1: 1259MHz\n"

    cases = (
        (stock, "stock", "offsets 0 and mclk request 1259 are stock, even though OD_RANGE says 1500"),
        (oc, "overclock", "tonight's OC table is overclock"),
        (mem_only, "overclock", "a memory request above 1259 is overclock with both offsets at 0"),
        (core_only, "overclock", "a nonzero core offset is overclock"),
        (volt_only, "overclock", "a nonzero voltage offset is overclock"),
        ("", "unknown", "empty text is unknown"),
        (missing, "unknown", "a table missing an offset is unknown"),
        ("not a table", "unknown", "unparseable text is unknown"),
    )
    for text, want, label in cases:
        got, parsed = classify_od(text)
        _expect(got == want, "%s (got %s)" % (label, got), fails)
    _, stock_parsed = classify_od(stock)
    _, oc_parsed = classify_od(oc)
    _expect(
        stock_parsed == {"sclk_offset_mhz": 0, "vddgfx_offset_mv": 0, "mclk_max_mhz": 1259},
        "stock request is 0 / 0 / 1259, not the level index or the 1500 range",
        fails,
    )
    _expect(
        oc_parsed == {"sclk_offset_mhz": 150, "vddgfx_offset_mv": -50, "mclk_max_mhz": 1409},
        "overclock request is 150 / -50 / 1409",
        fails,
    )

    sample = "log /home/someone/Projects/secret and /Users/someone/x"
    cleaned = scrub(sample)
    _expect(not leaks(cleaned) and "<home>" in cleaned, "scrub strips /home and /Users", fails)
    _expect(leaks(sample), "the leak detector rejects an unscrubbed /home path", fails)

    def _scrub_broken(text):
        return text  # the break: a scrub that does nothing

    break_out = _scrub_broken(sample)
    _expect(leaks(break_out), "deliberate break: a no-op scrub still leaks and must be rejected", fails)
    _expect(
        leaks(break_out) and not leaks(scrub(sample)),
        "the same detector passes the real scrub and fails the break",
        fails,
    )

    nested = scrub_tree({"cmd": sample, "inner": [sample]})
    blob = json.dumps(nested)
    _expect(not leaks(blob), "scrub_tree clears a nested document", fails)

    doc = {
        "generated_utc": "2026-11-26T00:00:00Z",
        "git": {"commit": "abc", "dirty": False, "repo": scrub("/home/someone/src"), "error": None},
        "host": {"kernel": "test", "os": "test", "cpu": "test", "machine": "x86_64"},
        "toolchain": {
            "mesa_radv": "test",
            "rocm": {"version": None, "ROCM_PATH": None, "amdclangpp": {"path": None, "version": None}, "hipcc": {"path": None, "version": None}},
        },
        "gpu": {
            "present": False,
            "requested_device": None,
            "track": "unknown",
            "error": "fixture",
            "od": {"sclk_offset_mhz": 0, "vddgfx_offset_mv": 0, "mclk_max_mhz": 1259},
        },
        "env": {"ROCM_PATH": scrub("/home/someone/rocm")},
        "artifacts": [{"path": scrub("/home/someone/a.bin"), "sha256": "ab" * 32, "bytes": 1, "error": None}],
    }
    missing_keys = [k for k in SCHEMA_KEYS if k not in doc]
    _expect(not missing_keys, "schema has %s" % ", ".join(SCHEMA_KEYS), fails)
    _expect(doc["gpu"]["track"] in ("stock", "overclock", "unknown"), "track is one of the three labels", fails)
    rendered = json.dumps(scrub_tree(doc))
    _expect(not leaks(rendered), "a fixture manifest renders with no /home or /Users path", fails)
    saved_run = _run

    def _fake_run(argv, timeout=20):
        if argv and argv[0] == "git" and "status" in argv:
            return None, "exit 128"
        if argv and argv[0] == "git" and "rev-parse" in argv:
            return "abc123", None
        return saved_run(argv, timeout)

    globals()["_run"] = _fake_run
    try:
        git_info = _git("/nowhere/not-a-checkout")
    finally:
        globals()["_run"] = saved_run
    # The old line was `None if rev-parse failed else bool(status)`. bool(None) is False.
    old_dirty = None if False else bool(None)
    _expect(old_dirty is False, "red: a failed git status used to serialize as dirty false", fails)
    _expect(git_info["dirty"] is None, "green: a failed git status serializes as dirty null", fails)
    _expect(git_info["commit"] == "abc123", "rev-parse is kept when only status fails", fails)
    _expect(git_info.get("error") == "exit 128", "the status error is recorded", fails)
    _expect('"dirty": null' in json.dumps({"dirty": git_info["dirty"]}), "dirty null is JSON null, not false", fails)
    _expect(LIVE is before, "self-test did not touch a live reader", fails)

    print("VERDICT env_manifest self-test " + ("PASS" if not fails else "FAIL"))
    return 0 if not fails else 1


def _parse_device(text):
    try:
        return int(text, 0)
    except ValueError:
        sys.stderr.write("device id must be an integer such as 0x7550\n")
        raise SystemExit(2)


def main(argv):
    p = argparse.ArgumentParser(description="Read-only environment manifest. Scrubs home paths.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out")
    p.add_argument("--repo", help="git checkout to record (commit and dirty flag only)")
    p.add_argument("--device", help="PCI device id to select, for example 0x7550. Default: 0x7550 if present, else the first AMD card")
    p.add_argument("--artifact", action="append", default=[], help="file to hash; repeat for several")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    device = _parse_device(args.device) if args.device else None
    doc = collect(repo=args.repo, device_id=device, artifacts=args.artifact)
    blob = json.dumps(doc, indent=1, sort_keys=True)
    if leaks(blob):
        sys.stderr.write("env_manifest: refused to write a document that still contains a home path\n")
        return 1
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(blob)
            fh.write("\n")
    else:
        sys.stdout.write(blob)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
