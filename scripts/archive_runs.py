#!/usr/bin/env python3
"""Machine-readable raw-run archiver and table-to-run manifest generator.

Stdlib only. Always run with `python3 -I` (no bytecode written).

Subcommands:
  manifest --repo DIR --out FILE [--runs-root DIR ...]
    Scan measurements/*.md and README.md for referenced run IDs/paths, check whether
    each run directory is present locally under --runs-root or <repo>/runs, hash key
    artifacts for present runs, scrub any home directory prefixes, and write JSON.

  pack --run-dir DIR --out-dir DIR [--repo DIR] [--name NAME] [--no-live-manifest]
    Create a deterministic, immutable bundle:
      <name>.tar.gz         (sorted relative paths, mtime=0, zero home paths)
      <name>.manifest.json  (metadata, git state, optional env_manifest, file inventory)
      <name>.SHA256SUMS     (SHA-256 checksums for .tar.gz and .manifest.json)

  verify (--out-dir DIR --name NAME | --checksums FILE)
    Verify .SHA256SUMS, verify .manifest.json has zero home-path leaks, and verify
    every member inside .tar.gz matches .manifest.json's exact inventory and SHA-256.

  --self-test
    CPU-only self-test with synthetic markdown, real read-only repo scan (if present),
    pack+verify round-trip, and 4 deliberate breaks (tamper, inventory mismatch,
    PII leak, missing tar member).
"""

import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True

_HOME_PRE = "/" + "home/"
_USERS_PRE = "/" + "Users/"
_HOME_RE = re.compile(r"(?:" + re.escape(_HOME_PRE) + r"|" + re.escape(_USERS_PRE) + r")[^/\s\"']+")
_LEAK_RE = re.compile(re.escape(_HOME_PRE) + r"|" + re.escape(_USERS_PRE))

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_RUN_PATH_RE = re.compile(r"\b((?:runs|afk-r\d+)/[A-Za-z0-9_./<>-]+)")
_BARE_RUN_RE = re.compile(r"\b(2026\d{4}(?:T\d{6}Z)?-[A-Za-z0-9_-]+)\b")

_MAX_KEY_ARTIFACTS = 64


def scrub(text):
    """Replace any home-directory prefix with <home>."""
    if not isinstance(text, str):
        return text
    return _HOME_RE.sub("<home>", text)


def scrub_tree(value):
    """Recursively scrub all strings in nested dicts/lists."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {scrub(str(k)): scrub_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_tree(v) for v in value]
    return value


def has_home_leak(text):
    """Return True if text contains a raw home-directory path prefix."""
    return bool(_LEAK_RE.search(text))


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _run_cmd(argv, timeout=20):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    out = (p.stdout or "") + (p.stderr or "")
    return (out.strip() or None), (None if p.returncode == 0 else f"exit {p.returncode}")


def _git_info(repo):
    if not repo:
        return {"commit": None, "dirty": None, "repo": None, "error": None}
    commit, err = _run_cmd(["git", "-C", repo, "rev-parse", "HEAD"])
    status, status_err = _run_cmd(["git", "-C", repo, "status", "--porcelain"])
    dirty = bool(status) if err is None and status_err is None else None
    return {
        "commit": commit if err is None else None,
        "dirty": dirty,
        "repo": scrub(os.path.abspath(repo)),
        "error": err if err is not None else status_err,
    }


def _is_placeholder(token):
    if "<" in token or ">" in token:
        return True
    if token.endswith("...") or token.endswith("-") or token in ("runs/", "runs"):
        return True
    return False


def _normalize_run_id(raw_token):
    """Normalize a markdown run token to its run folder identifier."""
    tok = raw_token.rstrip(".,;:)]}`'\"")
    if tok.startswith("runs/"):
        rest = tok[len("runs/"):].strip("/")
        if not rest:
            return None
        parts = rest.split("/")
        first = parts[0]
        if not first or first in ("latest", "RESTORE-FAILED", "gpu.lock", "...", "...") or first.endswith("-"):
            return None
        # Keep top-level run folder (e.g. runs/<stamp>/file.log -> <stamp>)
        return first
    if tok.startswith("afk-r"):
        parts = [p for p in tok.strip("/").split("/") if p]
        if len(parts) >= 3 and parts[1].endswith("-results"):
            return "/".join(parts[:3])
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return parts[0] if parts else None
    return tok


def extract_markdown_run_refs(md_text, rel_source):
    """Extract referenced run IDs/folders from one markdown document."""
    refs = []
    heading = "(top)"
    for lineno, line in enumerate(md_text.splitlines(), 1):
        hm = _HEADING_RE.match(line)
        if hm:
            heading = hm.group(2).strip()
        seen_on_line = []
        for m in _RUN_PATH_RE.finditer(line):
            raw = m.group(1).rstrip(".,;:)]}`'\"")
            if _is_placeholder(raw):
                continue
            rid = _normalize_run_id(raw)
            if not rid:
                continue
            refs.append({
                "source_file": rel_source,
                "section": heading,
                "line": lineno,
                "raw_token": raw,
                "run_id": rid,
            })
            seen_on_line.append((m.start(1), m.end(1)))
        for m in _BARE_RUN_RE.finditer(line):
            s, e = m.start(1), m.end(1)
            if any(ps <= s and e <= pe for ps, pe in seen_on_line):
                continue
            raw = m.group(1)
            refs.append({
                "source_file": rel_source,
                "section": heading,
                "line": lineno,
                "raw_token": raw,
                "run_id": raw,
            })
    return refs


def _is_key_artifact(rel_path):
    base = os.path.basename(rel_path)
    if base.startswith("env-manifest") and base.endswith(".json"):
        return True
    if base.startswith("summary") and (base.endswith(".txt") or base.endswith(".md")):
        return True
    if base in ("analysis.txt", "controls.txt", "card.txt", "config.txt", "jobs.txt"):
        return True
    if base.endswith((".csv", ".json", ".s")):
        return True
    return False


def _inspect_run_dir(run_dir, label):
    file_count = 0
    total_bytes = 0
    key_candidates = []
    for root, dirs, files in os.walk(run_dir):
        dirs.sort()
        files.sort()
        for fn in files:
            full = os.path.join(root, fn)
            if os.path.islink(full):
                continue
            try:
                sz = os.path.getsize(full)
            except OSError:
                continue
            file_count += 1
            total_bytes += sz
            rel = os.path.relpath(full, run_dir).replace(os.sep, "/")
            if _is_key_artifact(rel):
                key_candidates.append((rel, full, sz))
    key_artifacts = []
    for rel, full, sz in key_candidates[:_MAX_KEY_ARTIFACTS]:
        key_artifacts.append({
            "path": rel,
            "size_bytes": sz,
            "sha256": _sha256_file(full),
        })
    return {
        "status": "present",
        "location": scrub(label),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "key_artifacts": key_artifacts,
        "key_artifacts_truncated": len(key_candidates) > _MAX_KEY_ARTIFACTS,
    }


def _resolve_run_dir(run_id, repo, runs_roots):
    candidates = []
    if repo:
        candidates.append((os.path.join(repo, "runs", run_id), f"repo/runs/{run_id}"))
    for idx, rr in enumerate(runs_roots):
        tag = f"runs_root[{idx}]"
        candidates.append((os.path.join(rr, run_id), f"{tag}/{run_id}"))
        if "/" not in run_id:
            # Also check one level down (e.g., runs_root/afk-r13/<run_id>)
            try:
                for entry in sorted(os.listdir(rr)):
                    sub = os.path.join(rr, entry, run_id)
                    if os.path.isdir(sub):
                        candidates.append((sub, f"{tag}/{entry}/{run_id}"))
            except OSError:
                pass
    for path, label in candidates:
        if os.path.isdir(path):
            return _inspect_run_dir(path, label)
    return {
        "status": "external",
        "location": None,
        "file_count": None,
        "total_bytes": None,
        "key_artifacts": [],
        "key_artifacts_truncated": False,
    }


def generate_manifest(repo, runs_roots):
    """Build the table-to-run manifest dictionary for a repository."""
    repo_abs = os.path.abspath(repo)
    md_files = []
    readme = os.path.join(repo_abs, "README.md")
    if os.path.isfile(readme):
        md_files.append((readme, "README.md"))
    meas_dir = os.path.join(repo_abs, "measurements")
    if os.path.isdir(meas_dir):
        for fn in sorted(os.listdir(meas_dir)):
            if fn.endswith(".md"):
                full = os.path.join(meas_dir, fn)
                if os.path.isfile(full):
                    md_files.append((full, f"measurements/{fn}"))

    all_refs = []
    for full, rel in md_files:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            all_refs.extend(extract_markdown_run_refs(fh.read(), rel))

    unique_ids = sorted({r["run_id"] for r in all_refs})
    runs_index = {}
    for rid in unique_ids:
        info = _resolve_run_dir(rid, repo_abs, runs_roots)
        info["reference_count"] = sum(1 for r in all_refs if r["run_id"] == rid)
        info["sources"] = sorted({r["source_file"] for r in all_refs if r["run_id"] == rid})
        runs_index[rid] = info

    doc = {
        "schema_version": 1,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": _git_info(repo_abs),
        "scanned_files": [rel for _, rel in md_files],
        "reference_count": len(all_refs),
        "unique_run_count": len(unique_ids),
        "present_run_count": sum(1 for v in runs_index.values() if v["status"] == "present"),
        "external_run_count": sum(1 for v in runs_index.values() if v["status"] == "external"),
        "runs": runs_index,
        "references": all_refs,
    }
    return scrub_tree(doc)


def _write_json_strict(doc, out_path):
    blob = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if has_home_leak(blob):
        sys.stderr.write("archive_runs: refused to write JSON containing a home path\n")
        return 1
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(blob)
    return 0


def _collect_run_files(run_dir, exclude_dirs=(), exclude_names=(), exclude_top=()):
    """Collect sorted (rel_path, abs_path) pairs for regular files in run_dir."""
    items = []
    run_abs = os.path.abspath(run_dir)
    # Only exclude an out_dir when it is strictly inside run_dir (never when out_dir is parent or equal).
    strict_ex_dirs = []
    for d in exclude_dirs:
        if not d:
            continue
        ex_abs = os.path.abspath(d)
        try:
            if ex_abs != run_abs and os.path.commonpath([run_abs, ex_abs]) == run_abs:
                strict_ex_dirs.append(ex_abs)
        except ValueError:
            pass
    top_ex = set(exclude_top)
    name_ex = set(exclude_names)
    for root, dirs, files in os.walk(run_abs):
        kept_dirs = []
        for d in sorted(dirs):
            full_d = os.path.abspath(os.path.join(root, d))
            if root == run_abs and d in top_ex:
                continue
            if any(os.path.commonpath([full_d, ex]) == ex for ex in strict_ex_dirs):
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs
        files.sort()
        for fn in files:
            if fn == ".replicate.lock":
                continue
            if root == run_abs and (fn in name_ex or fn in top_ex):
                continue
            full = os.path.join(root, fn)
            if os.path.islink(full):
                continue
            rel = os.path.relpath(full, run_abs).replace(os.sep, "/")
            if rel.startswith("/") or rel.startswith("..") or "/../" in rel:
                raise ValueError(f"unsafe relative path: {rel}")
            if has_home_leak("/" + rel) or has_home_leak(rel):
                raise ValueError(f"home path segment in member name: {rel}")
            items.append((rel, full))
    items.sort(key=lambda t: t[0])
    return items


def _read_archived_bytes(path, scrub_text=True):
    """Read file bytes for archiving; when scrub_text is True, scrub home prefixes in UTF-8 text."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if not scrub_text or b"\x00" in raw:
        return raw
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    if not has_home_leak(text):
        return raw
    return scrub(text).encode("utf-8")


def _collect_live_env_manifest(repo):
    env_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env_manifest.py")
    if not os.path.isfile(env_script):
        return None, "env_manifest.py not found"
    cmd = [sys.executable, "-I", env_script]
    if repo:
        cmd.extend(["--repo", repo])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if p.returncode != 0:
        return None, (p.stderr or p.stdout or f"exit {p.returncode}").strip()
    try:
        return json.loads(p.stdout), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def pack_run(run_dir, out_dir, repo=None, name=None, live_manifest=True, exclude_top=(), scrub_text=True):
    """Create deterministic .tar.gz, .manifest.json, and .SHA256SUMS in out_dir."""
    if not os.path.isdir(run_dir):
        sys.stderr.write(f"archive_runs: run directory does not exist: {run_dir}\n")
        return 2
    bundle_name = name or os.path.basename(os.path.abspath(run_dir)) or "run-bundle"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", bundle_name):
        sys.stderr.write(f"archive_runs: invalid bundle name: {bundle_name}\n")
        return 2
    os.makedirs(out_dir, exist_ok=True)
    tar_name = f"{bundle_name}.tar.gz"
    manifest_name = f"{bundle_name}.manifest.json"
    sums_name = f"{bundle_name}.SHA256SUMS"
    run_abs = os.path.abspath(run_dir)
    out_abs = os.path.abspath(out_dir)
    same_dir_ex = (tar_name, manifest_name, sums_name) if run_abs == out_abs else ()
    try:
        items = _collect_run_files(
            run_dir,
            exclude_dirs=(out_dir,),
            exclude_names=same_dir_ex,
            exclude_top=exclude_top,
        )
    except ValueError as exc:
        sys.stderr.write(f"archive_runs: {exc}\n")
        return 1

    tar_path = os.path.join(out_dir, tar_name)
    manifest_path = os.path.join(out_dir, manifest_name)
    sums_path = os.path.join(out_dir, sums_name)

    inventory = []
    total_bytes = 0
    with open(tar_path, "wb") as raw_fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=0) as gz_fh:
            with tarfile.open(fileobj=gz_fh, mode="w", format=tarfile.PAX_FORMAT) as tf:
                for rel, full in items:
                    data = _read_archived_bytes(full, scrub_text=scrub_text)
                    if b"\x00" not in data:
                        try:
                            if has_home_leak(data.decode("utf-8")):
                                sys.stderr.write(f"archive_runs: refused to pack unscrubbed home path in {rel}\n")
                                return 1
                        except UnicodeDecodeError:
                            pass
                    sz = len(data)
                    total_bytes += sz
                    sha = _sha256_bytes(data)
                    inventory.append({
                        "path": rel,
                        "size_bytes": sz,
                        "sha256": sha,
                    })
                    ti = tarfile.TarInfo(name=rel)
                    ti.size = sz
                    ti.mode = 0o644
                    ti.mtime = 0
                    ti.uid = 0
                    ti.gid = 0
                    ti.uname = ""
                    ti.gname = ""
                    tf.addfile(ti, io.BytesIO(data))

    tar_sha = _sha256_file(tar_path)
    tar_size = os.path.getsize(tar_path)

    env_doc = None
    env_err = None
    if live_manifest:
        env_doc, env_err = _collect_live_env_manifest(repo)

    manifest = {
        "schema_version": 1,
        "bundle_name": bundle_name,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": _git_info(repo),
        "live_manifest_requested": bool(live_manifest),
        "env_manifest": env_doc,
        "env_manifest_error": env_err,
        "archive": {
            "filename": tar_name,
            "size_bytes": tar_size,
            "sha256": tar_sha,
        },
        "file_count": len(inventory),
        "total_bytes": total_bytes,
        "files": inventory,
    }
    manifest = scrub_tree(manifest)
    if _write_json_strict(manifest, manifest_path) != 0:
        return 1

    manifest_sha = _sha256_file(manifest_path)
    sums_text = f"{tar_sha}  {tar_name}\n{manifest_sha}  {manifest_name}\n"
    if has_home_leak(sums_text):
        sys.stderr.write("archive_runs: refused to write SHA256SUMS containing a home path\n")
        return 1
    with open(sums_path, "w", encoding="utf-8") as fh:
        fh.write(sums_text)
    return 0


def verify_bundle(out_dir=None, name=None, checksums_path=None):
    """Verify .SHA256SUMS, .manifest.json leaks/metadata, and every member in .tar.gz."""
    if checksums_path:
        sums_path = os.path.abspath(checksums_path)
        base_dir = os.path.dirname(sums_path)
        fn = os.path.basename(sums_path)
        if not name and fn.endswith(".SHA256SUMS"):
            name = fn[:-len(".SHA256SUMS")]
        if not name:
            sys.stderr.write("archive_runs: verify needs --name when --checksums does not end in .SHA256SUMS\n")
            return 2
    else:
        if not out_dir or not name:
            sys.stderr.write("archive_runs: verify requires (--out-dir DIR --name NAME) or --checksums FILE\n")
            return 2
        base_dir = os.path.abspath(out_dir)
        sums_path = os.path.join(base_dir, f"{name}.SHA256SUMS")

    if not os.path.isfile(sums_path):
        sys.stderr.write(f"verify FAIL: missing checksums file {sums_path}\n")
        return 1

    tar_name = f"{name}.tar.gz"
    manifest_name = f"{name}.manifest.json"
    expected_sums = {}
    with open(sums_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                sys.stderr.write(f"verify FAIL: malformed checksum line {lineno} in {sums_path}\n")
                return 1
            expected_sums[parts[1].lstrip("*")] = parts[0]

    for req in (tar_name, manifest_name):
        if req not in expected_sums:
            sys.stderr.write(f"verify FAIL: {req} missing from {sums_path}\n")
            return 1
        fpath = os.path.join(base_dir, req)
        if not os.path.isfile(fpath):
            sys.stderr.write(f"verify FAIL: missing artifact {fpath}\n")
            return 1
        got_sha = _sha256_file(fpath)
        if got_sha != expected_sums[req]:
            sys.stderr.write(f"verify FAIL: SHA-256 mismatch for {req} (expected {expected_sums[req]}, got {got_sha})\n")
            return 1

    manifest_path = os.path.join(base_dir, manifest_name)
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest_raw = fh.read()
    if has_home_leak(manifest_raw):
        sys.stderr.write(f"verify FAIL: home-path leak detected in {manifest_name}\n")
        return 1
    try:
        manifest = json.loads(manifest_raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"verify FAIL: invalid JSON in {manifest_name}: {exc}\n")
        return 1

    if not isinstance(manifest, dict):
        sys.stderr.write(f"verify FAIL: {manifest_name} root must be a JSON object\n")
        return 1
    tar_path = os.path.join(base_dir, tar_name)
    arc = manifest.get("archive")
    if not isinstance(arc, dict):
        sys.stderr.write("verify FAIL: manifest archive field is not an object\n")
        return 1
    if arc.get("sha256") != expected_sums[tar_name] or arc.get("size_bytes") != os.path.getsize(tar_path):
        sys.stderr.write("verify FAIL: manifest archive metadata does not match .tar.gz\n")
        return 1

    inv = manifest.get("files")
    if not isinstance(inv, list) or manifest.get("file_count") != len(inv):
        sys.stderr.write("verify FAIL: manifest file_count mismatch\n")
        return 1
    expected_files = {}
    for entry in inv:
        if not isinstance(entry, dict):
            sys.stderr.write("verify FAIL: manifest inventory entry is not an object\n")
            return 1
        p = entry.get("path")
        if not isinstance(p, str) or not p or p in expected_files:
            sys.stderr.write(f"verify FAIL: invalid or duplicate inventory entry {p!r}\n")
            return 1
        expected_files[p] = (entry.get("size_bytes"), entry.get("sha256"))

    seen_tar = {}
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    sys.stderr.write(f"verify FAIL: unexpected non-regular tar member {member.name!r}\n")
                    return 1
                mname = member.name
                if mname.startswith("/") or mname.startswith("..") or "/../" in mname or has_home_leak(mname):
                    sys.stderr.write(f"verify FAIL: unsafe tar member name {mname!r}\n")
                    return 1
                if mname in seen_tar:
                    sys.stderr.write(f"verify FAIL: duplicate tar member {mname!r}\n")
                    return 1
                fh = tf.extractfile(member)
                if fh is None:
                    sys.stderr.write(f"verify FAIL: unreadable tar member {mname!r}\n")
                    return 1
                data = fh.read()
                if b"\x00" not in data:
                    try:
                        if has_home_leak(data.decode("utf-8")):
                            sys.stderr.write(f"verify FAIL: home-path leak inside tar member {mname!r}\n")
                            return 1
                    except UnicodeDecodeError:
                        pass
                seen_tar[mname] = (len(data), _sha256_bytes(data))
    except (OSError, tarfile.TarError, EOFError, gzip.BadGzipFile) as exc:
        sys.stderr.write(f"verify FAIL: corrupt tar.gz archive: {exc}\n")
        return 1

    if set(seen_tar.keys()) != set(expected_files.keys()):
        missing = sorted(set(expected_files.keys()) - set(seen_tar.keys()))
        extra = sorted(set(seen_tar.keys()) - set(expected_files.keys()))
        sys.stderr.write(f"verify FAIL: inventory set mismatch missing={missing[:5]} extra={extra[:5]}\n")
        return 1

    for path, (exp_sz, exp_sha) in expected_files.items():
        got_sz, got_sha = seen_tar[path]
        if (got_sz, got_sha) != (exp_sz, exp_sha):
            sys.stderr.write(
                f"verify FAIL: member {path} mismatch (expected {exp_sz}/{exp_sha}, got {got_sz}/{got_sha})\n"
            )
            return 1

    print(f"VERDICT verify {name}: PASS ({len(seen_tar)} files verified)")
    return 0


def _rewrite_sums(out_dir, name):
    tar_name = f"{name}.tar.gz"
    manifest_name = f"{name}.manifest.json"
    sums_path = os.path.join(out_dir, f"{name}.SHA256SUMS")
    tar_sha = _sha256_file(os.path.join(out_dir, tar_name))
    man_sha = _sha256_file(os.path.join(out_dir, manifest_name))
    with open(sums_path, "w", encoding="utf-8") as fh:
        fh.write(f"{tar_sha}  {tar_name}\n{man_sha}  {manifest_name}\n")


def self_test():
    """CPU-only self-test with synthetic data, real markdown scan, and 4 deliberate breaks."""
    fails = []

    def check(cond, label):
        print(("PASS " if cond else "FAIL ") + label)
        if not cond:
            fails.append(label)

    # 1. Synthetic markdown extraction
    syn_md = (
        "# Top Heading\n\n"
        "| Run | Notes |\n|---|---|\n"
        "| `runs/20261117T160358Z-haz/` | compiler lane |\n"
        "| `afk-r13/slot2-results/ld_mat2-nopin/` | round 10 |\n\n"
        "## Sub Heading\n\n"
        "Earlier run `20261117T122956Z-memory` and template `runs/<UTC time>-haz/`.\n"
    )
    refs = extract_markdown_run_refs(syn_md, "measurements/synthetic.md")
    got_ids = [r["run_id"] for r in refs]
    check(
        got_ids == [
            "20261117T160358Z-haz",
            "afk-r13/slot2-results/ld_mat2-nopin",
            "20261117T122956Z-memory",
        ],
        f"synthetic markdown extractor got expected run IDs ({got_ids})",
    )
    check(refs[0]["section"] == "Top Heading" and refs[2]["section"] == "Sub Heading", "tracks nearest heading")
    check(refs[0]["line"] == 5 and refs[2]["line"] == 10, "tracks exact 1-based line numbers")

    # Real read-only repo markdown scan if running from repo
    repo_guess = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(repo_guess, "measurements")):
        real_doc = generate_manifest(repo_guess, [])
        real_blob = json.dumps(real_doc, sort_keys=True)
        check(real_doc["reference_count"] > 50 and real_doc["unique_run_count"] > 10,
              f"real read-only repo scan extracted {real_doc['reference_count']} refs across {real_doc['unique_run_count']} runs")
        check(not has_home_leak(real_blob), "real repo manifest has zero home-path leaks")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "run1")
        os.makedirs(os.path.join(run_dir, "sub"), exist_ok=True)
        with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as fh:
            fh.write("RESULT ok=1\n")
        with open(os.path.join(run_dir, "sub", "data.csv"), "w", encoding="utf-8") as fh:
            fh.write("a,b\n1,2\n")

        out_dir = os.path.join(tmp, "arc")
        out_dir2 = os.path.join(tmp, "arc2")
        rc_pack = pack_run(run_dir, out_dir, repo=None, name="testbundle", live_manifest=False)
        rc_pack2 = pack_run(run_dir, out_dir2, repo=None, name="testbundle", live_manifest=False)
        check(rc_pack == 0 and rc_pack2 == 0, "pack round-trip succeeded")
        check(
            _sha256_file(os.path.join(out_dir, "testbundle.tar.gz"))
            == _sha256_file(os.path.join(out_dir2, "testbundle.tar.gz")),
            "deterministic .tar.gz output (identical SHA-256 across two packs)",
        )
        check(verify_bundle(out_dir=out_dir, name="testbundle") == 0, "verify PASS on clean bundle")

        # Parent, equal, and nested out_dir layouts must preserve subdirs and never self-archive.
        rc_parent = pack_run(run_dir, tmp, repo=None, name="parentout", live_manifest=False)
        with open(os.path.join(tmp, "parentout.manifest.json"), "r", encoding="utf-8") as fh:
            parent_files = [e["path"] for e in json.load(fh)["files"]]
        check(
            rc_parent == 0
            and verify_bundle(out_dir=tmp, name="parentout") == 0
            and parent_files == ["sub/data.csv", "summary.txt"],
            f"out_dir as parent of run_dir preserves subdirectories ({parent_files})",
        )

        same_run = os.path.join(tmp, "samedir")
        os.makedirs(os.path.join(same_run, "sub"), exist_ok=True)
        with open(os.path.join(same_run, "summary.txt"), "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        with open(os.path.join(same_run, "sub", "deep.txt"), "w", encoding="utf-8") as fh:
            fh.write("deep\n")
        rc_same = pack_run(same_run, same_run, repo=None, name="sameout", live_manifest=False)
        with open(os.path.join(same_run, "sameout.manifest.json"), "r", encoding="utf-8") as fh:
            same_files = [e["path"] for e in json.load(fh)["files"]]
        check(
            rc_same == 0
            and verify_bundle(out_dir=same_run, name="sameout") == 0
            and same_files == ["sub/deep.txt", "summary.txt"],
            f"out_dir == run_dir preserves subdirectories and excludes bundle files ({same_files})",
        )

        nested_arc = os.path.join(run_dir, "archive")
        os.makedirs(nested_arc, exist_ok=True)
        with open(os.path.join(nested_arc, "pack.log"), "w", encoding="utf-8") as fh:
            fh.write("in-flight log\n")
        with open(os.path.join(run_dir, ".replicate.lock"), "w", encoding="utf-8") as fh:
            fh.write("")
        rc_nested = pack_run(run_dir, nested_arc, repo=None, name="nested", live_manifest=False)
        with open(os.path.join(nested_arc, "nested.manifest.json"), "r", encoding="utf-8") as fh:
            nested_files = [e["path"] for e in json.load(fh)["files"]]
        check(
            rc_nested == 0
            and verify_bundle(out_dir=nested_arc, name="nested") == 0
            and nested_files == ["sub/data.csv", "summary.txt"],
            f"nested out_dir inside run_dir excludes archive/ and .replicate.lock ({nested_files})",
        )

        # Deliberate break 1: tamper 1 byte in .tar.gz -> verify RED, restore -> PASS
        tar_path = os.path.join(out_dir, "testbundle.tar.gz")
        with open(tar_path, "rb") as fh:
            orig_tar = fh.read()
        tampered = bytearray(orig_tar)
        tampered[-1] ^= 0xFF
        with open(tar_path, "wb") as fh:
            fh.write(tampered)
        check(verify_bundle(out_dir=out_dir, name="testbundle") != 0,
              "deliberate break 1 (1-byte tar tamper): verify returns nonzero (RED)")
        with open(tar_path, "wb") as fh:
            fh.write(orig_tar)
        check(verify_bundle(out_dir=out_dir, name="testbundle") == 0,
              "deliberate break 1 restored: verify returns 0 (GREEN)")

        # Deliberate break 2: inventory SHA-256 mismatch in manifest (checksums updated) -> verify RED
        man_path = os.path.join(out_dir, "testbundle.manifest.json")
        with open(man_path, "r", encoding="utf-8") as fh:
            orig_man = fh.read()
        bad_man = json.loads(orig_man)
        bad_man["files"][0]["sha256"] = "0" * 64
        with open(man_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(bad_man, indent=1, sort_keys=True) + "\n")
        _rewrite_sums(out_dir, "testbundle")
        check(verify_bundle(out_dir=out_dir, name="testbundle") != 0,
              "deliberate break 2 (inventory sha256 mismatch): verify returns nonzero (RED)")
        with open(man_path, "w", encoding="utf-8") as fh:
            fh.write(orig_man)
        _rewrite_sums(out_dir, "testbundle")
        check(verify_bundle(out_dir=out_dir, name="testbundle") == 0,
              "deliberate break 2 restored: verify returns 0 (GREEN)")

        # Deliberate break 3: real leak checks on _write_json_strict, verify_bundle (manifest + tar), and pack_run
        sample_leak = _HOME_PRE + "alice/secret"
        leak_json_path = os.path.join(tmp, "should-not-exist.json")
        rc_leak_write = _write_json_strict({"leaked": sample_leak}, leak_json_path)
        check(
            rc_leak_write != 0 and not os.path.exists(leak_json_path),
            "deliberate break 3a (_write_json_strict): refuses JSON containing home path (RED)",
        )

        leak_man = json.loads(orig_man)
        leak_man["note"] = sample_leak
        with open(man_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(leak_man, indent=1, sort_keys=True) + "\n")
        _rewrite_sums(out_dir, "testbundle")
        check(
            verify_bundle(out_dir=out_dir, name="testbundle") != 0,
            "deliberate break 3b (manifest home leak): verify returns nonzero (RED)",
        )
        with open(man_path, "w", encoding="utf-8") as fh:
            fh.write(orig_man)
        _rewrite_sums(out_dir, "testbundle")
        check(
            verify_bundle(out_dir=out_dir, name="testbundle") == 0,
            "deliberate break 3b restored: verify returns 0 (GREEN)",
        )

        # Pack a run containing a raw home path: default scrubs text member (PASS); scrub_text=False refuses (RED);
        # direct raw tar injection fails verify_bundle (RED).
        leak_run = os.path.join(tmp, "leakrun")
        leak_arc = os.path.join(tmp, "leakarc")
        os.makedirs(leak_run, exist_ok=True)
        with open(os.path.join(leak_run, "log.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"cmd: {sample_leak}/run.sh\n")
        check(
            pack_run(leak_run, leak_arc, repo=None, name="unscrubbed", live_manifest=False, scrub_text=False) != 0,
            "deliberate break 3c (pack scrub_text=False): refuses raw home path in text member (RED)",
        )
        check(
            pack_run(leak_run, leak_arc, repo=None, name="scrubbed", live_manifest=False, scrub_text=True) == 0
            and verify_bundle(out_dir=leak_arc, name="scrubbed") == 0,
            "pack_run scrubs home path inside text member and verify passes (GREEN)",
        )
        raw_tar = os.path.join(leak_arc, "scrubbed.tar.gz")
        raw_man = os.path.join(leak_arc, "scrubbed.manifest.json")
        raw_bytes = f"cmd: {sample_leak}/run.sh\n".encode("utf-8")
        with open(raw_tar, "wb") as rf:
            with gzip.GzipFile(filename="", mode="wb", fileobj=rf, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf_out:
                    ti = tarfile.TarInfo(name="log.txt")
                    ti.size = len(raw_bytes)
                    ti.mtime = 0
                    tf_out.addfile(ti, io.BytesIO(raw_bytes))
        rm_doc = json.loads(open(raw_man, encoding="utf-8").read())
        rm_doc["archive"]["sha256"] = _sha256_file(raw_tar)
        rm_doc["archive"]["size_bytes"] = os.path.getsize(raw_tar)
        rm_doc["files"] = [{"path": "log.txt", "size_bytes": len(raw_bytes), "sha256": _sha256_bytes(raw_bytes)}]
        rm_doc["total_bytes"] = len(raw_bytes)
        with open(raw_man, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rm_doc, indent=1, sort_keys=True) + "\n")
        _rewrite_sums(leak_arc, "scrubbed")
        check(
            verify_bundle(out_dir=leak_arc, name="scrubbed") != 0,
            "deliberate break 3d (raw home path injected in tar member): verify returns nonzero (RED)",
        )

        # Deliberate break 4: remove a member from tar (and update manifest archive hash + sums) -> verify RED
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf_out:
                with tarfile.open(tar_path, "r:gz") as tf_in:
                    members = tf_in.getmembers()
                    for m in members[1:]:
                        tf_out.addfile(m, tf_in.extractfile(m))
        with open(tar_path, "wb") as fh:
            fh.write(buf.getvalue())
        man_drop = json.loads(orig_man)
        man_drop["archive"]["sha256"] = _sha256_file(tar_path)
        man_drop["archive"]["size_bytes"] = os.path.getsize(tar_path)
        with open(man_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(man_drop, indent=1, sort_keys=True) + "\n")
        _rewrite_sums(out_dir, "testbundle")
        check(verify_bundle(out_dir=out_dir, name="testbundle") != 0,
              "deliberate break 4 (missing tar member): verify returns nonzero (RED)")

    print("VERDICT archive_runs self-test " + ("PASS" if not fails else "FAIL") + f" ({len(fails)} failed)")
    return 0 if not fails else 1


def main(argv):
    if argv == ["--self-test"]:
        return self_test()

    p = argparse.ArgumentParser(description="Raw-run archiver and table-to-run manifest generator (stdlib only).")
    p.add_argument("--self-test", action="store_true", help="run CPU-only self-test with deliberate breaks")
    sub = p.add_subparsers(dest="cmd")

    pm = sub.add_parser("manifest", help="scan markdown tables/sections and resolve run directories")
    pm.add_argument("--repo", required=True, help="repository root to scan")
    pm.add_argument("--out", required=True, help="output JSON path")
    pm.add_argument("--runs-root", action="append", default=[], help="additional runs root directory (repeatable)")

    pp = sub.add_parser("pack", help="create deterministic tar.gz, manifest.json, and SHA256SUMS")
    pp.add_argument("--run-dir", required=True, help="run directory to archive")
    pp.add_argument("--out-dir", required=True, help="output directory for bundle files")
    pp.add_argument("--repo", help="optional git repository root for commit/dirty metadata")
    pp.add_argument("--name", help="bundle base name (default: basename of --run-dir)")
    pp.add_argument("--exclude", action="append", default=[], help="top-level file or directory name inside --run-dir to exclude (repeatable)")
    pp.add_argument("--no-scrub-text", action="store_true", help="refuse instead of scrubbing home paths in UTF-8 text members")
    pp.add_argument("--no-live-manifest", action="store_true", help="skip live env_manifest.py collection (CPU-only)")

    pv = sub.add_parser("verify", help="verify bundle SHA256SUMS, manifest, and tar inventory")
    pv.add_argument("--out-dir", help="directory holding bundle files")
    pv.add_argument("--name", help="bundle base name")
    pv.add_argument("--checksums", help="path to <name>.SHA256SUMS")

    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.cmd == "manifest":
        doc = generate_manifest(args.repo, args.runs_root)
        return _write_json_strict(doc, args.out)
    if args.cmd == "pack":
        return pack_run(
            args.run_dir,
            args.out_dir,
            repo=args.repo,
            name=args.name,
            live_manifest=not args.no_live_manifest,
            exclude_top=args.exclude,
            scrub_text=not args.no_scrub_text,
        )
    if args.cmd == "verify":
        return verify_bundle(out_dir=args.out_dir, name=args.name, checksums_path=args.checksums)
    p.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
