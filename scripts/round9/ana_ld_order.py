#!/usr/bin/env python3
"""Analysis of ld_order (round 9, item B). Standard library only; run with python3 -I.

Usage: ana_ld_order.py <log of ld_order run> [<more logs>...]     one log per run; the controls are judged on each
       ana_ld_order.py --selftest                                 synthetic inputs, with deliberate breaks that must FAIL
Prints one line per control: "CONTROL <name>: PASS|FAIL <numbers>", then "ORDER <verdict>" and "SUMMARY ...".
Exit 1 if any control fails or the order question cannot be answered.

Costs are cycles of the in-kernel counter between the start of the pattern and the snapshot after its last wait, minus
the pattern `empty` (two counter reads back to back). Expected numbers are in expect-ld_order.md, written before the run.
"""
import re
import sys

# Corrected after the first run: the draft band was 124-140 (129-135 cycles), taken from memory-system.md section 5, which times a
# repeating loop (address arithmetic and loop overhead in every batch). This probe times one load and its wait with two counter
# reads, the form of matrix-timing.md section 8a (77 cycles for a 4-byte L0 hit with its wait, line 354). Band: 77 +- 8.
HIT = (69.0, 85.0)
MISS = (550.0, 770.0)   # draft: 550-770 cycles for a pure GDDR6 miss
WRONG_SHARE = 0.90      # a deliberate break must show wrong words in at least this share of the lane checks
TOL_ABS, TOL_REL = 25.0, 0.12


def parse(text):
    rows = {}
    for ln in text.splitlines():
        if not ln.startswith("RESULT ld_order "):
            continue
        t = ln.split()
        name = t[2]
        d = {"name": name}
        i = 3
        while i < len(t):
            k = t[i]
            if k in ("c1", "c2"):
                d[k] = tuple(float(x) for x in t[i + 1:i + 4])
                i += 4
            elif k in ("ok1", "ok2"):
                d[k] = [int(x) for x in t[i + 1:i + 8]]
                i += 8
            elif k == "kinds":
                d[k] = t[i + 1]
                i += 2
            else:
                try:
                    d[k] = float(t[i + 1])
                except (IndexError, ValueError):
                    d[k] = t[i + 1] if i + 1 < len(t) else None
                i += 2
        rows[name] = d
    return rows


def cost(rows, name, snap=1):
    r = rows[name]
    return r["c%d" % snap][0] - rows["empty"]["c1"][0]


def lat_of(kind, lat):
    return lat.get(kind, 0.0)


def judge(rows):
    out = []   # (name, ok, text)

    def ctl(name, ok, text):
        out.append((name, bool(ok), text))

    # the verdict needs every pattern the question rests on: both deliberate-break rows and all 12 planned mixes
    mixes = ["%s_w%d" % (t, w) for t in ("mhhhh", "hhhhm", "hmhmh") for w in (1, 2, 3, 4)]
    need = ["empty", "pure_hit", "pure_miss", "mh_w1", "hm_w1", "mh_w0", "hm_w0", "brk_wrongcnt", "brk_wrongkm", "gl_dswait", "lds_only",
            "s_hit", "s_far", "m_s_kmwait"] + mixes
    missing = [n for n in need if n not in rows]
    if missing:
        ctl("complete", False, "patterns missing from the log: " + " ".join(missing))
        return out, "UNANSWERED"
    hit, miss = cost(rows, "pure_hit"), cost(rows, "pure_miss")
    ctl("pure_hit", HIT[0] <= hit <= HIT[1], "one L0 hit costs %.1f cycles (measured earlier: 77 in matrix-timing.md 8a; draft was 129-135; gate %.0f-%.0f)" % (hit, *HIT))
    ctl("pure_miss", MISS[0] <= miss <= MISS[1], "one GDDR6 miss costs %.1f cycles (draft: 550-770)" % miss)
    # bad words: every register the claim says must hold data, at every snapshot, in every lane of every repetition
    wrong = checks = 0
    for n, r in rows.items():
        if n.startswith("brk_") or n == "empty":
            continue
        lanes = int(r["lanes_per_check"])
        for s in (1, 2):
            if s > int(r["nsnap"]):
                continue
            must = int(r["must%d" % s])
            for i in range(7):
                if must >> i & 1:
                    checks += lanes
                    wrong += lanes - r["ok%d" % s][i]
    ctl("bad_zero", wrong == 0 and checks > 0, "%d wrong words in %d lane checks of registers that must hold data" % (wrong, checks))
    # the deliberate breaks: a wait on the wrong counter, then a read of the other counter's register
    for n in ("brk_wrongcnt", "brk_wrongkm"):
        if n not in rows:
            continue
        r = rows[n]
        lanes = int(r["lanes_per_check"])
        must = int(r["must1"])
        bad = tot = 0
        for i in range(7):
            if must >> i & 1 and r["kinds"][i] in "MF":  # the far register the wrong wait did not cover
                tot += lanes
                bad += lanes - r["ok1"][i]
        share = bad / tot if tot else 0.0
        ctl("break_" + n[4:], share >= WRONG_SHARE, "the wrong-counter wait shows wrong words in %.1f %% of %d lane checks (must be >= %d %%: else the probe cannot see an early return)" % (100 * share, tot, WRONG_SHARE * 100))
    # the order question
    mh1, hm1 = cost(rows, "mh_w1"), cost(rows, "hm_w1")
    in_order = mh1 >= 0.85 * miss and hm1 <= hit * 1.25 + TOL_ABS
    ooo = mh1 <= hit * 1.25 + TOL_ABS
    equal = abs(mh1 - hm1) <= 0.15 * max(mh1, hm1)
    ctl("order_probe_can_tell", not equal and (in_order or ooo), "[miss, hit] wait 1 costs %.1f, [hit, miss] wait 1 costs %.1f (a probe that cannot tell them apart answers nothing)" % (mh1, hm1))
    order = "IN ORDER" if in_order else ("OUT OF ORDER: STOP AND REPORT (the first two cost the same or the hit overtakes the miss)" if equal or ooo else "UNCLEAR")
    # longer mixes against the in-order and the out-of-order model
    lat = {"H": hit, "M": miss}
    fits_in = fits_out = n_mix = 0
    lines = []
    for n, r in sorted(rows.items()):
        m = re.match(r"^(mhhhh|hhhhm|hmhmh)_w(\d)$", n)
        if not m:
            continue
        w = int(m.group(2))
        seq = [k for k in r["kinds"] if k in "MH"]
        first = seq[: len(seq) - w]
        t_in = max(lat[k] for k in first)
        t_out = sorted(lat[k] for k in seq)[len(seq) - w - 1]
        c = cost(rows, n)
        tol = max(TOL_ABS, TOL_REL * max(t_in, t_out))
        n_mix += 1
        a, b = abs(c - t_in) <= tol, abs(c - t_out) <= tol
        fits_in += a
        fits_out += b
        lines.append("%s %.0f (in order %.0f, out of order %.0f)" % (n, c, t_in, t_out))
    if n_mix:
        ctl("mix_in_order", fits_in == n_mix, "%d of %d five-load mixes within max(%d, %d %%) of the in-order model; %d of %d fit the out-of-order model: %s" % (fits_in, n_mix, TOL_ABS, TOL_REL * 100, fits_out, n_mix, "; ".join(lines)))
    # the two [miss, hit] / [hit, miss] wait 0 patterns cost the miss
    ctl("wait0_costs_miss", cost(rows, "mh_w0") >= 0.85 * miss and cost(rows, "hm_w0") >= 0.85 * miss, "wait 0 after [miss, hit] %.1f and [hit, miss] %.1f (miss %.1f)" % (cost(rows, "mh_w0"), cost(rows, "hm_w0"), miss))
    # counters are independent: the LDS wait returns long before the miss, the loadcnt wait after it
    g1, g2 = cost(rows, "gl_dswait", 1), cost(rows, "gl_dswait", 2)
    lds = cost(rows, "lds_only")
    ctl("dscnt_independent", g1 <= 0.4 * miss and g2 >= 0.8 * miss, "miss + LDS: wait dscnt returns at %.1f (LDS alone %.1f), wait loadcnt at %.1f (miss %.1f)" % (g1, lds, g2, miss))
    k1, k2 = cost(rows, "m_s_kmwait", 1), cost(rows, "m_s_kmwait", 2)
    sh, sf = cost(rows, "s_hit"), cost(rows, "s_far")
    ctl("kmcnt_independent", k1 <= 0.4 * miss and k2 >= 0.8 * miss, "miss + scalar hit: wait kmcnt returns at %.1f (scalar hit alone %.1f), wait loadcnt at %.1f (far scalar alone %.1f)" % (k1, sh, k2, sf))
    # the register the wrong wait skipped: how many lanes already hold the far data (information)
    info, far_lanes = [], []
    for n in ("gl_dswait", "m_s_kmwait"):
        r = rows[n]
        lanes = int(r["lanes_per_check"])
        stale = int(r["stale1"])
        for i in range(7):
            if stale >> i & 1:
                far_lanes.append(int(r["ok1"][i]))
                info.append("%s register %d: far data present in %.1f %% of lanes at the first snapshot" % (n, i, 100.0 * r["ok1"][i] / lanes))
    if info:  # a number, not text: "0.0 %" is a substring of "100.0 %"
        ctl("stale_visible", all(c == 0 for c in far_lanes), "; ".join(info))
    dropped = sum(int(r.get("dropped", 0)) for r in rows.values())
    ctl("disturbance", dropped <= 0.05 * sum(int(r["reps"]) for r in rows.values()), "%d samples dropped by the disturbance rule" % dropped)
    return out, order


def report(rows, label=""):
    res, order = judge(rows)
    bad = 0
    for n, ok, t in res:
        print("CONTROL %s%s: %s %s" % (label, n, "PASS" if ok else "FAIL", t))
        bad += not ok
    print("ORDER %s%s" % (label, order))
    print("SUMMARY %s%d controls, %d failed; order verdict: %s" % (label, len(res), bad, order))
    return bad, order


# ---------------------------------------------------------------------------------------------------------------
# synthetic input


def synth(world="in", brk_sees=True, bad_words=False, miss=650.0, hit=77.0, base=6.0, stale_far=0.0, drop=()):
    """RESULT lines of a made-up card. world: 'in' (loads retire in order), 'out' (a hit overtakes a miss)."""
    lines = []
    lanes = 20 * 3 * 32
    reps = 60

    def row(name, tokens, c1, c2=0.0, kinds="-------", nsnap=1, must=(0, 0), stale=(0, 0), ok1=None, ok2=None):
        ok1 = ok1 or [lanes if (must[0] >> i & 1) else 0 for i in range(7)]
        ok2 = ok2 or [lanes if (must[1] >> i & 1) else 0 for i in range(7)]
        lines.append(
            "RESULT ld_order %s reps %d c1 %.1f %.1f %.1f c2 %.1f %.1f %.1f dropped 0 base %.1f nsnap %d kinds %s ok1 %s must1 %d stale1 %d ok2 %s must2 %d stale2 %d lanes_per_check %d"
            % (name, reps, c1 + base, c1 + base - 2, c1 + base + 2, c2 + base, c2 + base - 2, c2 + base + 2, base, nsnap, kinds,
               " ".join(map(str, ok1)), must[0], stale[0], " ".join(map(str, ok2)), must[1], stale[1], lanes))

    row("empty", "", 0)
    row("pure_hit", "", hit, kinds="H------", must=(1, 0))
    row("pure_miss", "", miss, kinds="M------", must=(1, 0))

    def t_wait(seq, w):
        """cycles until `wait w` returns for a sequence of loads (list of 'M'/'H')."""
        L = {"H": hit, "M": miss}
        n = len(seq) - w
        if n <= 0:
            return 0.0
        if world == "in":
            return max(L[k] for k in seq[:n])
        return sorted(L[k] for k in seq)[n - 1]

    def must_mask(seq, w, world_):
        n = len(seq) - w
        return (1 << n) - 1 if n > 0 else 0

    def pat(name, seq, w):
        kinds = "".join(seq).ljust(7, "-")
        mk = must_mask(seq, w, world)
        # out of order: the register of a not-yet-returned miss really is stale although the claim says it must be valid
        ok = [lanes if (mk >> i & 1) else 0 for i in range(7)]
        if world == "out":
            for i, k in enumerate(seq):
                if mk >> i & 1 and k == "M" and t_wait(seq, w) < miss:
                    ok[i] = 0
        if bad_words and name == "mh_w0":
            ok[0] = lanes // 2
        row(name, "", t_wait(seq, w), kinds=kinds, must=(mk, 0), ok1=ok, stale=(0, 0))

    pat("mh_w1", list("MH"), 1)
    pat("hm_w1", list("HM"), 1)
    pat("mh_w0", list("MH"), 0)
    pat("hm_w0", list("HM"), 0)
    for tag, seq in (("mhhhh", "MHHHH"), ("hhhhm", "HHHHM"), ("hmhmh", "HMHMH")):
        for w in (1, 2, 3, 4):
            pat("%s_w%d" % (tag, w), list(seq), w)
    lds, sh, sf = 70.0, 62.0, 700.0
    row("lds_only", "", lds, kinds="-----L-", must=(32, 0))
    row("s_hit", "", sh, kinds="------S", must=(64, 0))
    row("s_far", "", sf, kinds="------F", must=(64, 0))
    far = int(round(stale_far * lanes))  # lanes of the skipped register already holding the far data at the first snapshot
    row("gl_dswait", "", lds, miss, kinds="M----L-", nsnap=2, must=(32, 33), stale=(1, 0),
        ok1=[far, 0, 0, 0, 0, lanes, 0], ok2=[lanes, 0, 0, 0, 0, lanes, 0])
    row("m_s_kmwait", "", sh, miss, kinds="M-----S", nsnap=2, must=(64, 65), stale=(1, 0),
        ok1=[far, 0, 0, 0, 0, 0, lanes], ok2=[lanes, 0, 0, 0, 0, 0, lanes])
    row("s_m_lwait", "", miss, kinds="M-----S", must=(65, 0), ok1=[lanes, 0, 0, 0, 0, 0, lanes])
    seen = lanes if brk_sees else 0
    # the break claims register 0 (and the other counter's register) valid; the probe sees it stale in every lane
    row("brk_wrongcnt", "", lds, kinds="M----L-", must=(33, 0), ok1=[lanes - seen, 0, 0, 0, 0, lanes, 0])
    row("brk_wrongkm", "", sh, kinds="M-----S", must=(65, 0), ok1=[lanes - seen, 0, 0, 0, 0, 0, lanes])
    return "\n".join(ln for ln in lines if not any(ln.startswith("RESULT ld_order %s " % d) for d in drop)) + "\n"


def selftest():
    ok = True

    def run(label, text, expect_fail, expect_order=None, expect_ctl=None):
        nonlocal ok
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bad, order = report(parse(text))
        failed = bad > 0 or (expect_order is not None and not order.startswith(expect_order))
        good = failed == expect_fail
        if expect_ctl:  # the break must fail for its own reason, not another
            good = good and any(n == expect_ctl and not c for n, c, _ in judge(parse(text))[0])
        print("selftest %s: %s (%s)" % (label, "OK" if good else "WRONG", "controls failed: %d, order: %s" % (bad, order)))
        if not good:
            sys.stdout.write(buf.getvalue())
        ok &= good

    run("in-order card, every control passes", synth(), False, "IN ORDER")
    run("deliberate break: wrong-counter wait shows no wrong words (the probe is blind)", synth(brk_sees=False), True)
    run("deliberate break: the hit overtakes the miss (out of order card)", synth(world="out"), True)
    run("deliberate break: wrong words in a correct wait", synth(bad_words=True), True)
    run("deliberate break: miss costs the hit's time", synth(miss=140.0), True)
    run("deliberate break: far data already in every lane of the skipped register (100 %), the old substring check passed this",
        synth(stale_far=1.0), True, expect_ctl="stale_visible")
    run("deliberate break: far data in 10 % of the lanes of the skipped register", synth(stale_far=0.1), True, expect_ctl="stale_visible")
    run("deliberate break: the second break row is missing (no verdict)", synth(drop=("brk_wrongkm",)), True, "UNANSWERED", "complete")
    run("deliberate break: one of the 12 mixes is missing (no verdict)", synth(drop=("hmhmh_w4",)), True, "UNANSWERED", "complete")
    return 0 if ok else 1


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 2
    if a[0] == "--selftest":
        return selftest()
    bad_total = 0
    for p in a:
        rows = parse(open(p).read())
        bad, order = report(rows, label=("%s: " % p) if len(a) > 1 else "")
        bad_total += bad or order not in ("IN ORDER",)
    return 1 if bad_total else 0


if __name__ == "__main__":
    sys.exit(main())
