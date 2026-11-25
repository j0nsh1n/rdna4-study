"""Shared helpers of the round 9 analyses (standard library only). Loaded by path from each analyse_*.py, which runs under
`python3 -I` (isolated mode puts no script folder on sys.path)."""
import math


def kvs(line):
    """key=value tokens of a result line, as a dict of strings."""
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def pct(v, p):
    """p in 0..100, linear interpolation."""
    s = sorted(v)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def window_label(samp, t0, t1, min_samples=3, share=0.90):
    """The memory clock the sampler read DURING a window: (level, "too short" or "mixed", share, n samples, histogram) from the
    sampler rows (mono_ns, uclk_cur_MHz) inside [t0, t1]. "too short": fewer than min_samples rows in the window (a window of a
    few ms at a 5 ms sampler holds 0-2 rows; it says nothing about whether the clock moved). "mixed": enough rows, but less than
    `share` of them in one level (the clock did move inside the window)."""
    hist = {}
    for t, u in zip(samp["mono_ns"], samp["uclk_cur_MHz"]):
        if t0 <= t <= t1 and u > 0:
            hist[int(u)] = hist.get(int(u), 0) + 1
    n = sum(hist.values())
    if n < min_samples:
        return "too short", float("nan"), n, hist
    lev, cnt = max(hist.items(), key=lambda kv: kv[1])
    return (lev if cnt / n >= share else "mixed"), cnt / n, n, hist


def spread(v):
    """(max - min) / median."""
    m = median(v)
    return (max(v) - min(v)) / m if v and m else float("nan")


class Verdicts:
    """Collects PASS / FAIL / SKIP lines; `ok` is False once any FAIL was added."""

    def __init__(self, item):
        self.item = item
        self.rows = []

    def add(self, name, ok, text):
        """ok: True, False, or None (not judged: no data for it)."""
        self.rows.append((name, ok, text))
        tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print("VERDICT %s %s: %s %s" % (self.item, name, tag, text))

    @property
    def fails(self):
        return [r for r in self.rows if r[1] is False]

    @property
    def skips(self):
        return [r for r in self.rows if r[1] is None]

    def summary(self):
        p = sum(1 for r in self.rows if r[1])
        print("SUMMARY %s: %d PASS, %d FAIL, %d not judged" % (self.item, p, len(self.fails), len(self.skips)))
        return not self.fails


def run_selftest(item, cases, silent=lambda f: f()):
    """cases: list of (name, expect_ok, func returning a Verdicts[, prefix]). Output of the cases themselves is swallowed.
    A case that must PASS has to return no FAIL; a break (expect_ok False) has to return a FAIL, and when a verdict-name prefix
    is given, that very verdict has to be among the FAILs (a break that fails for another reason does not count)."""
    import contextlib
    import io

    bad = 0
    for case in cases:
        name, expect_ok, fn = case[:3]
        prefix = case[3] if len(case) > 3 else None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            v = fn()
        got_ok = not v.fails
        good = got_ok == expect_ok
        if prefix and not expect_ok:
            good = good and any(r[0].startswith(prefix) for r in v.fails)
        print("SELFTEST %s %s: expected %s, got %s (%s)" % (item, name, "PASS" if expect_ok else "FAIL",
                                                           "PASS" if got_ok else "FAIL", "ok" if good else "WRONG"))
        if not good:
            bad += 1
            print("\n".join("    " + l for l in buf.getvalue().splitlines() if "FAIL" in l or "PASS" in l))
    print("SELFTEST %s: %s" % (item, "PASS" if not bad else "FAIL (%d wrong)" % bad))
    return bad == 0
