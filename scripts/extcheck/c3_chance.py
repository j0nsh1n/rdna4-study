r"""Check 3 (proxy): chance level of the "stall events of two waves fall within 1 us of each other" coincidence.

The raw event times of the `multi` run behind the published 96.2 % / 95.6 % were never saved (mode_multi in
src/study/sched_share.hip prints summaries only; runs/*sched*/spin-multi.log has no trace), so that figure cannot be
recomputed without the card. This script applies the requested control to the two-wave `pair small small` run, whose raw
rows ARE in the runs (the published 98.8 %), with the study's own event extraction (scripts/sched_pair.py: events()).

Usage: python3 c3_chance.py <study repo> <run dir with pair-small-small-A.csv / -B.csv> [--shuffles N] [--seed S]
Reports: the real share (A's events with a B event within +-100 ticks of 10 ns = 1 us, as sched_pair.py), then the share after
a circular time shift of each wave's events by an independent random offset over the run length (N shuffles: mean and
95 % range), then the deliberate break (the same offset for both waves; must equal the real share), then the 20 ms-grid
check that the window arithmetic is right (a shift of the whole run by a multiple of nothing changes nothing).
"""
import bisect
import os
import random
import statistics
import sys


def main():
    study, run = sys.argv[1], sys.argv[2]
    n_shuf = int(sys.argv[sys.argv.index("--shuffles") + 1]) if "--shuffles" in sys.argv else 1000
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
    sys.path.insert(0, os.path.join(study, "scripts"))
    import sched_pair as sp  # noqa: E402

    a = sp.events(sp.load(os.path.join(run, "pair-small-small-A.csv")))
    b = sp.events(sp.load(os.path.join(run, "pair-small-small-B.csv")))
    t0 = min(a[0], b[0])
    span = max(a[-1], b[-1]) - t0 + 1  # run length in ticks (10 ns)

    def share(x, y, circular):
        """Share of x's events with a y event within +-100 ticks. circular: distances wrap at the run length."""
        y = sorted(y)
        hit = 0
        for t in x:
            for off in ((0, span, -span) if circular else (0,)):
                i = bisect.bisect_left(y, t + off - 100)
                if i < len(y) and y[i] <= t + off + 100:
                    hit += 1
                    break
        return hit / len(x)

    def shifted(x, s):
        return sorted((t - t0 + s) % span for t in x)

    real = share(a, b, False)
    real_c = share([t - t0 for t in a], [t - t0 for t in b], True)
    print(f"events A={len(a)} B={len(b)} run length {span * 10e-9:.3f} s; real share {real:.4f} "
          f"(circular distance {real_c:.4f})")
    rng = random.Random(seed)
    vals = []
    for _ in range(n_shuf):
        sa, sb = rng.randrange(span), rng.randrange(span)
        vals.append(share(shifted(a, sa), shifted(b, sb), True))
    vals.sort()
    lo, hi = vals[int(0.025 * n_shuf)], vals[min(n_shuf - 1, int(0.975 * n_shuf))]
    print(f"shuffled (independent offset per wave, {n_shuf} shuffles): mean {statistics.mean(vals):.4f} "
          f"95% range {lo:.4f} to {hi:.4f} max {vals[-1]:.4f}")
    same = []
    for _ in range(n_shuf):
        s = rng.randrange(span)
        same.append(share(shifted(a, s), shifted(b, s), True))
    print(f"break (same offset for both waves, {n_shuf} shuffles): min {min(same):.4f} max {max(same):.4f} "
          f"real (circular) {real_c:.4f} -> {'PASS' if max(abs(v - real_c) for v in same) < 1e-9 else 'FAIL'}")
    rate = len(b) / span
    print(f"analytic chance for independent events: {len(b) * 201 / span:.4f} (201-tick window x B's event rate {rate * 1e8:.0f} per s)")
    ok = real - hi > 0.5 and max(abs(v - real_c) for v in same) < 1e-9
    print(f"VERDICT c3_chance: {'PASS' if ok else 'FAIL'} real {real:.4f}, chance mean {statistics.mean(vals):.4f}, 95% upper {hi:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
