#!/usr/bin/env python3
"""ISA check of the two kernels of src/pair_overlap.hip (round 3, Q5): the clock reads must sit where the probe's argument needs them.
Usage: llm3_pair_isacheck.py <file.lto.s> [--break order|count]
  pair_a: first s_sendmsg_rtn_b64 (REALTIME) before the first global store; s_wait_storecnt after the last global store and before the LAST
          s_sendmsg_rtn_b64; exactly two REALTIME reads; exactly one global_store_b128 in the store loop (plus the two b64 stamp stores after the last read)
  pair_b: first REALTIME read before the first global load; exactly two REALTIME reads; the last after the loop and the atomic
--break order: the expectation is changed to 'the last stamp comes BEFORE the wait for the stores', which the code does not do: must FAIL (exit 1)
--break count: the expectation is three REALTIME reads in pair_a, the code has two: must FAIL
Exit 0 all hold, 1 a check fails, 2 usage."""
import re
import sys


def kernel_lines(text, name):
    out, on = [], False
    for ln in text.splitlines():
        if re.match(r"^_Z\d+" + name + r"\w*:", ln):
            on = True
            continue
        if on and ln.strip().startswith("s_endpgm"):
            break
        if on and re.match(r"^\s+[a-z_0-9]+", ln) and not ln.strip().startswith("."):
            out.append(ln.strip().split()[0])
    return out


def idx(ins, op, last=False):
    hits = [i for i, o in enumerate(ins) if o == op]
    return (hits[-1] if last else hits[0]) if hits else -1


def main():
    a = sys.argv[1:]
    if not a or a[0].startswith("-"):
        print(__doc__)
        return 2
    brk = a[a.index("--break") + 1] if "--break" in a else ""
    text = open(a[0]).read()
    A, B = kernel_lines(text, "pair_a"), kernel_lines(text, "pair_b")
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + name)
        fails += not ok

    if not A or not B:
        print("FAIL kernels pair_a and pair_b not found")
        return 1
    rtn = "s_sendmsg_rtn_b64"
    check("pair_a: %d REALTIME reads (expected %d)" % (A.count(rtn), 3 if brk == "count" else 2), A.count(rtn) == (3 if brk == "count" else 2))
    check("pair_a: first REALTIME read before the first global store", idx(A, rtn) < idx(A, "global_store_b128") and idx(A, rtn) >= 0)
    last_store = max(i for i, o in enumerate(A) if o == "global_store_b128")
    wait = [i for i, o in enumerate(A) if o == "s_wait_storecnt" and i > last_store]
    last_rtn = idx(A, rtn, last=True)
    if brk == "order":
        check("pair_a: the last REALTIME read comes BEFORE the wait for the stores (deliberately wrong expectation)", bool(wait) and last_rtn < wait[0])
    else:
        check("pair_a: s_wait_storecnt after the last store and before the last REALTIME read", bool(wait) and wait[0] < last_rtn)
    check("pair_a: one global_store_b128 in the store loop", A.count("global_store_b128") == 1)
    check("pair_b: %d REALTIME reads (expected 2)" % B.count(rtn), B.count(rtn) == 2)
    check("pair_b: first REALTIME read before the first global load", 0 <= idx(B, rtn) < idx(B, "global_load_b128"))
    check("pair_b: last REALTIME read after the global load and the atomic", idx(B, rtn, last=True) > idx(B, "global_atomic_add_u32"))
    print("VERDICT pair_overlap-isa " + ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
