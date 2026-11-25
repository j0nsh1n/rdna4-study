#!/usr/bin/env bash
# CPU-only self-test of the item E background loop's checks (no GPU, no sudo): (1) the throttle logic of r9_bg
# (r9_bg_throttle.h, compiled with g++ and run against a simulated heating card, plus its deliberate break -DBG_BREAK, which must
# fail); (2) e_bg_inner.sh against stub binaries: a background loop that keeps running must give exit 0 and three
# "INFO bg alive" lines; one that exits at once (the first version's behaviour) must give exit 13 and "INFO bg DEAD" lines.
# Usage: scripts/round9/selftest_e_bg.sh        Exit 0 only if every part passes.
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
t=$repo/build/round9/tmp-selftest-e-bg
rm -rf "$t" && mkdir -p "$t/alive" "$t/dead" "$t/out"
rc=0
g++ -O2 -std=c++17 -I"$repo/src/round9" "$repo/src/round9/r9_bg_selftest.cpp" -o "$t/ok" || exit 2
g++ -O2 -std=c++17 -DBG_BREAK -I"$repo/src/round9" "$repo/src/round9/r9_bg_selftest.cpp" -o "$t/broken" || exit 2
"$t/ok" >"$t/ok.txt" && echo "PASS throttle logic (heating card: no tripwire, loop alive for 30 s)" || { echo "FAIL throttle logic"; cat "$t/ok.txt"; rc=1; }
"$t/broken" >"$t/broken.txt" && { echo "FAIL the deliberate break (no pausing) was NOT caught"; rc=1; } || echo "PASS break caught: an unthrottled loop fails the heating simulation"
mk() { # mk <dir> <alive|dead>
    cat >"$1/r9_bg" <<STUB
#!/usr/bin/env bash
ready=\$2
touch "\$ready"
[[ $2 == dead ]] && { echo "r9_bg: 606 launches in 2.15 s, hotspot 84 C"; exit 0; }
n=0; trap 'exit 0' TERM
while true; do n=\$((n+50)); echo \$n >"\$ready.count"; sleep 0.1; done
STUB
    printf '#!/usr/bin/env bash\nsleep 1.2; echo "RESULT r9_lat stub"\n' >"$1/r9_lat"
    printf '#!/usr/bin/env bash\nsleep 0.5; echo "RESULT r9_bw stub"\n' >"$1/r9_bw"
    chmod +x "$1"/r9_bg "$1"/r9_lat "$1"/r9_bw
}
mk "$t/alive" alive
mk "$t/dead" dead
DRM_DEVICE=/nonexistent R9_BIN=$t/alive bash "$here/e_bg_inner.sh" "$t/out" a >"$t/a.txt" 2>&1
ra=$?
n=$(grep -c "^INFO bg alive after" "$t/a.txt")
((ra == 0 && n == 3)) && echo "PASS stub loop that keeps running: exit 0, $n alive lines" || { echo "FAIL running stub: exit $ra, $n alive lines"; rc=1; }
DRM_DEVICE=/nonexistent R9_BIN=$t/dead bash "$here/e_bg_inner.sh" "$t/out" d >"$t/d.txt" 2>&1
rd=$?
grep -q "^INFO bg DEAD" "$t/d.txt" && ((rd == 13)) && echo "PASS stub loop that exits at once: exit 13 and a DEAD line (the points are rejected)" || { echo "FAIL dead stub: exit $rd"; rc=1; }
exit $rc
