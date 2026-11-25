// CPU-only self-test of r9_bg_throttle.h: g++ -O2 -std=c++17 [-DBG_BREAK] r9_bg_selftest.cpp -o r9_bg_selftest
// -DBG_BREAK is the deliberate break: the throttle never pauses (the first version's behaviour); the test must then fail.
#include <cstdio>

#include "r9_bg_throttle.h"

#ifdef BG_BREAK
#define ACT(h, p, c) bg_act((h), (p), 1e9)
#else
#define ACT(h, p, c) bg_act((h), (p), (c))
#endif

static int fails = 0;
static void check(bool ok, const char * what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    fails += !ok;
}

int main() {
    const double cap = 76.0;
    check(ACT(60, false, cap) == BG_RUN, "60 C: runs");
    check(ACT(77, false, cap) == BG_PAUSE, "77 C (above the cap): pauses");
    check(ACT(74, true, cap) == BG_PAUSE, "74 C while paused: stays paused (resumes below 72)");
    check(ACT(71, true, cap) == BG_RUN, "71 C while paused: resumes");
    check(ACT(88, false, cap) == BG_TRIP, "88 C: tripwire");
    check(ACT(-1, false, cap) == BG_RUN, "no reading: runs");
    // a heating loop: 0.9 C per 20 ms step while running (the card went from 50 to 88 C in 2 s), 0.3 C cooling per step while
    // paused; 30 s of steps. The loop must never reach the tripwire and must still be alive (running or paused) at the end.
    double t = 50;
    bool paused = false, tripped = false;
    int run_steps = 0;
    double hottest = 0;
    for (int i = 0; i < 1500 && !tripped; ++i) {
        const BgAct a = ACT(t, paused, cap);
        if (a == BG_TRIP) {
            tripped = true;
        } else if (a == BG_PAUSE) {
            paused = true;
            t -= 0.3;
        } else {
            paused = false;
            t += 0.9;
            ++run_steps;
        }
        if (t > hottest) hottest = t;
    }
    std::printf("simulated 30 s: tripwire %d, running steps %d, hottest %.1f C\n", (int) tripped, run_steps, hottest);
    check(!tripped, "30 s of simulated heating never reaches the tripwire (the loop is still alive at the end)");
    check(run_steps > 300, "the loop keeps launching through the 30 s (not only at the start)");
    std::printf("%s\n", fails ? "SELFTEST FAILED" : "SELFTEST OK");
    return fails ? 1 : 0;
}
