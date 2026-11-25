// The throttle of r9_bg (round 9, item E), kept free of the GPU runtime so that r9_bg_selftest.cpp can test it on the CPU.
// The loop must stay alive for the whole time the latency and bandwidth probes run beside it. Round 9's first version ran
// flat out, reached its 88 C tripwire after 2 s and was gone before the chases started. Now it pauses at `cap` C and
// resumes 4 C lower; the tripwire only ends the loop if the card gets hotter anyway (and then the exit status is 5).
#pragma once

enum BgAct { BG_RUN, BG_PAUSE, BG_TRIP };

constexpr double BG_TRIPWIRE_C = 88.0;
constexpr double BG_RESUME_BELOW_CAP_C = 4.0;

inline BgAct bg_act(double hot, bool paused, double cap) {
    if (hot >= BG_TRIPWIRE_C) {
        return BG_TRIP;
    }
    if (hot < 0) {
        return BG_RUN;  // no reading (HWMON unset): the gate guards the temperature
    }
    if (paused) {
        return hot < cap - BG_RESUME_BELOW_CAP_C ? BG_RUN : BG_PAUSE;
    }
    return hot >= cap ? BG_PAUSE : BG_RUN;
}
