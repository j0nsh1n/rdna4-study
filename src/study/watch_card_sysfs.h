// Host-only, read-only access to the card's sysfs files for watch_card. Copied in part from sched_common.h (star_mhz,
// the gpu_metrics check and offsets, the throttle bit names) so that this program needs no HIP header: it must not
// open the GPU, because opening the device changes the clock behaviour it is meant to observe.
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

namespace watch {

// Reads a whole small file with one open, read and close (so a file replaced on disk is seen at once). Returns the
// byte count, or -1.
inline ssize_t read_file(const std::string & path, char * buf, size_t cap) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return -1;
    }
    size_t n = 0;
    while (n < cap) {
        const ssize_t r = ::read(fd, buf + n, cap - n);
        if (r < 0) {
            ::close(fd);
            return -1;
        }
        if (r == 0) {
            break;
        }
        n += (size_t) r;
    }
    ::close(fd);
    return (ssize_t) n;
}

inline std::string slurp(const std::string & path) {
    char b[8192];
    const ssize_t n = read_file(path, b, sizeof b);
    return n < 0 ? std::string() : std::string(b, (size_t) n);
}

// A whole number scaled; -1 when unreadable.
inline double read_num(const std::string & path, double scale) {
    char b[64];
    const ssize_t n = read_file(path, b, sizeof b - 1);
    if (n <= 0) {
        return -1;
    }
    b[n] = 0;
    char * end = nullptr;
    const double v = std::strtod(b, &end);
    if (end == b) {
        return -1;
    }
    return v * scale;
}

// A temperature file for the guard, scaled: digits only, then optional white space (sysfs gives integer millidegrees).
// "nan", "-nan", "inf", "96abc", a sign, an empty file and a file that cannot be read all give -1: strtod would turn
// the first two into NaN (which compares false against every limit) and "96abc" into 96.
inline double read_temp_strict(const std::string & path, double scale) {
    char b[64];
    const ssize_t n = read_file(path, b, sizeof b - 1);
    if (n <= 0) {
        return -1;
    }
    size_t i = 0;
    double v = 0;
    while (i < (size_t) n && b[i] >= '0' && b[i] <= '9' && i < 15) {
        v = v * 10 + (b[i] - '0');
        ++i;
    }
    if (i == 0) {
        return -1;
    }
    while (i < (size_t) n && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r')) {
        ++i;
    }
    return i == (size_t) n ? v * scale : -1;
}

// The '*' line of pp_dpm_sclk / pp_dpm_mclk: its label ("S", "0", "1", ...) and its MHz. False when there is none or
// when the line does not carry a positive number after the colon: a garbled line must read as missing, never as
// 0 MHz, which the summary would count as asleep.
inline bool star_line(const std::string & path, std::string & label, int & mhz) {
    const std::string s = slurp(path);
    size_t a = 0;
    while (a < s.size()) {
        size_t e = s.find('\n', a);
        if (e == std::string::npos) {
            e = s.size();
        }
        const std::string line = s.substr(a, e - a);
        if (line.find('*') != std::string::npos) {
            const size_t c = line.find(':');
            if (c == std::string::npos) {
                return false;
            }
            label = line.substr(0, c);
            while (!label.empty() && label.front() == ' ') {
                label.erase(0, 1);
            }
            const char * p = line.c_str() + c + 1;
            while (*p == ' ' || *p == '\t') {
                ++p;
            }
            char * end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (label.empty() || end == p || v <= 0 || v > 100000) {
                return false;
            }
            mhz = (int) v;
            return true;
        }
        a = e + 1;
    }
    return false;
}

// gpu_metrics format 1, content 3 (struct gpu_metrics_v1_3 of the kernel, 120 bytes). The study checked bytes 68,
// 106 and 112 on this card (sched_common.h); the gaming round's sampler uses 16, 22, 40, 54 and 96 as well
// (gaming-route.md: its clock equals pp_dpm_sclk whenever the file changes). Offsets: 16 average_gfx_activity (%),
// 22 average_socket_power (W), 40 average_gfxclk_frequency (MHz), 54 current_gfxclk (MHz), 58 current_uclk (MHz:
// the memory clock that really runs; pp_dpm_mclk marks only its listed levels and shows none at an overclocked
// 1408 MHz, seen 2026-11-18), 68 throttle_status,
// 106 voltage_gfx (mV), 112 indep_throttle_status. firmware_timestamp (96) reads all ones on this card, so a fresh
// table is recognised by its content instead: `id` is a hash of bytes 4 to 119 without system_clock_counter (32 to
// 39, which the driver stamps at every read).
struct Metrics {
    bool ok = false;
    int gfx_act = -1, socket_w = -1, avg_gfxclk = -1, cur_gfxclk = -1, cur_uclk = -1, vgfx_mv = -1;
    uint32_t thr = 0;
    uint64_t ithr = 0, id = 0;
};

inline Metrics decode_metrics(const unsigned char * b, size_t n) {
    Metrics m;
    if (n < 120 || b[0] != 120 || b[1] != 0 || b[2] != 1 || b[3] != 3) {
        return m;
    }
    uint16_t u;
    std::memcpy(&u, b + 16, 2);
    m.gfx_act = u;
    std::memcpy(&u, b + 22, 2);
    m.socket_w = u;
    std::memcpy(&u, b + 40, 2);
    m.avg_gfxclk = u;
    std::memcpy(&u, b + 54, 2);
    m.cur_gfxclk = u;
    std::memcpy(&u, b + 58, 2);
    m.cur_uclk = u;
    std::memcpy(&m.thr, b + 68, 4);
    m.id = 1469598103934665603ULL;
    for (size_t i = 4; i < 120; ++i) {
        if (i < 32 || i >= 40) {
            m.id = (m.id ^ b[i]) * 1099511628211ULL;
        }
    }
    std::memcpy(&u, b + 106, 2);
    m.vgfx_mv = u;
    std::memcpy(&m.ithr, b + 112, 8);
    m.ok = true;
    return m;
}

inline Metrics read_metrics(const std::string & path) {
    unsigned char b[256];
    const ssize_t n = read_file(path, reinterpret_cast<char *>(b), sizeof b);
    return n < 0 ? Metrics{} : decode_metrics(b, (size_t) n);
}

// Names of the driver's ASIC-independent throttler bits (SMU_THROTTLER_*_BIT in amdgpu_smu.h; the study's table in
// sched_common.h, checked against Linux master on 2026-11-17). nullptr for a bit without a name.
inline const char * throttle_name(int bit) {
    static const std::pair<int, const char *> k[] = {
        {0, "PPT0"},          {1, "PPT1"},          {2, "PPT2"},          {3, "PPT3"},          {4, "SPL"},
        {5, "FPPT"},          {6, "SPPT"},          {7, "SPPT_APU"},      {16, "TDC_GFX"},      {17, "TDC_SOC"},
        {18, "TDC_MEM"},      {19, "TDC_VDD"},      {20, "TDC_CVIP"},     {21, "EDC_CPU"},      {22, "EDC_GFX"},
        {23, "APCC"},         {32, "TEMP_GPU"},     {33, "TEMP_CORE"},    {34, "TEMP_MEM"},     {35, "TEMP_EDGE"},
        {36, "TEMP_HOTSPOT"}, {37, "TEMP_SOC"},     {38, "TEMP_VR_GFX"},  {39, "TEMP_VR_SOC"},  {40, "TEMP_VR_MEM0"},
        {41, "TEMP_VR_MEM1"}, {42, "TEMP_LIQUID0"}, {43, "TEMP_LIQUID1"}, {44, "VRHOT0"},       {45, "VRHOT1"},
        {46, "PROCHOT_CPU"},  {47, "PROCHOT_GFX"},  {56, "PPM"},          {57, "FIT"}};
    for (auto & e : k) {
        if (e.first == bit) {
            return e.second;
        }
    }
    return nullptr;
}

}  // namespace watch
