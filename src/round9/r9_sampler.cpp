// Round 9 read-only sampler: every period, one CSV row of the memory clock, the core clock and the throttle words from
// gpu_metrics (format 1 content 3: 16 average_gfx_activity, 54 current_gfxclk, 58 current_uclk, 68 throttle_status,
// 112 indep_throttle_status), the "*" level of pp_dpm_sclk, the hotspot, with the CLOCK_MONOTONIC time. It only opens
// sysfs files; it never opens the GPU and never writes. Usage: r9_sampler <out dir> <period ms> (DRM_DEVICE and HWMON in
// the environment); TERM or INT ends it. Writes <out>/samples.csv and <out>/ready after the first row.
#include <fcntl.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }
static int64_t now_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t) t.tv_sec * 1000000000LL + t.tv_nsec;
}

// The MHz of the line of pp_dpm_sclk that carries '*' (-1 if none).
static int star_mhz(int fd) {
    char b[512];
    const ssize_t n = pread(fd, b, sizeof b - 1, 0);
    if (n <= 0) return -1;
    b[n] = 0;
    for (char * l = b; l && *l;) {
        char * e = std::strchr(l, '\n');
        if (e) *e = 0;
        if (std::strchr(l, '*')) {
            const char * c = std::strchr(l, ':');
            if (c) return std::atoi(c + 1);
        }
        l = e ? e + 1 : nullptr;
    }
    return -1;
}

int main(int argc, char ** argv) {
    if (argc < 3 || !std::getenv("DRM_DEVICE") || !std::getenv("HWMON")) {
        std::fprintf(stderr, "usage: r9_sampler <out dir> <period ms>, with DRM_DEVICE and HWMON set\n");
        return 2;
    }
    const std::string out = argv[1];
    const double period_ms = std::atof(argv[2]);
    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);
    const std::string dd = std::getenv("DRM_DEVICE");
    const int mfd = open((dd + "/gpu_metrics").c_str(), O_RDONLY);
    const int sfd = open((dd + "/pp_dpm_sclk").c_str(), O_RDONLY);
    const int hfd = open((std::string(std::getenv("HWMON")) + "/temp2_input").c_str(), O_RDONLY);
    if (mfd < 0 || hfd < 0) {
        std::fprintf(stderr, "r9_sampler: cannot open gpu_metrics or temp2_input\n");
        return 3;
    }
    FILE * f = std::fopen((out + "/samples.csv").c_str(), "w");
    if (!f) return 2;
    static char buf[1 << 20];
    setvbuf(f, buf, _IOFBF, sizeof buf);
    std::fprintf(f, "# r9_sampler period_ms=%g\nmono_ns,read_us,uclk_cur_MHz,gfxclk_cur_MHz,sclk_star_MHz,gfx_act_pct,throttle_status,"
                    "indep_throttle_status,hotspot_C,metrics_ok\n", period_ms);
    const int64_t per = (int64_t) (period_ms * 1e6);
    int64_t next = now_ns(), count = 0;
    while (!g_stop) {
        const int64_t a = now_ns();
        unsigned char b[128];
        const ssize_t n = pread(mfd, b, sizeof b, 0);
        const int star = sfd >= 0 ? star_mhz(sfd) : -1;
        char tb[32];
        const ssize_t hn = pread(hfd, tb, sizeof tb - 1, 0);
        const int64_t c = now_ns();
        const bool ok = n >= 120 && b[2] == 1 && b[3] == 3;
        int u = -1, g = -1, act = -1;
        uint32_t thr = 0;
        uint64_t ithr = 0;
        if (ok) {
            act = b[16] | (b[17] << 8);
            g = b[54] | (b[55] << 8);
            u = b[58] | (b[59] << 8);
            std::memcpy(&thr, b + 68, 4);
            std::memcpy(&ithr, b + 112, 8);
        }
        double hot = -1;
        if (hn > 0) {
            tb[hn] = 0;
            hot = std::atof(tb) / 1000.0;
        }
        std::fprintf(f, "%lld,%lld,%d,%d,%d,%d,0x%x,0x%llx,%.1f,%d\n", (long long) a, (long long) ((c - a) / 1000), u, g, star, act,
                     thr, (unsigned long long) ithr, hot, ok ? 1 : 0);
        if (++count == 1) {
            std::fflush(f);
            FILE * r = std::fopen((out + "/ready").c_str(), "w");
            if (r) std::fclose(r);
        } else if (count % 400 == 0) {
            std::fflush(f);
        }
        next += per;
        if (now_ns() >= next) {
            next = now_ns() + per;
            continue;
        }
        while (!g_stop) {
            const int64_t d = next - now_ns();
            if (d <= 0) break;
            timespec ts = {0, (long) std::min<int64_t>(d, 1000000)};
            nanosleep(&ts, nullptr);
        }
    }
    std::fclose(f);
    return 0;
}
