// Round 11 read-only sampler: r9_sampler extended with the power, temperature, fan and video-block fields round 11 needs.
// Every period it writes one CSV row from the kernel's gpu_metrics table (format 1, content 3, struct gpu_metrics_v1_3 in the
// kernel's kgd_pp_interface.h; the RX 9070 XT exports this one, 120 bytes, as r9_sampler already checked), the "*" level of
// pp_dpm_sclk, and three hwmon files: hotspot (temp2), edge (temp1), memory (temp3), power1_average (microwatts) and fan1_input
// (RPM). It opens sysfs files only, never the GPU, and never writes to sysfs.
//
// Usage: r11_sampler <out dir> <period ms>   (DRM_DEVICE and HWMON in the environment); TERM or INT ends it.
//        r11_sampler --selftest [--break]    no sysfs: decodes a synthetic table and checks every field offset; --break shifts
//                                            the decoder by one field and must exit 1.
// Writes <out>/samples.csv and <out>/ready after the first row. The first nine columns are r9_sampler's, in the same order.
//
// A video clock or activity field that reads 0 or 0xffff is reported as read: the table does not say whether the block is power
// gated or the firmware does not fill the field. The analysis reads "vclk0_MHz > 0" as video activity only after the
// idle baseline has been read the same way.
#include <fcntl.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Copy of the kernel's struct gpu_metrics_v1_3 (NUM_HBM_INSTANCES is 4). The static_asserts pin every offset the sampler
// uses to the value the kernel header gives, so a different compiler layout is a build error, not a wrong column.
struct metrics_header {
    uint16_t structure_size;
    uint8_t format_revision;
    uint8_t content_revision;
};
struct gpu_metrics_v1_3 {
    metrics_header common_header;
    uint16_t temperature_edge, temperature_hotspot, temperature_mem, temperature_vrgfx, temperature_vrsoc, temperature_vrmem;
    uint16_t average_gfx_activity, average_umc_activity, average_mm_activity;
    uint16_t average_socket_power;
    uint64_t energy_accumulator;
    uint64_t system_clock_counter;
    uint16_t average_gfxclk_frequency, average_socclk_frequency, average_uclk_frequency, average_vclk0_frequency,
        average_dclk0_frequency, average_vclk1_frequency, average_dclk1_frequency;
    uint16_t current_gfxclk, current_socclk, current_uclk, current_vclk0, current_dclk0, current_vclk1, current_dclk1;
    uint32_t throttle_status;
    uint16_t current_fan_speed, pcie_link_width, pcie_link_speed, padding;
    uint32_t gfx_activity_acc, mem_activity_acc;
    uint16_t temperature_hbm[4];
    uint64_t firmware_timestamp;
    uint16_t voltage_soc, voltage_gfx, voltage_mem, padding1;
    uint64_t indep_throttle_status;
};
static_assert(sizeof(gpu_metrics_v1_3) == 120, "gpu_metrics_v1_3 is 120 bytes");
static_assert(offsetof(gpu_metrics_v1_3, temperature_edge) == 4, "edge");
static_assert(offsetof(gpu_metrics_v1_3, temperature_mem) == 8, "mem");
static_assert(offsetof(gpu_metrics_v1_3, average_gfx_activity) == 16, "gfx activity");
static_assert(offsetof(gpu_metrics_v1_3, average_mm_activity) == 20, "mm activity");
static_assert(offsetof(gpu_metrics_v1_3, average_socket_power) == 22, "socket power");
static_assert(offsetof(gpu_metrics_v1_3, average_vclk0_frequency) == 46, "avg vclk0");
static_assert(offsetof(gpu_metrics_v1_3, current_gfxclk) == 54, "gfxclk");
static_assert(offsetof(gpu_metrics_v1_3, current_uclk) == 58, "uclk");
static_assert(offsetof(gpu_metrics_v1_3, current_vclk0) == 60, "vclk0");
static_assert(offsetof(gpu_metrics_v1_3, current_dclk0) == 62, "dclk0");
static_assert(offsetof(gpu_metrics_v1_3, current_vclk1) == 64, "vclk1");
static_assert(offsetof(gpu_metrics_v1_3, current_dclk1) == 66, "dclk1");
static_assert(offsetof(gpu_metrics_v1_3, throttle_status) == 68, "throttle");
static_assert(offsetof(gpu_metrics_v1_3, current_fan_speed) == 72, "fan");
static_assert(offsetof(gpu_metrics_v1_3, pcie_link_speed) == 76, "link speed");
static_assert(offsetof(gpu_metrics_v1_3, indep_throttle_status) == 112, "independent throttle");

struct Row {
    int uclk, gfxclk, act, mm_act, vclk0, dclk0, vclk1, dclk1, edge, memt, sock_w, fan_pct, link_w, link_s;
    uint32_t thr;
    uint64_t ithr;
};

// shift = 0 is the real decoder; the selftest's deliberate break reads every field one 16-bit slot too early.
static Row decode(const unsigned char * b, int shift) {
    gpu_metrics_v1_3 m;
    std::memcpy(&m, b + shift, sizeof m - (size_t) shift > 0 ? sizeof m - (size_t) shift : 0);
    Row r;
    r.uclk = m.current_uclk;
    r.gfxclk = m.current_gfxclk;
    r.act = m.average_gfx_activity;
    r.mm_act = m.average_mm_activity;
    r.vclk0 = m.current_vclk0;
    r.dclk0 = m.current_dclk0;
    r.vclk1 = m.current_vclk1;
    r.dclk1 = m.current_dclk1;
    r.edge = m.temperature_edge;
    r.memt = m.temperature_mem;
    r.sock_w = m.average_socket_power;
    r.fan_pct = m.current_fan_speed;
    r.link_w = m.pcie_link_width;
    r.link_s = m.pcie_link_speed;
    r.thr = m.throttle_status;
    r.ithr = m.indep_throttle_status;
    return r;
}

static int selftest(bool brk) {
    gpu_metrics_v1_3 m;
    std::memset(&m, 0, sizeof m);
    m.common_header = {120, 1, 3};
    m.temperature_edge = 41;
    m.temperature_mem = 52;
    m.average_gfx_activity = 77;
    m.average_mm_activity = 63;
    m.average_socket_power = 211;
    m.current_gfxclk = 2987;
    m.current_uclk = 1124;
    m.current_vclk0 = 1111;
    m.current_dclk0 = 1222;
    m.current_vclk1 = 1333;
    m.current_dclk1 = 1444;
    m.throttle_status = 0xa5a5;
    m.current_fan_speed = 35;
    m.pcie_link_width = 16;
    m.pcie_link_speed = 320;
    m.indep_throttle_status = 0x123456789abcdefull;
    unsigned char b[sizeof m];
    std::memcpy(b, &m, sizeof m);
    const Row r = decode(b, brk ? 2 : 0);
    int bad = 0;
    auto chk = [&](const char * n, long long got, long long want) {
        if (got != want) {
            std::printf("selftest FAIL %s: got %lld want %lld\n", n, got, want);
            bad++;
        }
    };
    chk("edge", r.edge, 41);
    chk("memt", r.memt, 52);
    chk("act", r.act, 77);
    chk("mm_act", r.mm_act, 63);
    chk("sock_w", r.sock_w, 211);
    chk("gfxclk", r.gfxclk, 2987);
    chk("uclk", r.uclk, 1124);
    chk("vclk0", r.vclk0, 1111);
    chk("dclk0", r.dclk0, 1222);
    chk("vclk1", r.vclk1, 1333);
    chk("dclk1", r.dclk1, 1444);
    chk("thr", r.thr, 0xa5a5);
    chk("fan", r.fan_pct, 35);
    chk("link_w", r.link_w, 16);
    chk("link_s", r.link_s, 320);
    chk("ithr", (long long) r.ithr, 0x123456789abcdefll);
    std::printf("selftest%s: %d field(s) wrong\n", brk ? " (deliberate break)" : "", bad);
    return bad ? 1 : 0;
}

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

// A hwmon number read from an open file, divided by `div` (-1 if unreadable).
static double hw_num(int fd, double div) {
    if (fd < 0) return -1;
    char tb[32];
    const ssize_t n = pread(fd, tb, sizeof tb - 1, 0);
    if (n <= 0) return -1;
    tb[n] = 0;
    return std::atof(tb) / div;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && !std::strcmp(argv[1], "--selftest")) return selftest(argc >= 3 && !std::strcmp(argv[2], "--break"));
    if (argc < 3 || !std::getenv("DRM_DEVICE") || !std::getenv("HWMON")) {
        std::fprintf(stderr, "usage: r11_sampler <out dir> <period ms>, with DRM_DEVICE and HWMON set (or --selftest [--break])\n");
        return 2;
    }
    const std::string out = argv[1];
    const double period_ms = std::atof(argv[2]);
    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);
    const std::string dd = std::getenv("DRM_DEVICE"), hw = std::getenv("HWMON");
    const int mfd = open((dd + "/gpu_metrics").c_str(), O_RDONLY);
    const int sfd = open((dd + "/pp_dpm_sclk").c_str(), O_RDONLY);
    const int hfd = open((hw + "/temp2_input").c_str(), O_RDONLY);
    const int efd = open((hw + "/temp1_input").c_str(), O_RDONLY);
    const int tfd = open((hw + "/temp3_input").c_str(), O_RDONLY);
    const int pfd = open((hw + "/power1_average").c_str(), O_RDONLY);
    const int ffd = open((hw + "/fan1_input").c_str(), O_RDONLY);
    if (mfd < 0 || hfd < 0) {
        std::fprintf(stderr, "r11_sampler: cannot open gpu_metrics or temp2_input\n");
        return 3;
    }
    FILE * f = std::fopen((out + "/samples.csv").c_str(), "w");
    if (!f) return 2;
    static char buf[1 << 20];
    setvbuf(f, buf, _IOFBF, sizeof buf);
    std::fprintf(f, "# r11_sampler period_ms=%g gpu_metrics v1_3 (power_W is hwmon power1_average; sock_power_W is the table's)\n"
                    "mono_ns,read_us,uclk_cur_MHz,gfxclk_cur_MHz,sclk_star_MHz,gfx_act_pct,throttle_status,indep_throttle_status,"
                    "hotspot_C,edge_C,mem_C,power_W,sock_power_W,fan_rpm,fan_pct,vclk0_MHz,dclk0_MHz,vclk1_MHz,dclk1_MHz,"
                    "mm_act_pct,pcie_w,pcie_gts_x10,metrics_ok\n", period_ms);
    const int64_t per = (int64_t) (period_ms * 1e6);
    int64_t next = now_ns(), count = 0;
    while (!g_stop) {
        const int64_t a = now_ns();
        unsigned char b[sizeof(gpu_metrics_v1_3)];
        const ssize_t n = pread(mfd, b, sizeof b, 0);
        const int star = sfd >= 0 ? star_mhz(sfd) : -1;
        const double hot = hw_num(hfd, 1000.0), edge_hw = hw_num(efd, 1000.0), mem_hw = hw_num(tfd, 1000.0),
                     pw = hw_num(pfd, 1e6), fan = hw_num(ffd, 1.0);
        const int64_t c = now_ns();
        const bool ok = n >= (ssize_t) sizeof b && b[2] == 1 && b[3] == 3;
        Row r = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0};
        if (ok) r = decode(b, 0);
        std::fprintf(f, "%lld,%lld,%d,%d,%d,%d,0x%x,0x%llx,%.1f,%.1f,%.1f,%.1f,%d,%.0f,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", (long long) a,
                     (long long) ((c - a) / 1000), r.uclk, r.gfxclk, star, r.act, r.thr, (unsigned long long) r.ithr, hot, edge_hw,
                     mem_hw, pw, r.sock_w, fan, r.fan_pct, r.vclk0, r.dclk0, r.vclk1, r.dclk1, r.mm_act, r.link_w, r.link_s,
                     ok ? 1 : 0);
        if (++count == 1) {
            std::fflush(f);
            FILE * rf = std::fopen((out + "/ready").c_str(), "w");
            if (rf) std::fclose(rf);
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
