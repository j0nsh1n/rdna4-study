// watch_card: records what the card does over time, read-only, from sysfs and /proc only. It never opens the GPU
// (no HIP, Vulkan, OpenCL or DRM device node), because opening the device changes the clock behaviour to be observed.
//
// Usage:
//   watch_card record --out <dir> [--label <text>] [--period-ms 2] [--seconds 0] [--die-with-parent]
//                     [--guard-pidfile <file> --stop-hot <C> [--stop-mem <C>] [--guard-log <file>]]
//       Samples every period until the time ends (0: until INT, TERM or HUP), then finishes its files: <dir>/header.txt
//       (conditions at the start and at the end), samples.csv (one row per sample), summary.txt (as `summarize`),
//       clients-start.txt and clients-end.txt (names of the processes that hold the card). <dir>/ready appears once
//       the first sample is on disk. --die-with-parent: a TERM when the parent process ends (files still finished).
//       Guard: once <file> holds a process id, every sample checks the hotspot and the memory temperature; at the
//       --stop-hot or --stop-mem limit (or a temperature that cannot be read) it sends TERM to that process, and KILL
//       to its process group 1 s later if it is still running; each step is one line "THERMAL STOP: ..." appended to
//       the guard log (and stderr). The reaction time is one sampling period plus the time to read the files.
//   watch_card guard --pid <pid> --hot-file <file> --stop-hot <C> [--period-ms 20] [--log <file>] [--csv <file>]
//                    [--kill-after-ms 500] [--mem-file <file> --stop-mem <C>] [--scope <name.scope>]
//       The thermal guard of one GPU job, on its own (no recording, no LACT call, no /proc scan). It reads the
//       hotspot file (millidegrees) and, with --mem-file, the memory temperature file every period and writes one row per
//       sample to --csv (t_s,hotspot_C,mem_C,read_us). A reading counts only when the whole text is digits (nan, inf, "96abc",
//       an empty or missing file, and 0 are unreadable). At the --stop-hot limit, at the --stop-mem limit, or when the
//       hotspot file cannot be read (a sensor that does not answer is a stop too), it sends TERM to
//       the job's process group (the group whose leader is --pid; when --pid does not lead a group, to the process
//       only), and KILL to the group --kill-after-ms later if anything is left. A memory file that cannot be read while
//       the hotspot is fine is one "WARNING" line, not a stop. With --scope (the systemd scope the job runs in), the same
//       TERM and KILL also go to every process of that scope (`systemctl --user kill`), so a child that started its own
//       session or process group is stopped too, and the guard ends only when the group and the scope are empty.
//       Each step is one line "THERMAL STOP: ..."
//       with the reading, the time since the guard started and the UTC time, appended to --log (and stderr); the end of
//       the run adds one "guard: ..." line (samples, sample period, hotspot maximum). It ends when the job has ended.
//       Exit: 0 the job ended without a stop, 8 a thermal stop, 2 usage. The reaction time is one period plus the read.
//   watch_card summarize <samples.csv>   prints the summary of a recorded file again
//   watch_card clients [--pids]          prints the processes that hold the card, once (--pids: "pid name" per
//                                        process, for the kit's check that no game runs)
// Environment: DRM_DEVICE and HWMON (the card's sysfs folders; scripts/study-home.sh sets them from the card record),
//   WATCH_PDEV (PCI address in fdinfo; default: the last part of DRM_DEVICE), WATCH_PROC (default /proc),
//   WATCH_LACT (command that prints the LACT profile; default "timeout 20 lact cli profile get").
// Besides the sysfs reads it runs, at the start and the end only, `rpm -qa` (Mesa version), `date` and the WATCH_LACT
// command (`lact cli profile get`, read-only).
// gfxclk_avg_MHz (gpu_metrics offset 40) equalled gfxclk_cur_MHz (offset 54) in every one of 6,330 real samples on
// this card (three recordings, 2026-11-18): it is not an independent average here; the summary counts the matches.
// Shares and percentiles of the summary are weighted by the time each sample stands for (its interval).
// Exit: 0 done, 2 usage (a missing or non-numeric option value included) or a file that cannot be written or read,
// 3 the card's clock file cannot be read at the start.
#include "watch_card_sysfs.h"

#include <dirent.h>
#include <spawn.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace watch;

namespace {

volatile sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

int64_t now_ns(clockid_t c) {
    timespec t;
    clock_gettime(c, &t);
    return (int64_t) t.tv_sec * 1000000000LL + t.tv_nsec;
}

std::string env_or(const char * k, const std::string & def) {
    const char * v = std::getenv(k);
    return v && *v ? std::string(v) : def;
}

struct Paths {
    std::string drm, hw, pdev, proc, lact;
};

Paths get_paths() {
    Paths p;
    p.drm = env_or("DRM_DEVICE", "");
    p.hw = env_or("HWMON", "");
    std::string base = p.drm;
    while (!base.empty() && base.back() == '/') {
        base.pop_back();
    }
    const size_t s = base.rfind('/');
    p.pdev = env_or("WATCH_PDEV", s == std::string::npos ? base : base.substr(s + 1));
    p.proc = env_or("WATCH_PROC", "/proc");
    p.lact = env_or("WATCH_LACT", "timeout 20 lact cli profile get");
    return p;
}

std::string trim(std::string s) {
    while (!s.empty() && std::isspace((unsigned char) s.back())) {
        s.pop_back();
    }
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char) s[a])) {
        ++a;
    }
    return s.substr(a);
}

std::string run_cmd(const std::string & cmd) {
    std::string out;
    FILE * f = popen((cmd + " 2>/dev/null").c_str(), "r");
    if (!f) {
        return "";
    }
    char b[512];
    while (std::fgets(b, sizeof b, f)) {
        out += b;
    }
    pclose(f);
    return trim(out);
}

// Joins the lines of a small file into one, with " | " between them, for the header.
std::string flat(const std::string & s) {
    std::string r, line;
    std::istringstream in(s);
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        r += (r.empty() ? "" : " | ") + line;
    }
    return r.empty() ? "unreadable" : r;
}

// pp_power_profile_mode: the active mode (index:name of the line whose name carries '*') and every mode's name.
void power_modes(const std::string & file, std::string & active, std::string & names) {
    std::istringstream in(slurp(file));
    std::string line;
    active = "unreadable";
    names.clear();
    while (std::getline(in, line)) {
        const size_t c = line.find(':');
        if (c == std::string::npos) {
            continue;
        }
        std::istringstream h(line.substr(0, c));
        std::string idx, name, more;
        if (!(h >> idx) || !std::isdigit((unsigned char) idx[0])) {
            continue;
        }
        std::getline(h, name);
        name = trim(name);
        const bool star = name.find('*') != std::string::npos;
        name.erase(std::remove(name.begin(), name.end(), '*'), name.end());
        name = trim(name);
        if (name.empty() || name.find('(') != std::string::npos) {
            continue;  // a clock-type row of a mode, not a mode
        }
        names += (names.empty() ? "" : " ") + idx + ":" + name;
        if (star) {
            active = idx + ":" + name;
        }
    }
    if (names.empty()) {
        names = "unreadable";
    }
}

// Processes with an fdinfo entry for this card (drm-pdev), as "name: N clients" (distinct drm-client-id per name).
// Reads only; another user's process that cannot be read is not listed. With pids, also each process once as
// (pid, name).
std::vector<std::string> card_clients(const Paths & p, std::set<std::pair<long, std::string>> * pids = nullptr) {
    std::map<std::string, std::set<std::string>> by_name;
    DIR * d = opendir(p.proc.c_str());
    if (!d) {
        return {"(cannot read " + p.proc + ")"};
    }
    while (dirent * e = readdir(d)) {
        if (!std::isdigit((unsigned char) e->d_name[0])) {
            continue;
        }
        const std::string pid = e->d_name, fdi = p.proc + "/" + pid + "/fdinfo";
        DIR * f = opendir(fdi.c_str());
        if (!f) {
            continue;
        }
        while (dirent * g = readdir(f)) {
            if (!std::isdigit((unsigned char) g->d_name[0])) {
                continue;
            }
            const std::string s = slurp(fdi + "/" + g->d_name);
            const size_t a = s.find("drm-pdev:");
            if (a == std::string::npos) {
                continue;
            }
            const size_t eol = s.find('\n', a);
            if (trim(s.substr(a + 9, eol == std::string::npos ? std::string::npos : eol - a - 9)) != p.pdev) {
                continue;
            }
            std::string id = "fd" + std::string(g->d_name);
            const size_t c = s.find("drm-client-id:");
            if (c != std::string::npos) {
                const size_t ce = s.find('\n', c);
                id = trim(s.substr(c + 14, ce == std::string::npos ? std::string::npos : ce - c - 14));
            }
            const std::string name = trim(slurp(p.proc + "/" + pid + "/comm"));
            by_name[name.empty() ? "pid-" + pid : name].insert(pid + "/" + id);
            if (pids) {
                pids->insert({std::atol(pid.c_str()), name.empty() ? "pid-" + pid : name});
            }
        }
        closedir(f);
    }
    closedir(d);
    std::vector<std::string> r;
    for (auto & kv : by_name) {
        r.push_back(kv.first + ": " + std::to_string(kv.second.size()) + (kv.second.size() == 1 ? " client" : " clients"));
    }
    if (r.empty()) {
        r.push_back("(none seen)");
    }
    return r;
}

bool write_lines(const std::string & path, const std::vector<std::string> & lines) {
    FILE * f = std::fopen(path.c_str(), "w");
    if (!f) {
        return false;
    }
    for (auto & l : lines) {
        std::fprintf(f, "%s\n", l.c_str());
    }
    return std::fclose(f) == 0;
}

// Conditions that matter for reading the clock: written at the start and again at the end of a recording.
std::string conditions(const Paths & p, const char * when) {
    std::string active, names;
    power_modes(p.drm + "/pp_power_profile_mode", active, names);
    std::ostringstream o;
    o << when << " lact_profile: " << run_cmd(p.lact) << "\n"
      << when << " power1_cap_W: " << read_num(p.hw + "/power1_cap", 1e-6) << " (default "
      << read_num(p.hw + "/power1_cap_default", 1e-6) << ")\n"
      << when << " performance_level: " << trim(slurp(p.drm + "/power_dpm_force_performance_level")) << "\n"
      << when << " power_profile_mode_active: " << active << "\n"
      << when << " power_profile_modes: " << names << "\n"
      << when << " overdrive: " << flat(slurp(p.drm + "/pp_od_clk_voltage")) << "\n"
      << when << " pp_dpm_sclk: " << flat(slurp(p.drm + "/pp_dpm_sclk")) << "\n"
      << when << " pp_dpm_mclk: " << flat(slurp(p.drm + "/pp_dpm_mclk")) << "\n"
      << when << " fan_curve: " << flat(slurp(p.drm + "/gpu_od/fan_ctrl/fan_curve")) << "\n"
      << when << " fan_zero_rpm_enable: " << flat(slurp(p.drm + "/gpu_od/fan_ctrl/fan_zero_rpm_enable")) << "\n"
      << when << " hotspot_C: " << read_num(p.hw + "/temp2_input", 1e-3) << "\n";
    return o.str();
}

// ---------------------------------------------------------------------------------------------------------------
// Summary of a samples.csv (the record mode calls this on its own file, so both paths give the same text)

struct Table {
    std::map<std::string, std::string> meta;  // from "# key=value" lines
    std::vector<std::string> cols;
    std::vector<std::vector<double>> rows;    // -1: empty field; 0 for a hexadecimal field (its value is in words)
    std::vector<std::vector<uint64_t>> words;  // hexadecimal fields as 64-bit integers (a double keeps only 53 bits)
    int col(const std::string & name) const {
        for (size_t i = 0; i < cols.size(); ++i) {
            if (cols[i] == name) {
                return (int) i;
            }
        }
        return -1;
    }
};

std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> r;
    std::string cur;
    for (char c : s) {
        if (c == sep) {
            r.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    r.push_back(cur);
    return r;
}

bool load_csv(const std::string & path, Table & t) {
    FILE * f = std::fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }
    char b[4096];
    while (std::fgets(b, sizeof b, f)) {
        std::string line = trim(b);
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            for (auto & kv : split(line.substr(1), ' ')) {
                const size_t e = kv.find('=');
                if (e != std::string::npos) {
                    t.meta[kv.substr(0, e)] = kv.substr(e + 1);
                }
            }
            continue;
        }
        if (t.cols.empty()) {
            t.cols = split(line, ',');
            continue;
        }
        const auto f2 = split(line, ',');
        std::vector<double> r(t.cols.size(), -1);
        std::vector<uint64_t> w(t.cols.size(), 0);
        for (size_t i = 0; i < f2.size() && i < r.size(); ++i) {
            const std::string & v = f2[i];
            if (v.empty()) {
                continue;
            }
            // Hexadecimal throttle words and the metrics hash stay integers; level labels are kept as -2 (asleep
            // "S") or the level number.
            if (v.size() > 2 && v[0] == '0' && v[1] == 'x') {
                w[i] = std::strtoull(v.c_str() + 2, nullptr, 16);
                r[i] = 0;
            } else if (v == "S") {
                r[i] = -2;
            } else {
                r[i] = std::strtod(v.c_str(), nullptr);
            }
        }
        t.rows.push_back(r);
        t.words.push_back(w);
    }
    std::fclose(f);
    return !t.cols.empty();
}

// Nearest-rank percentile (p in 0..1) as in sched_common.h.
double pct(std::vector<double> v, double p) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    const size_t i = (size_t) std::llround(p * (double) (v.size() - 1));
    return v[std::min(i, v.size() - 1)];
}

std::string fmt(const char * f, double v) {
    if (std::isnan(v)) {
        return "n/a";
    }
    char b[64];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

// Time-weighted percentile of (value, duration) pairs: the smallest value whose share of the time, counted from the
// lowest value up, reaches p (p 0: the lowest value, p 1: the highest), so a sample counts as long as it stands for.
double wpct(std::vector<std::pair<double, double>> v, double p) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    double total = 0;
    for (auto & x : v) {
        total += x.second;
    }
    double cum = 0;
    for (auto & x : v) {
        cum += x.second;
        if (cum >= p * total - 1e-9 * total) {
            return x.first;
        }
    }
    return v.back().first;
}

// p10 / p50 / p90 / p99 / max (time-weighted by the sample intervals iv) of the valid (>= 0) values of one column,
// optionally only where keep_row is true; n is the number of samples.
template <class Keep>
std::string dist_if(const Table & t, int c, const std::vector<double> & iv, const char * f, Keep keep_row) {
    std::vector<std::pair<double, double>> v;
    if (c >= 0) {
        for (size_t i = 0; i < t.rows.size(); ++i) {
            if (t.rows[i][c] >= 0 && keep_row(t.rows[i])) {
                v.push_back({t.rows[i][c], iv[i]});
            }
        }
    }
    if (v.empty()) {
        return "no readable samples";
    }
    return "p10 " + fmt(f, wpct(v, 0.1)) + " p50 " + fmt(f, wpct(v, 0.5)) + " p90 " + fmt(f, wpct(v, 0.9)) + " p99 " +
           fmt(f, wpct(v, 0.99)) + " max " + fmt(f, wpct(v, 1.0)) + " (n " + std::to_string(v.size()) + ")";
}
std::string dist(const Table & t, const std::string & name, const std::vector<double> & iv, const char * f) {
    return dist_if(t, t.col(name), iv, f, [](const std::vector<double> &) { return true; });
}

std::string summarize(const Table & t) {
    std::ostringstream o;
    const size_t n = t.rows.size();
    const int ct = t.col("t_s"), cs = t.col("sclk_MHz"), cg = t.col("gfxclk_cur_MHz"), ch = t.col("hotspot_C"),
              ci = t.col("indep_throttle_status"), cl = t.col("throttle_status"), cf = t.col("metrics_id"),
              cr = t.col("read_us");
    o << "LABEL " << (t.meta.count("label") ? t.meta.at("label") : "none") << "\n";
    if (n < 2 || ct < 0 || cs < 0) {
        o << "SAMPLES " << n << " (too few to summarize)\n";
        return o.str();
    }
    // Sample intervals: interval i = t[i+1] - t[i]; the last sample gets the median interval. Every duration below
    // is a sum of these, so a late sample stretches the time it stands for instead of being lost.
    std::vector<double> gaps;
    for (size_t i = 0; i + 1 < n; ++i) {
        gaps.push_back((t.rows[i + 1][ct] - t.rows[i][ct]) * 1e3);
    }
    const double med_gap = pct(gaps, 0.5);
    std::vector<double> iv(gaps);
    iv.push_back(med_gap);
    const double span = t.rows[n - 1][ct] - t.rows[0][ct] + med_gap / 1e3;
    o << "SAMPLES " << n << " over " << fmt("%.3f", span) << " s; requested period "
      << (t.meta.count("period_ms") ? t.meta.at("period_ms") : "?") << " ms\n";
    o << "TIMING period_ms median " << fmt("%.3f", med_gap) << " p99 " << fmt("%.3f", pct(gaps, 0.99))
      << " longest_gap " << fmt("%.3f", pct(gaps, 1.0)) << "; missed slots "
      << (t.meta.count("missed_slots") ? t.meta.at("missed_slots") : "?") << "\n";
    if (cr >= 0) {
        o << "READ_US " << dist(t, "read_us", iv, "%.0f") << "\n";
    }
    if (cf >= 0) {
        // How often the firmware's metrics table really changed: a fresh table has new content (metrics_id).
        std::vector<double> ch_t;
        uint64_t last = 0;
        for (size_t i = 0; i < n; ++i) {
            const uint64_t id = t.words[i][cf];
            if (t.rows[i][cf] >= 0 && id != 0 && id != last) {
                ch_t.push_back(t.rows[i][ct]);
                last = id;
            }
        }
        std::vector<double> d;
        for (size_t i = 1; i < ch_t.size(); ++i) {
            d.push_back((ch_t[i] - ch_t[i - 1]) * 1e3);
        }
        o << "METRICS_REFRESH distinct tables " << ch_t.size() << "; interval_ms median " << fmt("%.2f", pct(d, 0.5))
          << " p90 " << fmt("%.2f", pct(d, 0.9)) << "\n";
    }
    // Clock bands from pp_dpm_sclk (the '*' line; it changes at nearly every sample, gaming-route.md).
    static const char * names[] = {"asleep <1.0", "1.0-1.65", "1.65-1.76", "1.76-1.9", "1.9-2.9", ">2.9"};
    double band[6] = {0, 0, 0, 0, 0, 0}, unread = 0, total = 0, awake = 0;
    for (size_t i = 0; i < n; ++i) {
        const double s = t.rows[i][cs];
        total += iv[i];
        if (s < 0) {
            unread += iv[i];
            continue;
        }
        const int b = s < 1000 ? 0 : s < 1650 ? 1 : s < 1760 ? 2 : s < 1900 ? 3 : s <= 2900 ? 4 : 5;
        band[b] += iv[i];
        if (s >= 1000) {
            awake += iv[i];
        }
    }
    o << "CLOCK_BANDS GHz (share of time, pp_dpm_sclk):";
    for (int b = 0; b < 6; ++b) {
        o << " " << names[b] << " " << fmt("%.1f", 100 * band[b] / total) << "%";
    }
    o << "; unreadable " << fmt("%.1f", 100 * unread / total) << "%\n";
    o << "AWAKE_SHARES (share of awake time): 1.65-1.9 GHz " << fmt("%.1f", awake > 0 ? 100 * (band[2] + band[3]) / awake : NAN)
      << "%, >2.9 GHz " << fmt("%.1f", awake > 0 ? 100 * band[5] / awake : NAN) << "%; awake " << fmt("%.1f", 100 * awake / total)
      << "% of the time\n";
    o << "SCLK_MHz all (time-weighted) " << dist(t, "sclk_MHz", iv, "%.0f") << "\n";
    {
        const std::string aw = dist_if(t, cs, iv, "%.0f", [cs](const std::vector<double> & r) { return r[cs] >= 1000; });
        o << "SCLK_MHz awake (time-weighted) " << (aw == "no readable samples" ? std::string("no awake samples") : aw) << "\n";
    }
    if (cg >= 0) {
        // The average column (offset 40) is reported only as a count of samples equal to the current clock: on this
        // card it has always equalled it, so it is not an independent average.
        const int ca = t.col("gfxclk_avg_MHz");
        size_t both = 0, same = 0;
        for (auto & r : t.rows) {
            if (ca >= 0 && r[cg] >= 0 && r[ca] >= 0) {
                ++both;
                same += r[cg] == r[ca];
            }
        }
        o << "GFXCLK_MHz (gpu_metrics current_gfxclk, a table that changes every few ms) " << dist(t, "gfxclk_cur_MHz", iv, "%.0f")
          << "; the average column equals it in " << same << " of " << both << " samples (not an independent average)\n";
    }
    // Runs of consecutive samples above 2.9 GHz, and below 1.9 GHz while awake.
    auto runs = [&](auto in) {
        std::vector<double> lens;
        double cur = 0;
        bool on = false;
        for (size_t i = 0; i < n; ++i) {
            if (in(t.rows[i][cs])) {
                cur += iv[i];
                on = true;
            } else if (on) {
                lens.push_back(cur);
                cur = 0;
                on = false;
            }
        }
        if (on) {
            lens.push_back(cur);
        }
        double sum = 0;
        for (double l : lens) {
            sum += l;
        }
        return std::to_string(lens.size()) + " times, total " + fmt("%.3f", sum / 1e3) + " s, median " +
               fmt("%.1f", pct(lens, 0.5)) + " ms, longest " + fmt("%.1f", pct(lens, 1.0)) + " ms";
    };
    o << "HIGH_RUNS above 2.9 GHz: " << runs([](double s) { return s > 2900; }) << "\n";
    o << "LOW_RUNS awake below 1.9 GHz: " << runs([](double s) { return s >= 1000 && s < 1900; }) << "\n";
    o << "POWER_W reported (scaled down by the power-limit mod) " << dist(t, "power_W", iv, "%.0f") << "\n";
    if (ch >= 0) {
        double a80 = 0, a90 = 0, mx = -1;
        for (size_t i = 0; i < n; ++i) {
            const double h = t.rows[i][ch];
            mx = std::max(mx, h);
            if (h > 80) {
                a80 += iv[i];
            }
            if (h > 90) {
                a90 += iv[i];
            }
        }
        o << "HOTSPOT_C max " << fmt("%.0f", mx < 0 ? NAN : mx) << "; time above 80 C " << fmt("%.3f", a80 / 1e3)
          << " s, above 90 C " << fmt("%.3f", a90 / 1e3) << " s\n";
    }
    o << "EDGE_C " << dist(t, "edge_C", iv, "%.0f") << "\n";
    o << "MEM_C " << dist(t, "mem_C", iv, "%.0f") << "\n";
    o << "FAN_RPM " << dist(t, "fan_rpm", iv, "%.0f") << "\n";
    // Throttle bits: every set bit of each word, with the number of samples that carry it.
    auto bits = [&](int c, int width, bool named) {
        std::map<int, int> cnt;
        int any = 0;
        for (size_t i = 0; i < n; ++i) {
            if (t.rows[i][c] < 0) {
                continue;
            }
            const uint64_t w = t.words[i][c];
            any += w != 0;
            for (int b = 0; b < width; ++b) {
                if (w >> b & 1) {
                    ++cnt[b];
                }
            }
        }
        std::string s = std::to_string(any) + " samples with a bit";
        for (auto & kv : cnt) {
            const char * nm = named ? throttle_name(kv.first) : nullptr;
            s += "; " + std::string(nm ? nm : "bit") + "(" + std::to_string(kv.first) + ") " + std::to_string(kv.second);
        }
        return s;
    };
    if (ci >= 0) {
        o << "THROTTLE indep_throttle_status: " << bits(ci, 64, true) << "\n";
    }
    if (cl >= 0) {
        o << "THROTTLE throttle_status (legacy word, bits not named): " << bits(cl, 32, false) << "\n";
    }
    o << "BUSY_PCT gpu " << dist(t, "busy_pct", iv, "%.0f") << "\n";
    o << "BUSY_PCT mem " << dist(t, "mem_busy_pct", iv, "%.0f") << "\n";
    o << "VGFX_mV " << dist(t, "vgfx_mV", iv, "%.0f") << "\n";
    o << "MCLK_MHz (pp_dpm_mclk: only its listed levels; missing when none is marked) " << dist(t, "mclk_MHz", iv, "%.0f")
      << "\n";
    if (t.col("uclk_cur_MHz") >= 0) {
        o << "UCLK_MHz (gpu_metrics current_uclk: the memory clock that really runs) " << dist(t, "uclk_cur_MHz", iv, "%.0f")
          << "\n";
    }
    o << "VRAM_USED_MiB " << dist(t, "vram_used_MiB", iv, "%.0f") << "\n";
    return o.str();
}

// ---------------------------------------------------------------------------------------------------------------
// Recording

const char * kCols =
    "t_s,mono_ns,wall_unix_s,read_us,sclk_lvl,sclk_MHz,mclk_lvl,mclk_MHz,gfxclk_cur_MHz,gfxclk_avg_MHz,gfx_act_pct,"
    "socket_power_W,vgfx_mV,throttle_status,indep_throttle_status,metrics_id,power_W,edge_C,hotspot_C,mem_C,fan_rpm,"
    "busy_pct,mem_busy_pct,vram_used_MiB,uclk_cur_MHz";

struct Files {
    std::string sclk, mclk, metrics, busy, mbusy, vram, power, edge, hot, memt, fan;
};

// One sample row; hot and memt return the hotspot and memory temperatures (C, -1 when unreadable) for the guard.
void sample(FILE * f, const Files & fl, int64_t t0, double & hot_out, double & memt_out) {
    const int64_t m0 = now_ns(CLOCK_MONOTONIC), w0 = now_ns(CLOCK_REALTIME);
    std::string sl = "", ml = "";
    int smhz = -1, mmhz = -1;
    if (!star_line(fl.sclk, sl, smhz)) {
        sl = "", smhz = -1;
    }
    if (!star_line(fl.mclk, ml, mmhz)) {
        ml = "", mmhz = -1;
    }
    const Metrics m = read_metrics(fl.metrics);
    const double busy = read_num(fl.busy, 1), mbusy = read_num(fl.mbusy, 1), vram = read_num(fl.vram, 1.0 / 1048576),
                 pw = read_num(fl.power, 1e-6), edge = read_num(fl.edge, 1e-3), hot = read_num(fl.hot, 1e-3),
                 memt = read_num(fl.memt, 1e-3), fan = read_num(fl.fan, 1);
    const int64_t m1 = now_ns(CLOCK_MONOTONIC);
    std::fprintf(f, "%.6f,%lld,%.6f,%.0f,%s,%d,%s,%d,", (m0 - t0) * 1e-9, (long long) m0, w0 * 1e-9, (m1 - m0) * 1e-3,
                 sl.c_str(), smhz, ml.c_str(), mmhz);
    if (m.ok) {
        std::fprintf(f, "%d,%d,%d,%d,%d,0x%x,0x%llx,0x%016llx,", m.cur_gfxclk, m.avg_gfxclk, m.gfx_act, m.socket_w, m.vgfx_mv,
                     m.thr, (unsigned long long) m.ithr, (unsigned long long) m.id);
    } else {
        std::fprintf(f, ",,,,,,,,");
    }
    std::fprintf(f, "%.1f,%.1f,%.1f,%.1f,%.0f,%.0f,%.0f,%.0f,", pw, edge, hot, memt, fan, busy, mbusy, vram);
    if (m.ok) {
        std::fprintf(f, "%d\n", m.cur_uclk);
    } else {
        std::fprintf(f, "\n");
    }
    hot_out = hot;
    memt_out = memt;
}

// The temperature guard of one load (see the usage at the top).
struct Guard {
    std::string pidfile, log;
    double stop_hot = 0, stop_mem = 0;  // stop_mem 0: no memory limit
    pid_t pid = 0;
    bool termed = false, killed = false;
    int64_t t_term = 0;
};

// True while the process exists and has not ended (a zombie waiting for its parent counts as ended).
bool running(pid_t pid) {
    const std::string s = slurp("/proc/" + std::to_string(pid) + "/stat");
    const size_t r = s.rfind(')');
    if (s.empty() || r == std::string::npos || r + 2 >= s.size()) {
        return false;
    }
    return s[r + 2] != 'Z' && s[r + 2] != 'X';
}

void guard_note(const Guard & g, const std::string & line) {
    if (!g.log.empty()) {
        if (FILE * f = std::fopen(g.log.c_str(), "a")) {
            std::fprintf(f, "%s\n", line.c_str());
            std::fclose(f);
        }
    }
    std::fprintf(stderr, "%s\n", line.c_str());
}

void guard_step(Guard & g, double hot, double memt, int64_t now) {
    if (g.pidfile.empty() || g.killed) {
        return;
    }
    if (g.pid <= 0) {
        // The script writes the load's process id once the load has started; until then there is nothing to stop.
        const std::string s = trim(slurp(g.pidfile));
        char * end = nullptr;
        const long v = std::strtol(s.c_str(), &end, 10);
        if (s.empty() || *end != 0 || v <= 1) {
            return;
        }
        g.pid = (pid_t) v;
    }
    if (!g.termed) {
        std::string why;
        if (hot < 0) {
            why = "hotspot unreadable";
        } else if (hot >= g.stop_hot) {
            why = "hotspot " + fmt("%.0f", hot) + " C";
        } else if (g.stop_mem > 0 && memt < 0) {
            why = "memory temperature unreadable";
        } else if (g.stop_mem > 0 && memt >= g.stop_mem) {
            why = "memory " + fmt("%.0f", memt) + " C";
        }
        if (why.empty()) {
            return;
        }
        ::kill(g.pid, SIGTERM);
        g.termed = true;
        g.t_term = now;
        guard_note(g, "THERMAL STOP: " + why + " (stops at hotspot " + fmt("%.0f", g.stop_hot) + " C" +
                          (g.stop_mem > 0 ? ", memory " + fmt("%.0f", g.stop_mem) + " C" : std::string()) +
                          "); TERM sent to the load (pid " + std::to_string(g.pid) + ")");
        return;
    }
    if (now - g.t_term >= 1000000000LL) {
        g.killed = true;
        if (running(g.pid)) {
            // timeout(1) leads its own process group, so the group holds the load and everything it started.
            ::kill(-g.pid, SIGKILL);
            ::kill(g.pid, SIGKILL);
            guard_note(g, "THERMAL STOP: the load was still running 1 s after TERM; KILL sent to its process group (pid " +
                              std::to_string(g.pid) + ")");
        }
    }
}

// A finite number with nothing after it ("nan", "inf", "2x" and "" are refused).
bool num_arg(const std::string & v, double & out) {
    if (v.empty()) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const double d = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != 0 || errno == ERANGE || !std::isfinite(d)) {
        return false;
    }
    out = d;
    return true;
}

int record(int argc, char ** argv) {
    std::string out, label = "none";
    double period_ms = 2, seconds = 0;
    bool die_with_parent = false;
    Guard g;
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--die-with-parent") {
            die_with_parent = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "watch_card: option %s needs a value\n", k.c_str());
            return 2;
        }
        const std::string v = argv[++i];
        bool ok = true;
        if (k == "--out") {
            out = v;
        } else if (k == "--label") {
            label = v;
        } else if (k == "--period-ms") {
            ok = num_arg(v, period_ms) && period_ms >= 0.5 && period_ms <= 60000;
        } else if (k == "--seconds") {
            ok = num_arg(v, seconds) && seconds >= 0;
        } else if (k == "--guard-pidfile") {
            g.pidfile = v;
        } else if (k == "--guard-log") {
            g.log = v;
        } else if (k == "--stop-hot") {
            ok = num_arg(v, g.stop_hot) && g.stop_hot > 0;
        } else if (k == "--stop-mem") {
            ok = num_arg(v, g.stop_mem) && g.stop_mem >= 0;
        } else {
            std::fprintf(stderr, "watch_card: unknown option %s\n", k.c_str());
            return 2;
        }
        if (!ok || v.empty()) {
            std::fprintf(stderr, "watch_card: bad value '%s' for %s\n", v.c_str(), k.c_str());
            return 2;
        }
    }
    if (out.empty() || (!g.pidfile.empty() && g.stop_hot <= 0)) {
        std::fprintf(stderr, "usage: watch_card record --out <dir> [--label L] [--period-ms 2] [--seconds 0] [--die-with-parent]\n"
                             "         [--guard-pidfile F --stop-hot C [--stop-mem C] [--guard-log F]]\n");
        return 2;
    }
    for (char & c : label) {
        if (c == ' ' || c == ',' || c == '\n') {
            c = '_';
        }
    }
    const Paths p = get_paths();
    const Files fl{p.drm + "/pp_dpm_sclk",     p.drm + "/pp_dpm_mclk",     p.drm + "/gpu_metrics",
                   p.drm + "/gpu_busy_percent", p.drm + "/mem_busy_percent", p.drm + "/mem_info_vram_used",
                   p.hw + "/power1_average",    p.hw + "/temp1_input",      p.hw + "/temp2_input",
                   p.hw + "/temp3_input",       p.hw + "/fan1_input"};
    std::string lbl;
    int mhz;
    if (!star_line(fl.sclk, lbl, mhz)) {
        std::fprintf(stderr, "watch_card: cannot read %s (DRM_DEVICE=%s)\n", fl.sclk.c_str(), p.drm.c_str());
        return 3;
    }
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGHUP, &sa, nullptr);
    if (die_with_parent) {
        const pid_t parent = getppid();
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        if (getppid() != parent) {
            g_stop = 1;  // the parent ended before the request took effect
        }
    }

    if (::mkdir(out.c_str(), 0755) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "watch_card: cannot create %s\n", out.c_str());
        return 2;
    }
    const std::string hdr = out + "/header.txt", csv = out + "/samples.csv";
    FILE * h = std::fopen(hdr.c_str(), "w");
    if (!h) {
        std::fprintf(stderr, "watch_card: cannot write %s\n", hdr.c_str());
        return 2;
    }
    std::fprintf(h, "label: %s\nperiod_ms: %g\nseconds: %g (0 = until INT, TERM or HUP)\nkernel: %s\nmesa: %s\n"
                    "drm_device: %s\nhwmon: %s\npdev: %s\nstart_utc: %s\n",
                 label.c_str(), period_ms, seconds, trim(slurp(p.proc + "/sys/kernel/osrelease")).c_str(),
                 flat(run_cmd("rpm -qa --qf '%{NAME} %{VERSION}-%{RELEASE}\\n' 'mesa-vulkan-drivers*' 'mesa-dri-drivers'"))
                     .c_str(),
                 p.drm.c_str(), p.hw.c_str(), p.pdev.c_str(), run_cmd("date -u +%FT%T.%3NZ").c_str());
    if (!g.pidfile.empty()) {
        std::fprintf(h, "guard: pid file %s; stop at hotspot %g C, memory %s; TERM, then KILL to the group after 1 s\n",
                     g.pidfile.c_str(), g.stop_hot, g.stop_mem > 0 ? (fmt("%g", g.stop_mem) + " C").c_str() : "no limit");
    }
    std::fprintf(h, "%s", conditions(p, "start").c_str());
    std::fflush(h);
    write_lines(out + "/clients-start.txt", card_clients(p));

    FILE * f = std::fopen(csv.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "watch_card: cannot write %s\n", csv.c_str());
        std::fclose(h);
        return 2;
    }
    static char buf[1 << 20];
    std::setvbuf(f, buf, _IOFBF, sizeof buf);
    std::fprintf(f, "# label=%s period_ms=%g\n%s\n", label.c_str(), period_ms, kCols);

    const int64_t per = (int64_t) std::llround(period_ms * 1e6), t0 = now_ns(CLOCK_MONOTONIC);
    const int64_t end = seconds > 0 ? t0 + (int64_t) std::llround(seconds * 1e9) : INT64_MAX;
    int64_t next = t0, missed = 0, count = 0;
    while (!g_stop) {
        double hot = -1, memt = -1;
        sample(f, fl, t0, hot, memt);
        ++count;
        guard_step(g, hot, memt, now_ns(CLOCK_MONOTONIC));
        if (count == 1) {
            // The first sample is on disk: the script may start a load now.
            std::fflush(f);
            write_lines(out + "/ready", {"first sample written"});
        } else if (count % 500 == 0) {
            std::fflush(f);
        }
        next += per;
        const int64_t now = now_ns(CLOCK_MONOTONIC);
        if (now >= next) {
            // Behind schedule: skip the slots already past instead of sampling in a burst; count them.
            const int64_t late = (now - next) / per + 1;
            missed += late;
            next += late * per;
        }
        if (next >= end) {
            break;
        }
        const timespec ts{(time_t) (next / 1000000000LL), (long) (next % 1000000000LL)};
        while (!g_stop && clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
        }
    }
    std::fprintf(f, "# samples=%lld missed_slots=%lld stopped_by=%s\n", (long long) count, (long long) missed,
                 g_stop ? "signal" : "time");
    const bool ok = std::fclose(f) == 0;
    write_lines(out + "/clients-end.txt", card_clients(p));
    std::fprintf(h, "end_utc: %s\nstopped_by: %s\n", run_cmd("date -u +%FT%T.%3NZ").c_str(), g_stop ? "signal" : "time");
    std::fprintf(h, "%s", conditions(p, "end").c_str());
    std::fclose(h);

    Table t;
    if (!ok || !load_csv(csv, t)) {
        std::fprintf(stderr, "watch_card: cannot read back %s\n", csv.c_str());
        return 2;
    }
    const std::string s = summarize(t);
    write_lines(out + "/summary.txt", {s});
    std::printf("%s", s.c_str());
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// guard: the thermal guard of one GPU job on its own (see the usage at the top)

std::string utc_now() {
    timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    tm g;
    gmtime_r(&t.tv_sec, &g);
    char b[64];
    ::strftime(b, sizeof b, "%FT%T", &g);
    char o[96];
    std::snprintf(o, sizeof o, "%s.%03ldZ", b, t.tv_nsec / 1000000);
    return o;
}

// Does anything of the job's process group still exist (a zombie that nobody has waited for counts as existing)?
bool group_exists(pid_t leader) {
    return ::kill(-leader, 0) == 0 || errno == EPERM;
}

// The job's systemd scope (systemd-run --user --scope --unit=NAME): `systemctl --user kill` signals every process in it,
// also those that left the job's process group with setsid. The call is spawned and not waited for, so the guard's own
// timing does not depend on it; the children are reaped in later rounds.
struct Scope {
    std::string name;
    std::vector<pid_t> kids;
};

void scope_signal(Scope & sc, const char * sig) {
    if (sc.name.empty()) {
        return;
    }
    const std::string opt = std::string("--signal=") + sig;
    char * const args[] = {(char *) "systemctl", (char *) "--user", (char *) "kill", (char *) opt.c_str(), (char *) "--kill-whom=all",
                           (char *) sc.name.c_str(), nullptr};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    pid_t kid = 0;
    if (posix_spawnp(&kid, "systemctl", &fa, nullptr, args, environ) == 0) {
        sc.kids.push_back(kid);
    }
    posix_spawn_file_actions_destroy(&fa);
}

void scope_reap(Scope & sc) {
    for (size_t i = 0; i < sc.kids.size();) {
        int st = 0;
        if (::waitpid(sc.kids[i], &st, WNOHANG) != 0) {
            sc.kids.erase(sc.kids.begin() + (long) i);
        } else {
            ++i;
        }
    }
}

// How many processes are left in the scope's cgroup (0 when the scope is gone or none is named).
int scope_procs(const Scope & sc) {
    if (sc.name.empty()) {
        return 0;
    }
    const std::string uid = std::to_string(::getuid());
    const std::string s = slurp("/sys/fs/cgroup/user.slice/user-" + uid + ".slice/user@" + uid + ".service/app.slice/" + sc.name +
                                "/cgroup.procs");
    return (int) std::count(s.begin(), s.end(), '\n');
}

int guard_main(int argc, char ** argv) {
    long pid = 0;
    std::string hot_file, mem_file, log, csv_path;
    Scope scope;
    double stop_hot = 0, stop_mem = 0, period_ms = 20, kill_after_ms = 500;
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        if (i + 1 >= argc) {
            std::fprintf(stderr, "watch_card guard: option %s needs a value\n", k.c_str());
            return 2;
        }
        const std::string v = argv[++i];
        double d = 0;
        bool ok = true;
        if (k == "--pid") {
            ok = num_arg(v, d) && d > 1;
            pid = (long) d;
        } else if (k == "--hot-file") {
            hot_file = v;
        } else if (k == "--mem-file") {
            mem_file = v;
        } else if (k == "--stop-hot") {
            ok = num_arg(v, stop_hot) && stop_hot > 0;
        } else if (k == "--stop-mem") {
            ok = num_arg(v, stop_mem) && stop_mem > 0;
        } else if (k == "--scope") {
            scope.name = v;
            ok = v.find('/') == std::string::npos && v.find(' ') == std::string::npos && v[0] != '-';
        } else if (k == "--period-ms") {
            ok = num_arg(v, period_ms) && period_ms >= 1 && period_ms <= 1000;
        } else if (k == "--kill-after-ms") {
            ok = num_arg(v, kill_after_ms) && kill_after_ms >= 0 && kill_after_ms <= 60000;
        } else if (k == "--log") {
            log = v;
        } else if (k == "--csv") {
            csv_path = v;
        } else {
            std::fprintf(stderr, "watch_card guard: unknown option %s\n", k.c_str());
            return 2;
        }
        if (!ok || v.empty()) {
            std::fprintf(stderr, "watch_card guard: bad value '%s' for %s\n", v.c_str(), k.c_str());
            return 2;
        }
    }
    if (pid <= 1 || hot_file.empty() || stop_hot <= 0 || (mem_file.empty() != (stop_mem <= 0))) {
        std::fprintf(stderr, "usage: watch_card guard --pid P --hot-file F --stop-hot C [--period-ms 20] [--log F] [--csv F] "
                             "[--kill-after-ms 500] [--mem-file F --stop-mem C] [--scope NAME.scope]\n");
        return 2;
    }
    const pid_t job = (pid_t) pid;
    Guard note_to;  // only for guard_note: the log path
    note_to.log = log;
    FILE * csv = nullptr;
    if (!csv_path.empty()) {
        csv = std::fopen(csv_path.c_str(), "w");
        if (!csv) {
            std::fprintf(stderr, "watch_card guard: cannot write %s\n", csv_path.c_str());
            return 2;
        }
        std::fprintf(csv, "# period_ms=%g stop_hot=%g stop_mem=%g pid=%ld scope=%s\nt_s,hotspot_C,mem_C,read_us\n", period_ms, stop_hot,
                     stop_mem, pid, scope.name.empty() ? "none" : scope.name.c_str());
    }
    const int64_t per = (int64_t) std::llround(period_ms * 1e6), kill_after = (int64_t) std::llround(kill_after_ms * 1e6);
    const int64_t t0 = now_ns(CLOCK_MONOTONIC);
    int64_t next = t0, count = 0, missed = 0, t_term = 0;
    bool termed = false, killed = false, group = false, mem_warned = false;
    std::vector<double> gaps;
    double hot_max = -1, hot_max_t = 0, mem_max = -1, last_t = -1;
    while (true) {
        const int64_t m0 = now_ns(CLOCK_MONOTONIC);
        const double hot = read_temp_strict(hot_file, 1e-3);
        const double memt = mem_file.empty() ? -1 : read_temp_strict(mem_file, 1e-3);
        const int64_t m1 = now_ns(CLOCK_MONOTONIC);
        const double t = (m0 - t0) * 1e-9;
        const bool hot_ok = hot > 0, mem_ok = memt > 0;  // a reading of 0 C from a running card is a dead sensor
        ++count;
        if (last_t >= 0) {
            gaps.push_back((t - last_t) * 1e3);
        }
        last_t = t;
        if (hot_ok && hot > hot_max) {
            hot_max = hot;
            hot_max_t = t;
        }
        if (mem_ok && memt > mem_max) {
            mem_max = memt;
        }
        if (csv) {
            std::fprintf(csv, "%.6f,%.1f,%.1f,%.0f\n", t, hot_ok ? hot : -1.0, mem_ok ? memt : -1.0, (m1 - m0) * 1e-3);
            if (count % 25 == 0 || !termed) {
                std::fflush(csv);
            }
        }
        if (!termed) {
            if (!running(job)) {
                break;  // the job has ended by itself
            }
            std::string why;
            if (!hot_ok) {
                why = "sensor unreadable (" + hot_file + " gave no usable number)";
            } else if (hot >= stop_hot) {
                why = "hotspot " + fmt("%.0f", hot) + " C";
            } else if (!mem_file.empty() && mem_ok && memt >= stop_mem) {
                why = "memory temperature " + fmt("%.0f", memt) + " C (limit " + fmt("%.0f", stop_mem) + " C)";
            }
            if (!mem_file.empty() && !mem_ok && !mem_warned && why.empty()) {
                mem_warned = true;
                guard_note(note_to, "WARNING: memory temperature unreadable (" + mem_file + " gave no usable number) at t=" + fmt("%.3f", t) +
                                        " s; the hotspot reads fine, so this is not a stop");
            }
            if (!why.empty()) {
                group = ::getpgid(job) == job;
                ::kill(group ? -job : job, SIGTERM);
                termed = true;
                t_term = now_ns(CLOCK_MONOTONIC);
                scope_signal(scope, "TERM");
                guard_note(note_to, "THERMAL STOP: " + why + " at t=" + fmt("%.3f", t) + " s of the guard (" + utc_now() +
                                        "), limit " + fmt("%.0f", stop_hot) + " C; TERM sent to the job's " +
                                        (group ? "process group " : "process (it leads no group) ") + std::to_string(pid) +
                                        (scope.name.empty() ? "" : " and to its scope " + scope.name) + ", " +
                                        fmt("%.1f", (t_term - m1) * 1e-6) + " ms after the reading");
            }
        } else {
            scope_reap(scope);
            const bool alive = (group ? group_exists(job) : running(job)) || scope_procs(scope) > 0;
            if (!alive) {
                break;
            }
            if (!killed && now_ns(CLOCK_MONOTONIC) - t_term >= kill_after) {
                killed = true;
                ::kill(group ? -job : job, SIGKILL);
                scope_signal(scope, "KILL");
                guard_note(note_to, "THERMAL STOP: still running " + fmt("%.0f", kill_after_ms) + " ms after TERM at t=" + fmt("%.3f", t) +
                                        " s (" + utc_now() + "); KILL sent to the job's " + (group ? "process group " : "process ") +
                                        std::to_string(pid) + (scope.name.empty() ? "" : " and to its scope " + scope.name));
            } else if (killed && now_ns(CLOCK_MONOTONIC) - t_term >= kill_after + 5000000000LL) {
                guard_note(note_to, "THERMAL STOP: the job's " + std::string(group ? "group" : "process") +
                                        " is still there 5 s after KILL (a zombie nobody waited for, or a process stuck in the driver); the guard ends");
                break;
            }
        }
        next += per;
        const int64_t now = now_ns(CLOCK_MONOTONIC);
        if (now >= next) {
            const int64_t late = (now - next) / per + 1;
            missed += late;
            next += late * per;
        }
        const timespec ts{(time_t) (next / 1000000000LL), (long) (next % 1000000000LL)};
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
        }
    }
    for (int i = 0; i < 50 && !scope.kids.empty(); ++i) {
        scope_reap(scope);
        if (!scope.kids.empty()) {
            ::usleep(10000);
        }
    }
    if (csv) {
        std::fprintf(csv, "# samples=%lld missed_slots=%lld\n", (long long) count, (long long) missed);
        std::fclose(csv);
    }
    guard_note(note_to, "guard: " + std::to_string(count) + " samples over " + fmt("%.3f", (now_ns(CLOCK_MONOTONIC) - t0) * 1e-9) +
                            " s, period median " + fmt("%.2f", pct(gaps, 0.5)) + " ms p99 " + fmt("%.2f", pct(gaps, 0.99)) +
                            " ms longest " + fmt("%.2f", pct(gaps, 1.0)) + " ms, missed slots " + std::to_string(missed) +
                            "; hotspot max " + fmt("%.0f", hot_max) + " C at t=" + fmt("%.3f", hot_max_t) + " s; stop limit " +
                            fmt("%.0f", stop_hot) + " C" +
                            (mem_file.empty() ? std::string() : "; memory max " + fmt("%.0f", mem_max) + " C, stop limit " + fmt("%.0f", stop_mem) + " C") +
                            (termed ? "; THERMAL STOP" : ""));
    return termed ? 8 : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "record") {
        return record(argc, argv);
    }
    if (mode == "guard") {
        return guard_main(argc, argv);
    }
    if (mode == "summarize" && argc == 3) {
        Table t;
        if (!load_csv(argv[2], t)) {
            std::fprintf(stderr, "watch_card: cannot read %s\n", argv[2]);
            return 2;
        }
        std::printf("%s", summarize(t).c_str());
        return 0;
    }
    if (mode == "clients" && argc == 3 && std::string(argv[2]) == "--pids") {
        std::set<std::pair<long, std::string>> pids;
        card_clients(get_paths(), &pids);
        for (auto & e : pids) {
            std::printf("%ld %s\n", e.first, e.second.c_str());
        }
        return 0;
    }
    if (mode == "clients" && argc == 2) {
        for (auto & l : card_clients(get_paths())) {
            std::printf("%s\n", l.c_str());
        }
        return 0;
    }
    std::fprintf(stderr,
                 "usage: watch_card record --out <dir> [--label L] [--period-ms 2] [--seconds 0]\n"
                 "       watch_card guard --pid P --hot-file F --stop-hot C [--period-ms 20] [--log F] [--csv F]\n"
                 "       watch_card summarize <samples.csv>\n"
                 "       watch_card clients [--pids]\n");
    return 2;
}
