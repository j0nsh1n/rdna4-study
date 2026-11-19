// Leftovers C, lane A, LD-4: how long from one wave's store to another wave's load seeing it.
//
// Usage: left_comm <lds|global> [--break]
//   Built twice: CU mode (-mcumode, left_comm) and WGP mode (left_comm_wgp).
//   lds     one work-group of 8 waves; two of them are picked by placement (HW_ID1): on the same SIMD, on the two SIMDs
//           of one CU, and (WGP build) on the two CUs of the WGP. Wave A stores 2i+1 into an LDS word and polls until
//           it reads 2i+2; wave B polls for 2i+1 and answers 2i+2. EXCH round trips; A's cycle counter and REALTIME
//           give the time per round trip. Every poll loop has a cap, so a stuck wave ends with a timeout, not a hang.
//   global  two one-wave work-groups (40 KiB of LDS each, so on different CUs) exchange the same way through a word
//           in global memory with device-scope atomic loads and stores (scope:SCOPE_DEV).
//   Calibration: one wave storing then loading the same LDS word, the loaded value plus one stored next: 53 cycles per
//   link (lds item 1: store+load 36; an add between an LDS load and the next LDS op costs 17, as in its 49-cycle
//   load-add-address chain), +-2. The same exchange loop run by one wave on its own store gives the loop's floor.
//   Verdict comm_bound: every LDS round trip at least 0.9 x 72 cycles (two store-to-load links), no timeout, and the
//   calibration; times are INFO. --break claims a round trip of at most one link (36 cycles): FAIL.
#include "mem_common.h"

#include <cstdarg>
#include <map>

namespace {

constexpr uint32_t EXCH = 20000;
constexpr uint32_t CAP = 2000000;  // polls per wait before giving up

int fails = 0;
void check(const char * n, bool ok, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
void check(const char * n, bool ok, const char * fmt, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    verdict(n, ok ? "PASS" : "FAIL", b);
    fails += !ok;
}

// The exchange loops in inline assembly, so that a poll is exactly: load, wait, v_readfirstlane, wait for the SGPR,
// compare, branch (the compiler's version of the C loop was branchy scalar code that set the time). PING stores
// 2i+1 and polls for 2i+2; PONG polls for 2i+1 and stores 2i+2; SELF stores 2i+1 and polls for its own store (no
// partner: the floor of one store-to-visible-load step of this loop). ST/LD/W: the store, load and wait text.
#define COMM_LOOP(KIND, ST, LD, W)                                                                                   \
    asm volatile("s_mov_b32 %[i], 0\n s_mov_b32 %[to], 0\n"                                                        \
                 "1:\n s_lshl_b32 %[v], %[i], 1\n s_add_co_u32 %[v], %[v], 1\n"                                     \
                 KIND##_PRE(ST)                                                                                      \
                 " s_mov_b32 %[n], %[cap]\n"                                                                         \
                 "2:\n " LD " %[t], %[a]\n " W "\n v_readfirstlane_b32 %[st], %[t]\n s_wait_alu depctr_va_sdst(0)\n" \
                 " s_cmp_eq_u32 %[st], %[w]\n s_cbranch_scc1 3f\n s_sub_co_u32 %[n], %[n], 1\n s_cmp_eq_u32 %[n], 0\n" \
                 " s_cbranch_scc0 2b\n s_add_co_u32 %[to], %[to], 1\n s_branch 4f\n"                               \
                 "3:\n" KIND##_POST(ST) " s_add_co_u32 %[i], %[i], 1\n s_cmp_lt_u32 %[i], %[ex]\n s_cbranch_scc1 1b\n4:\n" \
                 : [i] "=&s"(si), [v] "=&s"(sv), [w] "=&s"(sw), [n] "=&s"(sn), [st] "=&s"(stt), [to] "=&s"(timeouts), \
                   [t] "=&v"(vt), [vv] "=&v"(vv)                                                                     \
                 : [a] "v"(addr), [cap] "s"(CAP), [ex] "s"(EXCH), BASE_OP                                            \
                 : "memory", "scc")
// ping: store v, then wait for v + 1
#define PING_PRE(ST) " v_mov_b32 %[vv], %[v]\n " ST "\n s_add_co_u32 %[w], %[v], 1\n"
#define PING_POST(ST) ""
// pong: wait for v, then store v + 1
#define PONG_PRE(ST) " s_mov_b32 %[w], %[v]\n"
#define PONG_POST(ST) " s_add_co_u32 %[w], %[w], 1\n v_mov_b32 %[vv], %[w]\n " ST "\n"
// self: store v, then wait for v
#define SELF_PRE(ST) " v_mov_b32 %[vv], %[v]\n " ST "\n s_mov_b32 %[w], %[v]\n"
#define SELF_POST(ST) ""
#define LDS_ST "ds_store_b32 %[a], %[vv]"
#define LDS_LD "ds_load_b32"
#define LDS_W "s_wait_dscnt 0"
#define GL_ST "global_store_b32 %[a], %[vv], %[base] scope:SCOPE_DEV"
#define GL_LD "global_load_b32"
#define GL_W "s_wait_loadcnt 0"

// out: [0] calibration cycles per store+load link
__global__ __launch_bounds__(32) void k_cal(uint64_t * out) {
    __shared__ uint32_t w[64];
    if (threadIdx.x < 64) w[threadIdx.x] = 0;
    __syncthreads();
    uint32_t a = 4 * (threadIdx.x % 32), v = threadIdx.x;
    const uint64_t c0 = wave_cycles();
    for (int i = 0; i < 1000; ++i) {
        // store, then load the same word; the loaded value feeds the next store (a chain of store-to-load links)
        asm volatile("ds_store_b32 %0, %1\n ds_load_b32 %1, %0\n s_wait_dscnt 0" : "+v"(a), "+v"(v)::"memory");
        v += 1;
    }
    const uint64_t c1 = wave_cycles();
    if (threadIdx.x == 0) {
        out[0] = (c1 - c0);
        out[1] = v + w[5];
    }
}

// mode: 0 same SIMD, 1 other SIMD of the CU, 2 other CU of the WGP. out: [0] A cycles, [1] A REALTIME ticks,
// [2] timeouts, [3] placement A, [4] placement B, [5] found (1 if a pair of that kind existed)
__global__ __launch_bounds__(256) void k_lds(int mode, int selfloop, uint64_t * out) {
    __shared__ volatile uint32_t flag;
    __shared__ uint32_t pl[8];
    __shared__ int roleA, roleB;
    const int w = threadIdx.x / 32;
    const uint32_t p = mem::placement();
    if (threadIdx.x % 32 == 0) pl[w] = p;
    if (threadIdx.x == 0) {
        flag = 0;
        roleA = roleB = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int a = 0; a < 8 && roleA < 0; ++a) {
            for (int b = a + 1; b < 8; ++b) {
                const uint32_t ca = (pl[a] >> 9) & 1, cb = (pl[b] >> 9) & 1, sa = (pl[a] >> 10) & 1, sb = (pl[b] >> 10) & 1;
                const bool ok = mode == 0 ? (ca == cb && sa == sb) : mode == 1 ? (ca == cb && sa != sb) : (ca != cb);
                if (ok) {
                    roleA = a;
                    roleB = b;
                    break;
                }
            }
        }
    }
    __syncthreads();
    const int A = roleA, B = roleB;
    if (w != A && w != B) return;
    uint32_t si, sv, sw, sn, stt, timeouts, vt, vv;
    const uint32_t addr = (uint32_t) (uintptr_t) &flag;
#define BASE_OP [base] "s"(0)
    if (w == A) {
        const uint64_t r0 = mem::realtime();
        const uint64_t c0 = wave_cycles();
        if (selfloop) {
            COMM_LOOP(SELF, LDS_ST, LDS_LD, LDS_W);
        } else {
            COMM_LOOP(PING, LDS_ST, LDS_LD, LDS_W);
        }
        const uint64_t c1 = wave_cycles();
        const uint64_t r1 = mem::realtime();
        if (threadIdx.x % 32 == 0) {
            out[0] = c1 - c0;
            out[1] = r1 - r0;
            out[2] += timeouts;
            out[3] = pl[A];
            out[4] = pl[B];
            out[5] = 1;
        }
    } else if (!selfloop) {
        COMM_LOOP(PONG, LDS_ST, LDS_LD, LDS_W);
        if (threadIdx.x % 32 == 0 && timeouts) atomicAdd((unsigned long long *) &out[2], (unsigned long long) timeouts);
    }
#undef BASE_OP
    (void) vt;
}

__global__ __launch_bounds__(32) void k_global(uint32_t * flag, int selfloop, uint64_t * out) {
    extern __shared__ uint32_t pad[];
    const uint32_t p = mem::placement();
    uint32_t si, sv, sw, sn, stt, timeouts = 0, vt, vv;
    const uint32_t addr = 0;
#define BASE_OP [base] "s"(flag)
    if (blockIdx.x == 0) {
        const uint64_t r0 = mem::realtime();
        const uint64_t c0 = wave_cycles();
        if (selfloop) {
            COMM_LOOP(SELF, GL_ST, GL_LD " %[t], %[a], %[base] scope:SCOPE_DEV //", GL_W);
        } else {
            COMM_LOOP(PING, GL_ST, GL_LD " %[t], %[a], %[base] scope:SCOPE_DEV //", GL_W);
        }
        const uint64_t c1 = wave_cycles();
        const uint64_t r1 = mem::realtime();
        if (threadIdx.x == 0) {
            out[0] = c1 - c0;
            out[1] = r1 - r0;
            out[2] += timeouts;
            out[3] = p;
            pad[0] = 1;
        }
    } else if (!selfloop) {
        COMM_LOOP(PONG, GL_ST, GL_LD " %[t], %[a], %[base] scope:SCOPE_DEV //", GL_W);
        if (threadIdx.x == 0) {
            out[4] = p;
            out[6] = timeouts;
        }
    }
#undef BASE_OP
    (void) vt;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? NAN : v[v.size() / 2];
}

}  // namespace

int main(int argc, char ** argv) {
    require_arch();
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool brk = mem::has_flag(argc, argv, "--break");
    uint64_t * out;
    uint32_t * flag;
    CHECK(hipMalloc(&out, 64));
    CHECK(hipMalloc(&flag, 256));
    mem::clocks("start");
    // calibration
    std::vector<double> cv;
    for (int t = 0; t < 5; ++t) {
        hipLaunchKernelGGL(k_cal, dim3(1), dim3(32), 0, 0, out);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        uint64_t o;
        CHECK(hipMemcpy(&o, out, 8, hipMemcpyDeviceToHost));
        cv.push_back((double) o / 1000.0);
    }
    const double cal = median(cv);
    std::printf("RESULT comm calibration store+load link %.2f cycles (lds item 1: 36 + 17 for the add between: 53)\n", cal);
    // the bound: a round trip is two store-to-load links (lds item 1: 36 each). The break claims the opposite: a round
    // trip of at most one link
    const double want = 72.0;
    auto bound_ok = [&](double rt) { return brk ? rt <= 36.0 : rt >= 0.9 * want; };
    if (mode == "lds") {
        const char * names[4] = {"same SIMD", "other SIMD of the CU", "other CU of the WGP", "one wave (own store)"};
        double worst = 1e9;
        uint64_t tmo = 0;
        int found = 0;
        for (int m = 0; m < 4; ++m) {
            std::vector<double> cyc, ns;
            bool had = false;
            uint32_t pa = 0, pb = 0;
            for (int t = 0; t < 5; ++t) {
                CHECK(hipMemset(out, 0, 64));
                hipLaunchKernelGGL(k_lds, dim3(1), dim3(256), 0, 0, m == 3 ? 0 : m, m == 3 ? 1 : 0, out);
                CHECK(hipGetLastError());
                CHECK(hipDeviceSynchronize());
                uint64_t o[6];
                CHECK(hipMemcpy(o, out, sizeof o, hipMemcpyDeviceToHost));
                if (!o[5]) continue;
                had = true;
                tmo += o[2];
                cyc.push_back((double) o[0] / EXCH);
                ns.push_back((double) o[1] * 10.0 / EXCH);
                pa = (uint32_t) o[3];
                pb = (uint32_t) o[4];
            }
            if (!had) {
                std::printf("RESULT comm lds %-22s no such pair of waves in this build\n", names[m]);
                continue;
            }
            const double c = median(cyc);
            if (m < 3) {
                ++found;
                worst = std::min(worst, *std::min_element(cyc.begin(), cyc.end()));
            }
            std::printf("RESULT comm lds %-22s round trip %.1f cycles (%.1f-%.1f), %.1f ns; A cu %u simd %u, B cu %u simd %u\n",
                        names[m], c, *std::min_element(cyc.begin(), cyc.end()), *std::max_element(cyc.begin(), cyc.end()),
                        median(ns), (pa >> 9) & 1, (pa >> 10) & 1, (pb >> 9) & 1, (pb >> 10) & 1);
        }
        check("comm_bound", std::fabs(cal - 53) <= 2 && found > 0 && bound_ok(worst) && tmo == 0,
              "calibration link %.2f cycles (lds item 1: store+load 36, plus 17 for an add between an LDS load and the next "
              "LDS op as in its 49-cycle chain; +- 2); smallest LDS round trip %.1f cycles (expect %s); %d placements "
              "found; timeouts %llu",
              cal, worst, brk ? "<= 36: deliberate break" : ">= 0.9 x 72", found, (unsigned long long) tmo);
    } else if (mode == "global") {
        // the one-wave floor of the same loop (store, then poll for its own store)
        std::vector<double> sc;
        for (int t = 0; t < 5; ++t) {
            CHECK(hipMemset(out, 0, 64));
            CHECK(hipMemset(flag, 0, 256));
            hipLaunchKernelGGL(k_global, dim3(1), dim3(32), 40 << 10, 0, flag, 1, out);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            uint64_t o[3];
            CHECK(hipMemcpy(o, out, sizeof o, hipMemcpyDeviceToHost));
            sc.push_back((double) o[0] / EXCH);
        }
        std::printf("RESULT comm global one wave (own store) %.1f cycles per step (%.1f-%.1f)\n", median(sc),
                    *std::min_element(sc.begin(), sc.end()), *std::max_element(sc.begin(), sc.end()));
        std::vector<double> cyc, ns;
        uint64_t tmo = 0;
        uint32_t pa = 0, pb = 0;
        for (int t = 0; t < 5; ++t) {
            CHECK(hipMemset(out, 0, 64));
            CHECK(hipMemset(flag, 0, 256));
            hipLaunchKernelGGL(k_global, dim3(2), dim3(32), 40 << 10, 0, flag, 0, out);
            CHECK(hipGetLastError());
            CHECK(hipDeviceSynchronize());
            uint64_t o[7];
            CHECK(hipMemcpy(o, out, sizeof o, hipMemcpyDeviceToHost));
            tmo += o[2] + o[6];
            cyc.push_back((double) o[0] / EXCH);
            ns.push_back((double) o[1] * 10.0 / EXCH);
            pa = (uint32_t) o[3];
            pb = (uint32_t) o[4];
        }
        std::printf("RESULT comm global round trip %.1f cycles (%.1f-%.1f), %.1f ns; A se %u wgp %u cu %u, B se %u wgp %u cu %u\n",
                    median(cyc), *std::min_element(cyc.begin(), cyc.end()), *std::max_element(cyc.begin(), cyc.end()),
                    median(ns), mem::pl_se(pa), (pa >> 5) & 15, (pa >> 9) & 1, mem::pl_se(pb), (pb >> 5) & 15, (pb >> 9) & 1);
        const double mn = *std::min_element(cyc.begin(), cyc.end());
        check("comm_bound", std::fabs(cal - 53) <= 2 && bound_ok(mn) && tmo == 0 && mem::pl_cu(pa) != mem::pl_cu(pb),
              "calibration link %.2f cycles (53 +- 2, as in lds mode); smallest global round trip %.1f cycles (expect %s); "
              "timeouts %llu; the two waves on different CUs: %s",
              cal, mn, brk ? "<= 36: deliberate break" : ">= 0.9 x 72", (unsigned long long) tmo,
              mem::pl_cu(pa) != mem::pl_cu(pb) ? "yes" : "NO");
    } else {
        std::fprintf(stderr, "usage: left_comm <lds|global> [--break]\n");
        return 2;
    }
    mem::clocks("end");
    return fails ? 1 : 0;
}
