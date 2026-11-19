// Lane 6, item 4: is each way of widening E4M3 weight bytes exact? Every method runs the same instructions as its
// timed block in cvt_time (scripts/cvt_gen_timing.py) on all 2^32 32-bit words (4 weights each, so every code at
// every byte position, with every neighbour), and each output is compared with the E4M3 value from cvt_ref.h
// (tested against llama.cpp's decoder by cvt_refcheck).
//   a  4 x v_cvt_f32_fp8 (byte_sel 0-3)            -> FP32, exact value
//   b  2 x v_cvt_pk_f32_fp8 (word select 0, 1)       -> FP32, exact value
//   c  shifts and masks (RDNA 3)                     -> FP16 pairs (b0, b2), (b1, b3), value x 2^-8
//   d  2 x v_cvt_pk_f32_fp8 + 2 x v_cvt_pk_rtz_f16   -> FP16 pairs (b0, b1), (b2, b3), exact value
//   e  FP16 table in LDS, 4 x ds_load_u16_d16(_hi)   -> FP16 pairs (b0, b1), (b2, b3), exact value
// NaN codes (0x7F, 0xFF) are counted apart: "NaN kept" when the output is a NaN.
// Usage: cvt_widen [--break]. --break writes one wrong entry (code 0x2A) into the LDS table; method e must FAIL.
#include "harness.h"
#include "cvt_ref.h"

#include <algorithm>
#include <string>

__host__ __device__ inline uint16_t f16_exact(uint32_t f) {
    const uint32_t s = (f >> 16) & 0x8000u, a = f & 0x7FFFFFFFu;
    if (a > 0x7F800000u) {
        return (uint16_t) (s | 0x7E00u);
    }
    if (a == 0) {
        return (uint16_t) s;
    }
    const int exp = (int) (a >> 23) - 127;
    const uint32_t m24 = (a & 0x7FFFFFu) | 0x800000u;
    if (exp >= -14) {
        return (uint16_t) (s | ((uint32_t) (exp + 15) << 10) | ((a >> 13) & 0x3FFu));
    }
    return (uint16_t) (s | (m24 >> (-exp - 1)));  // FP16 subnormal, units of 2^-24
}

__host__ __device__ inline bool f16_is_nan(uint32_t h) { return (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu); }

struct WStats {
    unsigned long long bad[5], nan_kept[5], nan_lost[5];
    unsigned int nex;
    uint32_t ex[16][4];  // method, word, byte position, output bits
    uint32_t lds_addr;
};

__device__ inline void record(WStats * st, int m, uint32_t x, int p, uint32_t out) {
    const unsigned k = atomicAdd(&st->nex, 1u);
    if (k < 16) {
        st->ex[k][0] = m;
        st->ex[k][1] = x;
        st->ex[k][2] = p;
        st->ex[k][3] = out;
    }
}

template <int M>
__global__ __launch_bounds__(256) void widen(WStats * st, uint64_t start, uint64_t count, int brk) {
    __shared__ uint16_t tab[256];
    if (M == 4) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            tab[i] = f16_exact(cr::decode8(cr::FP8, i));
        }
        __syncthreads();
        if (brk && threadIdx.x == 0) {
            tab[0x2A] ^= 1;
        }
        __syncthreads();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            st->lds_addr = (uint32_t) (size_t) tab;
        }
    }
    unsigned long long bad = 0, kept = 0, lost = 0;
    for (uint64_t i = start + blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; i < start + count;
         i += (uint64_t) gridDim.x * blockDim.x) {
        const uint32_t x = (uint32_t) i;
        uint32_t o[4];  // output per byte position: FP32 bits (a, b) or FP16 bits (c, d, e)
        if (M == 0) {
            asm volatile("v_cvt_f32_fp8_e64 %0, %1 byte_sel:0" : "=v"(o[0]) : "v"(x));
            asm volatile("v_cvt_f32_fp8_e64 %0, %1 byte_sel:1" : "=v"(o[1]) : "v"(x));
            asm volatile("v_cvt_f32_fp8_e64 %0, %1 byte_sel:2" : "=v"(o[2]) : "v"(x));
            asm volatile("v_cvt_f32_fp8_e64 %0, %1 byte_sel:3" : "=v"(o[3]) : "v"(x));
        } else if (M == 1 || M == 3) {
            uint64_t lo, hi;
            asm volatile("v_cvt_pk_f32_fp8_e64 %0, %1" : "=v"(lo) : "v"(x));
            asm volatile("v_cvt_pk_f32_fp8_e64 %0, %1 op_sel:[1,0]" : "=v"(hi) : "v"(x));
            if (M == 1) {
                o[0] = (uint32_t) lo;
                o[1] = (uint32_t) (lo >> 32);
                o[2] = (uint32_t) hi;
                o[3] = (uint32_t) (hi >> 32);
            } else {
                uint32_t p01, p23;
                asm volatile("v_cvt_pk_rtz_f16_f32 %0, %1, %2" : "=v"(p01) : "v"((uint32_t) lo), "v"((uint32_t) (lo >> 32)));
                asm volatile("v_cvt_pk_rtz_f16_f32 %0, %1, %2" : "=v"(p23) : "v"((uint32_t) hi), "v"((uint32_t) (hi >> 32)));
                o[0] = p01 & 0xFFFF;
                o[1] = p01 >> 16;
                o[2] = p23 & 0xFFFF;
                o[3] = p23 >> 16;
            }
        } else if (M == 2) {
            uint32_t t0, t1, t2;
            asm volatile(
                "v_lshlrev_b32 %0, 8, %3\n"
                "v_lshlrev_b32 %1, 7, %3\n"
                "v_lshrrev_b32 %2, 1, %3\n"
                "v_and_b32 %1, 0x3f803f80, %1\n"
                "v_and_b32 %2, 0x3f803f80, %2\n"
                "v_and_or_b32 %0, %0, 0x80008000, %1\n"
                "v_and_or_b32 %2, %3, 0x80008000, %2"
                : "=&v"(t0), "=&v"(t1), "=&v"(t2)
                : "v"(x));
            o[0] = t0 & 0xFFFF;
            o[2] = t0 >> 16;
            o[1] = t2 & 0xFFFF;
            o[3] = t2 >> 16;
        } else {
            uint32_t p01 = 0, p23 = 0;
            const uint32_t a0 = (x << 1) & 0x1FE, a1 = (x >> 7) & 0x1FE, a2 = (x >> 15) & 0x1FE, a3 = (x >> 23) & 0x1FE;
            asm volatile(
                "ds_load_u16_d16 %0, %2\n"
                "ds_load_u16_d16_hi %0, %3\n"
                "ds_load_u16_d16 %1, %4\n"
                "ds_load_u16_d16_hi %1, %5\n"
                "s_wait_dscnt 0"
                : "+v"(p01), "+v"(p23)
                : "v"(a0), "v"(a1), "v"(a2), "v"(a3)
                : "memory");
            o[0] = p01 & 0xFFFF;
            o[1] = p01 >> 16;
            o[2] = p23 & 0xFFFF;
            o[3] = p23 >> 16;
        }
        for (int p = 0; p < 4; ++p) {
            const uint32_t c = (x >> (8 * p)) & 0xFF;
            const uint32_t v = cr::decode8(cr::FP8, c);
            if (cr::is_nan8(cr::FP8, c)) {
                const bool isn = M <= 1 ? ((o[p] & 0x7FFFFFFFu) > 0x7F800000u) : f16_is_nan(o[p]);
                kept += isn;
                lost += !isn;
                continue;
            }
            uint32_t want;
            if (M <= 1) {
                want = v;
            } else if (M == 2) {
                // value x 2^-8: exponent lowered by 8 (every E4M3 value is far above FP32's subnormal range)
                want = (v & 0x7FFFFFFFu) ? f16_exact(v - (8u << 23)) : f16_exact(v);
            } else {
                want = f16_exact(v);
            }
            if (o[p] != want) {
                ++bad;
                record(st, M, x, p, o[p]);
            }
        }
    }
    atomicAdd(&st->bad[M], bad);
    atomicAdd(&st->nan_kept[M], kept);
    atomicAdd(&st->nan_lost[M], lost);
}

int main(int argc, char ** argv) {
    require_arch();
    const bool brk = argc > 1 && std::strcmp(argv[1], "--break") == 0;
    WStats * dst;
    CHECK(hipMalloc(&dst, sizeof(WStats)));
    CHECK(hipMemset(dst, 0, sizeof(WStats)));
    const uint64_t total = 1ull << 32, chunk = 1ull << 30;
    for (uint64_t s = 0; s < total; s += chunk) {
        widen<0><<<4096, 256>>>(dst, s, chunk, brk);
        widen<1><<<4096, 256>>>(dst, s, chunk, brk);
        widen<2><<<4096, 256>>>(dst, s, chunk, brk);
        widen<3><<<4096, 256>>>(dst, s, chunk, brk);
        widen<4><<<4096, 256>>>(dst, s, chunk, brk);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
    }
    WStats st;
    CHECK(hipMemcpy(&st, dst, sizeof st, hipMemcpyDeviceToHost));
    CHECK(hipFree(dst));
    const char * names[5] = {"a cvt_f32_fp8 x4", "b cvt_pk_f32_fp8 x2", "c shift (x 2^-8)", "d pk_f32 + pk_rtz_f16",
                             "e LDS table"};
    bool ok = st.lds_addr == 0;
    std::string text;
    char buf[200];
    for (int m = 0; m < 5; ++m) {
        std::printf("RESULT widen %-22s: %llu wrong of %llu non-NaN weights; NaN codes: %llu stay NaN, %llu do not\n",
                    names[m], st.bad[m], (1ull << 34) - st.nan_kept[m] - st.nan_lost[m], st.nan_kept[m],
                    st.nan_lost[m]);
        // c is known to turn NaN codes into +-480 x 2^-8 (rdna3-fp8 README); every other method must keep NaN
        ok = ok && st.bad[m] == 0 && (m == 2 || st.nan_lost[m] == 0);
        std::snprintf(buf, sizeof buf, "%s%c %llu wrong%s", m ? "; " : "", "abcde"[m], st.bad[m],
                      st.nan_lost[m] ? " (NaN not kept)" : "");
        text += buf;
    }
    for (unsigned k = 0; k < std::min(st.nex, 16u); ++k) {
        std::printf("  example method %c word 0x%08x byte %u output 0x%x\n", "abcde"[st.ex[k][0]], st.ex[k][1],
                    st.ex[k][2], st.ex[k][3]);
    }
    if (st.lds_addr) {
        text += "; LDS table not at address 0";
    }
    text += brk ? " (--break: one LDS table entry wrong)" : " (all 2^32 words)";
    verdict("cvt_widen", ok ? "PASS" : "FAIL", text.c_str());
    return ok ? 0 : 1;
}
