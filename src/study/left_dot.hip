// Lane 6 follow-up: vector dot products, three studies in one program.
//   left_dot nan  [--break]   NaN payloads and invalid operations of the eight float dot forms, every slot position.
//   left_dot int  [--break]   v_dot4_i32_iu8 / v_dot8_i32_iu4: sign settings (neg_lo) and the clamp bit.
//   left_dot mods [--break]   which modifier, clamp and op_sel bits the assembler accepts, and what each one does.
// Each mode prints "RESULT ..." lines and ends with "VERDICT left_dot_<mode>: PASS|FAIL <text>" (exit 1 on FAIL).
// --break judges with a deliberately wrong rule, so the verdict must be FAIL.
#include "../harness.h"
#include <cstring>

#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

// One line per instruction variant: id, 16-bit-destination flag, full assembler text. Generated list; the text is
// also the lookup key (vid) so tests name a form and its modifier instead of a number.
#define VLIST(X) \
  X(0, 0, "v_dot4_f32_fp8_fp8 %0, %1, %2, %3") \
  X(1, 0, "v_dot4_f32_fp8_bf8 %0, %1, %2, %3") \
  X(2, 0, "v_dot4_f32_bf8_fp8 %0, %1, %2, %3") \
  X(3, 0, "v_dot4_f32_bf8_bf8 %0, %1, %2, %3") \
  X(4, 0, "v_dot2_f32_f16 %0, %1, %2, %3") \
  X(5, 0, "v_dot2_f32_bf16 %0, %1, %2, %3") \
  X(6, 1, "v_dot2_f16_f16 %0, %1, %2, %3") \
  X(7, 1, "v_dot2_bf16_bf16 %0, %1, %2, %3") \
  X(8, 0, "v_dot4_i32_iu8 %0, %1, %2, %3 neg_lo:[0,0,0]") \
  X(9, 0, "v_dot4_i32_iu8 %0, %1, %2, %3 neg_lo:[0,0,0] clamp") \
  X(10, 0, "v_dot4_i32_iu8 %0, %1, %2, %3 neg_lo:[1,0,0]") \
  X(11, 0, "v_dot4_i32_iu8 %0, %1, %2, %3 neg_lo:[1,0,0] clamp") \
  X(12, 0, "v_dot4_i32_iu8 %0, %1, %2, %3 neg_lo:[0,1,0]") \
  X(13, 0, "v_dot4_i32_iu8 %0, %1, %2, %3 neg_lo:[0,1,0] clamp") \
  X(14, 0, "v_dot4_i32_iu8 %0, %1, %2, %3 neg_lo:[1,1,0]") \
  X(15, 0, "v_dot4_i32_iu8 %0, %1, %2, %3 neg_lo:[1,1,0] clamp") \
  X(16, 0, "v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[0,0,0]") \
  X(17, 0, "v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[0,0,0] clamp") \
  X(18, 0, "v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[1,0,0]") \
  X(19, 0, "v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[1,0,0] clamp") \
  X(20, 0, "v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[0,1,0]") \
  X(21, 0, "v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[0,1,0] clamp") \
  X(22, 0, "v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[1,1,0]") \
  X(23, 0, "v_dot8_i32_iu4 %0, %1, %2, %3 neg_lo:[1,1,0] clamp") \
  X(24, 0, "v_dot4_f32_fp8_fp8 %0, %1, %2, %3 neg_lo:[0,0,1]") \
  X(25, 0, "v_dot4_f32_fp8_fp8 %0, %1, %2, %3 neg_hi:[0,0,1]") \
  X(26, 0, "v_dot4_f32_fp8_bf8 %0, %1, %2, %3 neg_lo:[0,0,1]") \
  X(27, 0, "v_dot4_f32_fp8_bf8 %0, %1, %2, %3 neg_hi:[0,0,1]") \
  X(28, 0, "v_dot4_f32_bf8_fp8 %0, %1, %2, %3 neg_lo:[0,0,1]") \
  X(29, 0, "v_dot4_f32_bf8_fp8 %0, %1, %2, %3 neg_hi:[0,0,1]") \
  X(30, 0, "v_dot4_f32_bf8_bf8 %0, %1, %2, %3 neg_lo:[0,0,1]") \
  X(31, 0, "v_dot4_f32_bf8_bf8 %0, %1, %2, %3 neg_hi:[0,0,1]") \
  X(32, 0, "v_dot2_f32_f16 %0, %1, %2, %3 neg_lo:[1,0,0]") \
  X(33, 0, "v_dot2_f32_f16 %0, %1, %2, %3 neg_hi:[1,0,0]") \
  X(34, 0, "v_dot2_f32_f16 %0, %1, %2, %3 op_sel:[1,0,0]") \
  X(35, 0, "v_dot2_f32_f16 %0, %1, %2, %3 neg_lo:[0,1,0]") \
  X(36, 0, "v_dot2_f32_f16 %0, %1, %2, %3 neg_hi:[0,1,0]") \
  X(37, 0, "v_dot2_f32_f16 %0, %1, %2, %3 op_sel:[0,1,0]") \
  X(38, 0, "v_dot2_f32_f16 %0, %1, %2, %3 neg_lo:[0,0,1]") \
  X(39, 0, "v_dot2_f32_f16 %0, %1, %2, %3 neg_hi:[0,0,1]") \
  X(40, 0, "v_dot2_f32_f16 %0, %1, %2, %3 op_sel:[0,0,1]") \
  X(41, 0, "v_dot2_f32_f16 %0, %1, %2, %3 clamp") \
  X(42, 0, "v_dot2_f32_f16 %0, %1, %2, %3 op_sel_hi:[0,1,1]") \
  X(43, 0, "v_dot2_f32_f16 %0, %1, %2, %3 op_sel_hi:[1,0,1]") \
  X(44, 0, "v_dot2_f32_f16 %0, %1, %2, %3 op_sel_hi:[1,1,0]") \
  X(45, 0, "v_dot2_f32_f16 %0, %1, %2, %3 op_sel:[0,0,0,1]") \
  X(46, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 neg_lo:[1,0,0]") \
  X(47, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 neg_hi:[1,0,0]") \
  X(48, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 op_sel:[1,0,0]") \
  X(49, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 neg_lo:[0,1,0]") \
  X(50, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 neg_hi:[0,1,0]") \
  X(51, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 op_sel:[0,1,0]") \
  X(52, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 neg_lo:[0,0,1]") \
  X(53, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 neg_hi:[0,0,1]") \
  X(54, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 op_sel:[0,0,1]") \
  X(55, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 clamp") \
  X(56, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 op_sel_hi:[0,1,1]") \
  X(57, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 op_sel_hi:[1,0,1]") \
  X(58, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 op_sel_hi:[1,1,0]") \
  X(59, 0, "v_dot2_f32_bf16 %0, %1, %2, %3 op_sel:[0,0,0,1]") \
  X(60, 1, "v_dot2_f16_f16 %0, %1, %2, %3 op_sel:[0,0,1]") \
  X(61, 1, "v_dot2_f16_f16 %0, %1, %2, %3 op_sel:[0,0,0,1]") \
  X(62, 1, "v_dot2_f16_f16 %0, -%1, %2, %3") \
  X(63, 1, "v_dot2_f16_f16 %0, %1, -%2, %3") \
  X(64, 1, "v_dot2_f16_f16 %0, %1, %2, -%3") \
  X(65, 1, "v_dot2_f16_f16 %0, |%1|, %2, %3") \
  X(66, 1, "v_dot2_f16_f16 %0, %1, |%2|, %3") \
  X(67, 1, "v_dot2_f16_f16 %0, %1, %2, |%3|") \
  X(68, 1, "v_dot2_bf16_bf16 %0, %1, %2, %3 op_sel:[0,0,1]") \
  X(69, 1, "v_dot2_bf16_bf16 %0, %1, %2, %3 op_sel:[0,0,0,1]") \
  X(70, 1, "v_dot2_bf16_bf16 %0, -%1, %2, %3") \
  X(71, 1, "v_dot2_bf16_bf16 %0, %1, -%2, %3") \
  X(72, 1, "v_dot2_bf16_bf16 %0, %1, %2, -%3") \
  X(73, 1, "v_dot2_bf16_bf16 %0, |%1|, %2, %3") \
  X(74, 1, "v_dot2_bf16_bf16 %0, %1, |%2|, %3") \
  X(75, 1, "v_dot2_bf16_bf16 %0, %1, %2, |%3|") \

#define NVARS 76


template <int V>
__device__ __forceinline__ uint32_t dot_v(uint32_t a, uint32_t b, uint32_t c, uint32_t d);
// The 16-bit-output forms write one half of the destination only, so d goes in as a read-write operand and the
// other half survives; the 32-bit forms overwrite it.
#define X(id, o16, s)                                                                      \
    template <>                                                                            \
    __device__ __forceinline__ uint32_t dot_v<id>(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { \
        if (o16) {                                                                         \
            asm volatile(s : "+v"(d) : "v"(a), "v"(b), "v"(c));                            \
        } else {                                                                           \
            asm volatile(s : "=v"(d) : "v"(a), "v"(b), "v"(c));                            \
        }                                                                                  \
        return d;                                                                          \
    }
VLIST(X)
#undef X

template <int V>
__global__ void kern(const uint32_t * a, const uint32_t * b, const uint32_t * c, const uint32_t * din, uint32_t * out,
                     int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = dot_v<V>(a[i], b[i], c[i], din[i]);
    }
}

struct VarInfo {
    int id, o16;
    const char * text;
};
static const VarInfo VARS[] = {
#define X(id, o16, s) {id, o16, s},
    VLIST(X)
#undef X
};

static int vid(const std::string & form, const std::string & mods = "", const std::string & ops = "%0, %1, %2, %3") {
    const std::string t = form + " " + ops + (mods.empty() ? "" : " " + mods);
    for (const VarInfo & v : VARS) {
        if (t == v.text) {
            return v.id;
        }
    }
    std::fprintf(stderr, "no variant '%s'\n", t.c_str());
    std::exit(2);
}

struct D {
    std::vector<uint32_t> a, b, c, din;
    size_t n() const { return a.size(); }
    void push(uint32_t x, uint32_t y, uint32_t z, uint32_t d = 0) {
        a.push_back(x);
        b.push_back(y);
        c.push_back(z);
        din.push_back(d);
    }
};

static uint32_t *g_da, *g_db, *g_dc, *g_din, *g_dout;
static size_t g_cap = 0;

static std::vector<uint32_t> gpu(int v, const D & d) {
    const size_t n = d.n();
    if (n > g_cap) {
        for (uint32_t * p : {g_da, g_db, g_dc, g_din, g_dout}) {
            if (p) {
                CHECK(hipFree(p));
            }
        }
        g_cap = n;
        for (uint32_t ** p : {&g_da, &g_db, &g_dc, &g_din, &g_dout}) {
            CHECK(hipMalloc(p, g_cap * 4));
        }
    }
    CHECK(hipMemcpy(g_da, d.a.data(), n * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(g_db, d.b.data(), n * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(g_dc, d.c.data(), n * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(g_din, d.din.data(), n * 4, hipMemcpyHostToDevice));
    const int blocks = (int) std::min<size_t>(4096, (n + 255) / 256);
    switch (v) {
#define X(id, o16, s)                                                      \
    case id:                                                               \
        kern<id><<<blocks, 256>>>(g_da, g_db, g_dc, g_din, g_dout, (int) n); \
        break;
        VLIST(X)
#undef X
        default:
            std::exit(2);
    }
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> o(n);
    CHECK(hipMemcpy(o.data(), g_dout, n * 4, hipMemcpyDeviceToHost));
    return o;
}

static std::string hx(uint32_t v, int digits = 8) {
    char b[16];
    std::snprintf(b, sizeof b, "0x%0*X", digits, v);
    return b;
}

static int g_fail = 0;
static void finish(const char * mode, bool ok, const std::string & text) {
    char name[64];
    std::snprintf(name, sizeof name, "left_dot_%s", mode);
    verdict(name, ok ? "PASS" : "FAIL", text.c_str());
    std::exit(ok ? 0 : 1);
}

// ---- element kinds -------------------------------------------------------------------------------------------
enum { E4, E5, H, B, F32K };
struct Kd {
    int w, eb, mb, bias;
    uint32_t one, two, zero, pinf, ninf;  // pinf = 0 and ninf = 0 where the format has no infinity
    bool hasinf;
    std::vector<uint32_t> nans;
};
static const Kd KD[5] = {
    {8, 4, 3, 7, 0x38, 0x40, 0, 0, 0, false, {0x7F, 0xFF}},
    {8, 5, 2, 15, 0x3C, 0x40, 0, 0x7C, 0xFC, true, {0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF}},
    {16, 5, 10, 15, 0x3C00, 0x4000, 0, 0x7C00, 0xFC00, true, {0x7C01, 0x7E00, 0x7FFF, 0xFC01, 0xFE55, 0xFFFF}},
    {16, 8, 7, 127, 0x3F80, 0x4000, 0, 0x7F80, 0xFF80, true, {0x7F81, 0x7FC0, 0x7FFF, 0xFF81, 0xFFC1, 0xFFFF}},
    {32, 8, 23, 127, 0x3F800000, 0x40000000, 0, 0x7F800000, 0xFF800000, true, {0x7F800001, 0xFFC12345, 0x7FA00000}},
};

static uint32_t pack(const Kd & k, int n, const uint32_t * e) {
    uint32_t r = 0;
    for (int i = 0; i < n; ++i) {
        r |= (e[i] & ((1u << k.w) - 1)) << (k.w * i);
    }
    return r;
}

// ---- mode nan ------------------------------------------------------------------------------------------------
struct FormN {
    const char * name;
    const char * op;
    int n, ka, kb, out;  // out: 0 FP32, 1 FP16, 2 BF16
};
static const FormN NFORMS[8] = {
    {"dot4_f32_fp8_fp8", "v_dot4_f32_fp8_fp8", 4, E4, E4, 0}, {"dot4_f32_fp8_bf8", "v_dot4_f32_fp8_bf8", 4, E4, E5, 0},
    {"dot4_f32_bf8_fp8", "v_dot4_f32_bf8_fp8", 4, E5, E4, 0}, {"dot4_f32_bf8_bf8", "v_dot4_f32_bf8_bf8", 4, E5, E5, 0},
    {"dot2_f32_f16", "v_dot2_f32_f16", 2, H, H, 0},          {"dot2_f32_bf16", "v_dot2_f32_bf16", 2, B, B, 0},
    {"dot2_f16_f16", "v_dot2_f16_f16", 2, H, H, 1},          {"dot2_bf16_bf16", "v_dot2_bf16_bf16", 2, B, B, 2},
};
enum { EX_NONE, EX_NOTNAN, EX_EXACT, EX_NANIN, EX_INVALID, EX_CQUIET };
struct Case {
    std::string cls, key;
    int ex = EX_NONE, s1 = -1, s2 = -1;
    uint32_t exv = 0, a = 0, b = 0, c = 0;
};

static uint32_t out_mask(int out) { return out ? 0xFFFFu : 0xFFFFFFFFu; }
static bool is_nan_out(int out, uint32_t v) {
    v &= out_mask(out);
    return out == 0 ? (v & 0x7FFFFFFFu) > 0x7F800000u : out == 1 ? (v & 0x7FFFu) > 0x7C00u : (v & 0x7FFFu) > 0x7F80u;
}
// Values of the matrix instructions (measurements/matrix-arithmetic.md): NaN input, invalid operation, NaN in C.
static uint32_t mx_nanin(int out) { return out == 0 ? 0xFFC00A3Du : out == 1 ? 0xFE3Du : 0xFFFDu; }
static uint32_t mx_invalid(const FormN & f) {
    if (f.ka == E4 || f.ka == E5) {
        return 0xFFC00A3Du;
    }
    return f.out == 0 ? 0xFFC00000u : f.out == 1 ? 0xFE00u : 0xFFC0u;
}
static uint32_t mx_cquiet(int out, uint32_t c) { return out == 0 ? c | 0x400000u : out == 1 ? (c | 0x200u) & 0xFFFFu : (c | 0x40u) & 0xFFFFu; }
static uint32_t brk_nan(int out) { return out == 0 ? 0x7FC00000u : out == 1 ? 0x7E00u : 0x7FC0u; }

static std::vector<Case> build_cases(const FormN & f) {
    const Kd &KA = KD[f.ka], &KB = KD[f.kb];
    const int n = f.n, o = f.out;
    const uint32_t cfin = o == 0 ? 0x40400000u : o == 1 ? 0x4200u : 0x4040u;
    const uint32_t cpinf = o == 0 ? 0x7F800000u : o == 1 ? 0x7C00u : 0x7F80u;
    const uint32_t cninf = o == 0 ? 0xFF800000u : o == 1 ? 0xFC00u : 0xFF80u;
    std::vector<uint32_t> cnans = o == 0   ? std::vector<uint32_t>{0x7F800001u, 0xFFC12345u, 0x7FA00000u, 0x7FC00000u, 0xFF800001u}
                                  : o == 1 ? std::vector<uint32_t>{0x7C01u, 0xFE55u, 0x7D00u, 0x7E00u}
                                           : std::vector<uint32_t>{0x7F81u, 0xFFC1u, 0x7FA0u, 0x7FC0u};
    std::vector<Case> cs;
    auto word = [&](const Kd & k, int slot, uint32_t val, uint32_t fill) {
        uint32_t e[8];
        for (int i = 0; i < n; ++i) {
            e[i] = i == slot ? val : fill;
        }
        return pack(k, n, e);
    };
    auto add = [&](const char * cls, const std::string & key, int ex, uint32_t exv, int s1, int s2, uint32_t a, uint32_t b, uint32_t c) {
        Case x;
        x.cls = cls;
        x.key = key;
        x.ex = ex;
        x.exv = exv;
        x.s1 = s1;
        x.s2 = s2;
        x.a = a;
        x.b = b;
        x.c = c;
        cs.push_back(x);
    };
    const uint32_t onesA = word(KA, -1, 0, KA.one), onesB = word(KB, -1, 0, KB.one);
    // controls: nothing here may come out as NaN
    add("ctrl_finite", "all_one", EX_NOTNAN, 0, -1, -1, onesA, onesB, cfin);
    add("ctrl_finite", "all_zero", EX_NOTNAN, 0, -1, -1, word(KA, -1, 0, 0), word(KB, -1, 0, 0), 0);
    for (int k = 0; k < n; ++k) {
        add("ctrl_finite", "two_at_slot", EX_NOTNAN, 0, k, -1, word(KA, k, KA.two, KA.one), onesB, cfin);
    }
    add("ctrl_cinf", "c_pinf", EX_EXACT, cpinf, -1, -1, onesA, onesB, cpinf);
    add("ctrl_cinf", "c_ninf", EX_EXACT, cninf, -1, -1, onesA, onesB, cninf);
    for (uint32_t p : KA.nans) {
        for (int k = 0; k < n; ++k) {
            add("nanA", hx(p, 2), EX_NANIN, 0, k, -1, word(KA, k, p, KA.one), onesB, cfin);
        }
    }
    for (uint32_t p : KB.nans) {
        for (int k = 0; k < n; ++k) {
            add("nanB", hx(p, 2), EX_NANIN, 0, k, -1, onesA, word(KB, k, p, KB.one), cfin);
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
            add("nanAB", "first_last", EX_NANIN, 0, j, k, word(KA, j, KA.nans.front(), KA.one), word(KB, k, KB.nans.back(), KB.one), cfin);
        }
    }
    for (uint32_t q : cnans) {
        add("nanC", hx(q, o ? 4 : 8), EX_CQUIET, q, -1, -1, onesA, onesB, q);
    }
    for (uint32_t p : KA.nans) {
        for (uint32_t q : cnans) {
            for (int k = 0; k < n; ++k) {
                add("nanAC", hx(p, 2) + "+" + hx(q, o ? 4 : 8), EX_NONE, 0, k, -1, word(KA, k, p, KA.one), onesB, q);
            }
        }
    }
    for (int sg = 0; sg < 2; ++sg) {
        const std::string sn = sg ? "neg" : "pos";
        if (KA.hasinf) {
            for (int k = 0; k < n; ++k) {
                add("zero_x_inf_A", sn, EX_INVALID, 0, k, -1, word(KA, k, sg ? KA.ninf : KA.pinf, KA.one), word(KB, k, 0, KB.one), cfin);
            }
        }
        if (KB.hasinf) {
            for (int k = 0; k < n; ++k) {
                add("zero_x_inf_B", sn, EX_INVALID, 0, k, -1, word(KA, k, 0, KA.one), word(KB, k, sg ? KB.ninf : KB.pinf, KB.one), cfin);
            }
        }
        if (KA.hasinf) {
            add("single_inf_A", sn, EX_EXACT, sg ? cninf : cpinf, -1, -1, word(KA, 0, sg ? KA.ninf : KA.pinf, KA.one), onesB, cfin);
            add("inf_and_C_same", sn, EX_EXACT, sg ? cninf : cpinf, -1, -1, word(KA, 0, sg ? KA.ninf : KA.pinf, KA.one), onesB, sg ? cninf : cpinf);
            add("infprod_C_opposite", sn, EX_INVALID, 0, -1, -1, word(KA, 0, sg ? KA.ninf : KA.pinf, KA.one), onesB, sg ? cpinf : cninf);
        }
        if (KB.hasinf) {
            add("single_inf_B", sn, EX_EXACT, sg ? cninf : cpinf, -1, -1, onesA, word(KB, 0, sg ? KB.ninf : KB.pinf, KB.one), cfin);
            add("inf_and_C_same", sn + "_B", EX_EXACT, sg ? cninf : cpinf, -1, -1, onesA, word(KB, 0, sg ? KB.ninf : KB.pinf, KB.one), sg ? cninf : cpinf);
            add("infprod_C_opposite", sn + "_B", EX_INVALID, 0, -1, -1, onesA, word(KB, 0, sg ? KB.ninf : KB.pinf, KB.one), sg ? cpinf : cninf);
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
            if (j == k) {
                continue;
            }
            if (KA.hasinf) {
                // +inf product in slot j and -inf product in slot k
                add("inf_minus_inf_A", "pj_nk", EX_INVALID, 0, j, k, word(KA, j, KA.pinf, KA.one) & ~(((1u << KA.w) - 1) << (KA.w * k)) | (KA.ninf << (KA.w * k)), onesB, cfin);
            }
            if (KB.hasinf) {
                add("inf_minus_inf_B", "pj_nk", EX_INVALID, 0, j, k, onesA, word(KB, j, KB.pinf, KB.one) & ~(((1u << KB.w) - 1) << (KB.w * k)) | (KB.ninf << (KB.w * k)), cfin);
            }
        }
    }
    return cs;
}

static int run_nan(bool brk) {
    int bad_repro = 0, bad_slot = 0, bad_ctrl = 0, bad_break = 0, total = 0;
    for (const FormN & f : NFORMS) {
        const std::vector<Case> cs = build_cases(f);
        const int v = vid(f.op);
        D d1, d2;
        const size_t m = cs.size();
        for (size_t i = 0; i < m; ++i) {
            const Case & x = cs[i];
            d1.push(x.a, x.b, x.c, 0xA5A50000u | (x.c & 0xFFFFu));
            const Case & y = cs[m - 1 - i];
            d2.push(y.a, y.b, y.c, 0xA5A50000u | (y.c & 0xFFFFu));
        }
        const std::vector<uint32_t> r1 = gpu(v, d1), r2raw = gpu(v, d2);
        std::vector<uint32_t> r2(m);
        for (size_t i = 0; i < m; ++i) {
            r2[i] = r2raw[m - 1 - i];
        }
        std::map<std::string, std::vector<size_t>> bycls;
        std::map<std::string, std::set<uint32_t>> bykey;
        int repro = 0, hi_ok = 0;
        for (size_t i = 0; i < m; ++i) {
            const Case & x = cs[i];
            const uint32_t r = r1[i] & out_mask(f.out);
            if (r1[i] != r2[i]) {
                ++repro;
            }
            if (f.out && (r1[i] >> 16) == 0xA5A5u) {
                ++hi_ok;
            }
            bycls[x.cls].push_back(i);
            bykey[x.cls + "|" + x.key].insert(r);
            std::printf("RESULT case form=%s cls=%s key=%s s1=%d s2=%d a=%s b=%s c=%s d=%s\n", f.name, x.cls.c_str(), x.key.c_str(),
                        x.s1, x.s2, hx(x.a).c_str(), hx(x.b).c_str(), hx(x.c).c_str(), hx(r1[i]).c_str());
        }
        int slotbad = 0;
        for (auto & kv : bykey) {
            // payload-dependent classes may legitimately differ between keys, never between slots of one key
            if (kv.second.size() > 1) {
                ++slotbad;
                std::printf("RESULT slot_inconsistent form=%s group=%s distinct=%zu\n", f.name, kv.first.c_str(), kv.second.size());
            }
        }
        int ctrlbad = 0, brkbad = 0;
        for (auto & kv : bycls) {
            std::map<uint32_t, int> words;
            int nonnan = 0, same = 0, ncmp = 0;
            for (size_t i : kv.second) {
                const Case & x = cs[i];
                const uint32_t r = r1[i] & out_mask(f.out);
                ++words[r];
                nonnan += !is_nan_out(f.out, r);
                uint32_t want = 0;
                bool cmp = true;
                switch (x.ex) {
                    case EX_NANIN: want = mx_nanin(f.out); break;
                    case EX_INVALID: want = mx_invalid(f); break;
                    case EX_CQUIET: want = mx_cquiet(f.out, x.exv); break;
                    default: cmp = false;
                }
                if (cmp) {
                    ++ncmp;
                    same += r == want;
                }
                if (x.ex == EX_NOTNAN && is_nan_out(f.out, r)) {
                    ++ctrlbad;
                }
                if (x.ex == EX_EXACT && r != x.exv) {
                    ++ctrlbad;
                }
                if (x.cls[0] == 'n' && x.ex == EX_NANIN && r != brk_nan(f.out)) {
                    ++brkbad;
                }
            }
            std::string w;
            for (auto & p : words) {
                w += hx(p.first, f.out ? 4 : 8) + "x" + std::to_string(p.second) + " ";
            }
            std::printf("RESULT table form=%s cls=%s cases=%zu distinct=%zu words=%s nonNaN=%d matrix=%s\n", f.name, kv.first.c_str(),
                        kv.second.size(), words.size(), w.c_str(), nonnan, ncmp == 0 ? "n/a" : same == ncmp ? "same" : "differs");
        }
        std::printf("RESULT form_summary form=%s cases=%zu launch_diff=%d slot_inconsistent=%d control_bad=%d high_half_kept=%d\n",
                    f.name, m, repro, slotbad, ctrlbad, hi_ok);
        bad_repro += repro;
        bad_slot += slotbad;
        bad_ctrl += ctrlbad;
        bad_break += brkbad;
        total += (int) m;
    }
    char t[256];
    if (brk) {
        std::snprintf(t, sizeof t, "break: expecting %s-style quiet NaN for every NaN-input case, %d of them differ", "0x7FC00000", bad_break);
        finish("nan", bad_break == 0, t);
    }
    std::snprintf(t, sizeof t, "%d cases x 8 forms-launches twice: launch_diff=%d slot_inconsistent=%d control_bad=%d", total, bad_repro, bad_slot, bad_ctrl);
    finish("nan", bad_repro == 0 && bad_slot == 0 && bad_ctrl == 0, t);
    return 0;
}

// ---- mode int ------------------------------------------------------------------------------------------------
static int64_t elem_of(uint32_t w, int k, int bits, bool sgn) {
    const uint32_t v = (w >> (bits * k)) & ((1u << bits) - 1);
    if (sgn && (v >> (bits - 1))) {
        return (int64_t) v - (1ll << bits);
    }
    return v;
}
static int64_t sum_prod(uint32_t a, uint32_t b, int bits, bool sa, bool sb) {
    int64_t s = 0;
    for (int k = 0; k < 32 / bits; ++k) {
        s += elem_of(a, k, bits, sa) * elem_of(b, k, bits, sb);
    }
    return s;
}
enum { R_WRAP, R_S32, R_U32, R_MIXED, R_MIXED2, NRULE };
static const char * RNAME[NRULE] = {"wrap", "sat_s32", "sat_u32", "mixed", "mixed2"};
static uint32_t sat(int64_t v, int64_t lo, int64_t hi) { return (uint32_t) std::min(std::max(v, lo), hi); }
// C is read as signed (ci) or unsigned (cu) depending on the rule; the sum is exact in 64 bits first.
static uint32_t rule(int r, int64_t s, uint32_t c, bool sa, bool sb) {
    const int64_t ci = (int32_t) c, cu = c;
    const bool uu = !sa && !sb;
    switch (r) {
        case R_WRAP: return (uint32_t) (ci + s);
        case R_S32: return sat(ci + s, INT32_MIN, INT32_MAX);
        case R_U32: return sat(cu + s, 0, UINT32_MAX);
        case R_MIXED: return uu ? sat(cu + s, 0, UINT32_MAX) : sat(ci + s, INT32_MIN, INT32_MAX);
        default: return uu ? sat(ci + s, 0, UINT32_MAX) : sat(ci + s, INT32_MIN, INT32_MAX);
    }
}
// cvt_dot.hip item 7, written the way that file does it: shift out the field, sign-extend, exact 64-bit sum, wrap.
static uint32_t item7(uint32_t a, uint32_t b, uint32_t c, int n) {
    int64_t s = (int32_t) c;
    const int w = 32 / n, sh = 32 - w;
    for (int k = 0; k < n; ++k) {
        const int32_t x = ((int32_t) (a << (sh - w * k))) >> sh, y = ((int32_t) (b << (sh - w * k))) >> sh;
        s += (int64_t) x * y;
    }
    return (uint32_t) s;
}

static uint32_t ext_word(std::mt19937_64 & r, int bits) {
    // every field at an extreme code (largest, most negative, all ones, zero), so products reach the 32-bit edges
    const uint32_t pick[4] = {(1u << (bits - 1)) - 1, 1u << (bits - 1), (1u << bits) - 1, 0};
    uint32_t w = 0;
    for (int k = 0; k < 32 / bits; ++k) {
        w |= pick[r() & 3] << (bits * k);
    }
    return w;
}
static uint32_t c_ext(std::mt19937_64 & r) {
    const uint32_t base[6] = {0x80000000u, 0x7FFFFFFFu, 0u, 0xFFFFFFFFu, 0x40000000u, 0xC0000000u};
    const uint32_t d = (uint32_t) (r() % 4096) - 2048;
    return base[r() % 6] + d;
}
static D int_rows(int bits, int rows, uint64_t seed) {
    std::mt19937_64 r(seed);
    D d;
    for (int i = 0; i < rows; ++i) {
        uint32_t a, b, c;
        switch (i & 3) {
            case 0: a = (uint32_t) r(); b = (uint32_t) r(); c = (uint32_t) r(); break;
            case 1: a = (uint32_t) r(); b = (uint32_t) r(); c = c_ext(r); break;
            case 2: a = ext_word(r, bits); b = ext_word(r, bits); c = c_ext(r); break;
            default: a = ext_word(r, bits); b = ext_word(r, bits); c = (uint32_t) r(); break;
        }
        d.push(a, b, c);
    }
    return d;
}

static int run_int(bool brk) {
    const int ROWS = 1000000;
    bool ok = true;
    std::string summary;
    for (int iu4 = 0; iu4 < 2; ++iu4) {
        const int bits = iu4 ? 4 : 8;
        const char * op = iu4 ? "v_dot8_i32_iu4" : "v_dot4_i32_iu8";
        for (int s = 0; s < 4; ++s) {
            const D d = int_rows(bits, ROWS, 77 + iu4 * 10 + s);
            const std::string neg = "neg_lo:[" + std::to_string(s & 1) + "," + std::to_string((s >> 1) & 1) + ",0]";
            const std::vector<uint32_t> nc = gpu(vid(op, neg), d), cl = gpu(vid(op, neg + " clamp"), d);
            // which bit makes which operand signed: H0 bit0 -> A, bit1 -> B; H1 the other way round
            int h0 = 0, h1 = 0, i7 = 0;
            long cnt[NRULE] = {};
            long clamp_hit = 0;
            for (int i = 0; i < ROWS; ++i) {
                const bool sa0 = s & 1, sb0 = s & 2;
                const uint32_t w0 = rule(R_WRAP, sum_prod(d.a[i], d.b[i], bits, sa0, sb0), d.c[i], sa0, sb0);
                const uint32_t w1 = rule(R_WRAP, sum_prod(d.a[i], d.b[i], bits, sb0, sa0), d.c[i], sb0, sa0);
                h0 += nc[i] != w0;
                h1 += nc[i] != w1;
                if (s == 3) {
                    i7 += nc[i] != item7(d.a[i], d.b[i], d.c[i], 32 / bits);
                }
                const int64_t sp = sum_prod(d.a[i], d.b[i], bits, sa0, sb0);
                for (int r = 0; r < NRULE; ++r) {
                    cnt[r] += cl[i] != rule(r, sp, d.c[i], sa0, sb0);
                }
                clamp_hit += cl[i] != nc[i];
            }
            std::printf("RESULT int_sign form=%s neg_lo_setting=%d (bit0=A,bit1=B) rows=%d mismatch_H0(bit0->A signed,bit1->B signed)=%d mismatch_H1(swapped)=%d%s\n", op, s,
                        ROWS, h0, h1, s == 3 ? "" : "");
            if (s == 3) {
                std::printf("RESULT int_item7 form=%s signed_x_signed_no_clamp rows=%d mismatch_vs_cvt_dot_item7_wrap=%d\n", op, ROWS, i7);
                ok = ok && i7 == 0;
            }
            std::string line;
            int accepted = -1;
            for (int r = 0; r < NRULE; ++r) {
                line += std::string(RNAME[r]) + "=" + std::to_string(cnt[r]) + " ";
                if (cnt[r] == 0 && accepted < 0 && r != R_WRAP) {
                    accepted = r;
                }
            }
            std::printf("RESULT int_clamp form=%s setting=%d rows=%d clamp_changes_result_vs_noclamp=%ld mismatches: %s\n", op, s, ROWS, clamp_hit, line.c_str());
            // zero-mismatch rules that are not the plain wrap; mixed and the signed rule coincide where an operand is signed
            std::string zeros;
            for (int r = 0; r < NRULE; ++r) {
                zeros += cnt[r] == 0 ? std::string(RNAME[r]) + " " : "";
            }
            std::printf("RESULT int_clamp_accepted form=%s setting=%d zero_mismatch_rules=%s\n", op, s, zeros.c_str());
            const int judge = brk ? R_WRAP : (accepted < 0 ? R_WRAP : accepted);
            const bool pass = cnt[judge] == 0 && h0 == 0 && clamp_hit > 1000;
            if (!pass) {
                ok = false;
            }
            summary += std::string(iu4 ? "iu4" : "iu8") + "/" + std::to_string(s) + ":" + (accepted < 0 ? "none" : RNAME[accepted]) + " ";
        }
    }
    finish("int", ok, (brk ? "break: judging clamp rows with wrap-around. " : "") + std::string("accepted clamp rule per setting: ") + summary);
    return 0;
}

// ---- mode mods -----------------------------------------------------------------------------------------------
static uint32_t gen_el(const Kd & k, std::mt19937_64 & r) {
    const uint32_t wm = k.w == 32 ? 0xFFFFFFFFu : (1u << k.w) - 1;
    const uint32_t sign = (uint32_t) (r() & 1) << (k.w - 1);
    const uint32_t emax = (1u << k.eb) - 1, mmask = (1u << k.mb) - 1;
    const int pick = (int) (r() % 10);
    if (pick < 3) {
        return (uint32_t) r() & wm;
    }
    if (pick == 3) {
        const uint32_t m = (uint32_t) r() & mmask;
        switch (r() & 3) {
            case 0: return sign;                                       // signed zero
            case 1: return sign | (emax << k.mb);                      // infinity (or a large normal in E4M3)
            case 2: return sign | (emax << k.mb) | (m ? m : 1);        // NaN with a payload
            default: return sign | m;                                  // subnormal
        }
    }
    const int e = k.bias - 6 + (int) (r() % 13);
    return sign | ((uint32_t) std::max(e, 1) << k.mb) | ((uint32_t) r() & mmask);
}
static D gen_rows(const FormN & f, int rows, uint64_t seed) {
    std::mt19937_64 r(seed);
    const Kd &KA = KD[f.ka], &KB = KD[f.kb], &KC = KD[f.out == 0 ? F32K : f.out == 1 ? H : B];
    D d;
    for (int i = 0; i < rows; ++i) {
        uint32_t a = 0, b = 0, c;
        for (int k = 0; k < f.n; ++k) {
            a |= gen_el(KA, r) << (KA.w * k);
            b |= gen_el(KB, r) << (KB.w * k);
        }
        c = f.out == 0 ? gen_el(KC, r) : (gen_el(KC, r) | (gen_el(KC, r) << 16));
        d.push(a, b, c, 0xA5A55A5Au);
    }
    return d;
}

enum { T_ID, T_FLIP, T_ABS, T_DUP, T_SWAP, T_CHI2LO };
struct Cand {
    std::string name;
    int kind = T_ID, op = 0, k = 0;
    uint32_t mask = 0;
};
static void apply_cand(const Cand & c, const FormN & f, D & d) {
    const int w = KD[f.ka].w;
    for (size_t i = 0; i < d.n(); ++i) {
        uint32_t * p = c.op == 0 ? &d.a[i] : c.op == 1 ? &d.b[i] : &d.c[i];
        switch (c.kind) {
            case T_FLIP: *p ^= c.mask; break;
            case T_ABS: *p &= ~c.mask; break;
            case T_DUP: *p = ((*p >> (w * c.k)) & 0xFFFFu) * 0x10001u; break;
            case T_SWAP: *p = (*p << 16) | (*p >> 16); break;
            case T_CHI2LO: *p = (*p & 0xFFFF0000u) | (*p >> 16); break;
            default: break;
        }
    }
}
static std::vector<Cand> cands_for(const FormN & f) {
    std::vector<Cand> v;
    v.push_back({"identity", T_ID, 0, 0, 0});
    const int w = KD[f.ka].w;
    const char * on[3] = {"A", "B", "C"};
    for (int op = 0; op < 2; ++op) {
        if (f.n == 4) {
            v.push_back({std::string("flip_") + on[op] + "_all", T_FLIP, op, 0, 0x80808080u});
        } else {
            v.push_back({std::string("flip_") + on[op] + "_lo", T_FLIP, op, 0, 0x8000u});
            v.push_back({std::string("flip_") + on[op] + "_hi", T_FLIP, op, 0, 0x80000000u});
            v.push_back({std::string("flip_") + on[op] + "_both", T_FLIP, op, 0, 0x80008000u});
            v.push_back({std::string("dup_") + on[op] + "_lo", T_DUP, op, 0, 0});
            v.push_back({std::string("dup_") + on[op] + "_hi", T_DUP, op, 1, 0});
            v.push_back({std::string("swap_") + on[op], T_SWAP, op, 0, 0});
        }
    }
    (void) w;
    if (f.out == 0) {
        v.push_back({"flip_C_bit31", T_FLIP, 2, 0, 0x80000000u});
        // VOP3P keeps absolute value in the neg_hi field, so neg_hi on the scalar C is tried as |C|
        v.push_back({"abs_C_bit31", T_ABS, 2, 0, 0x80000000u});
    } else {
        v.push_back({"flip_C_lo16", T_FLIP, 2, 0, 0x8000u});
        v.push_back({"flip_C_hi16", T_FLIP, 2, 0, 0x80000000u});
        v.push_back({"C_hi_to_lo", T_CHI2LO, 2, 0, 0});
        v.push_back({"abs_C_lo16", T_ABS, 2, 0, 0x8000u});
        for (int op = 0; op < 2; ++op) {
            v.push_back({std::string("abs_") + on[op] + "_lo", T_ABS, op, 0, 0x8000u});
            v.push_back({std::string("abs_") + on[op] + "_hi", T_ABS, op, 0, 0x80000000u});
            v.push_back({std::string("abs_") + on[op] + "_both", T_ABS, op, 0, 0x80008000u});
        }
    }
    return v;
}

// ---- assembler acceptance --------------------------------------------------------------------------------------
static std::string sh_run(const std::string & cmd) {
    std::string out;
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) {
        return "popen failed";
    }
    char buf[512];
    while (std::fgets(buf, sizeof buf, p)) {
        out += buf;
    }
    pclose(p);
    return out;
}
static std::string mc(const std::string & instr, bool & ok) {
    const char * rp = std::getenv("ROCM_PATH");
    const std::string tool = std::string(rp ? rp : ".") + "/llvm/bin/llvm-mc";
    const std::string o = sh_run("printf '%s\\n' '" + instr + "' | " + tool + " -triple=amdgcn -mcpu=gfx1201 -show-encoding 2>&1");
    const size_t e = o.find("encoding: ");
    ok = e != std::string::npos && o.find("error") == std::string::npos;
    if (ok) {
        const size_t q = o.find('\n', e);
        return o.substr(e + 10, q - e - 10);
    }
    const size_t m = o.find("error: ");
    const size_t q = o.find('\n', m);
    return m == std::string::npos ? o : o.substr(m + 7, q - m - 7);
}
static std::string sub_regs(std::string s) {
    for (int i = 3; i >= 0; --i) {
        const std::string from = "%" + std::to_string(i), to = "v" + std::to_string(i);
        size_t p;
        while ((p = s.find(from)) != std::string::npos) {
            s.replace(p, 2, to);
        }
    }
    return s;
}

static int count_ne(const std::vector<uint32_t> & x, const std::vector<uint32_t> & y, uint32_t m) {
    int c = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        c += (x[i] & m) != (y[i] & m);
    }
    return c;
}

static int run_mods(bool brk) {
    const char * FORMS_ALL[10] = {"v_dot4_f32_fp8_fp8", "v_dot4_f32_fp8_bf8", "v_dot4_f32_bf8_fp8", "v_dot4_f32_bf8_bf8", "v_dot2_f32_f16",
                                  "v_dot2_f32_bf16",    "v_dot2_f16_f16",     "v_dot2_bf16_bf16",   "v_dot4_i32_iu8",     "v_dot8_i32_iu4"};
    std::vector<std::string> mods = {"neg_lo:[1,0,0]", "neg_lo:[0,1,0]", "neg_lo:[0,0,1]", "neg_lo:[1,1,1]", "neg_hi:[1,0,0]", "neg_hi:[0,1,0]",
                                     "neg_hi:[0,0,1]", "neg_hi:[1,1,1]", "clamp",          "op_sel:[1,0,0]", "op_sel:[0,1,0]", "op_sel:[0,0,1]",
                                     "op_sel:[1,1,1]", "op_sel:[0,0,0,1]", "op_sel_hi:[0,0,0]", "op_sel_hi:[0,1,1]", "op_sel_hi:[1,0,1]",
                                     "op_sel_hi:[1,1,0]", "op_sel_hi:[1,1,1]"};
    std::vector<std::string> opsv;
    for (const char * m : {"-%s", "|%s|"}) {
        for (int i = 1; i <= 3; ++i) {
            std::string o[4] = {"v0", "v1", "v2", "v3"};
            char t[16];
            std::snprintf(t, sizeof t, m, o[i].c_str());
            o[i] = t;
            opsv.push_back(o[0] + ", " + o[1] + ", " + o[2] + ", " + o[3]);
        }
    }
    int nacc = 0, nrej = 0;
    std::map<std::string, std::string> plain_enc;
    for (const char * f : FORMS_ALL) {
        bool ok;
        plain_enc[f] = mc(std::string(f) + " v0, v1, v2, v3", ok);
        std::printf("RESULT asm form=%s mod=plain accepted enc=%s\n", f, plain_enc[f].c_str());
        for (const std::string & m : mods) {
            const std::string e = mc(std::string(f) + " v0, v1, v2, v3 " + m, ok);
            std::printf("RESULT asm form=%s mod=%s %s %s%s\n", f, m.c_str(), ok ? "accepted" : "rejected", ok ? "enc=" : "msg=", e.c_str());
            if (ok) {
                std::printf("RESULT asm_enc_vs_plain form=%s mod=%s %s\n", f, m.c_str(), e == plain_enc[f] ? "identical_to_plain (assembler drops the bit)" : "bit_set");
            }
            ok ? ++nacc : ++nrej;
        }
        for (const std::string & o : opsv) {
            const std::string e = mc(std::string(f) + " " + o, ok);
            std::printf("RESULT asm form=%s mod=\"%s\" %s %s%s\n", f, o.c_str(), ok ? "accepted" : "rejected", ok ? "enc=" : "msg=", e.c_str());
            ok ? ++nacc : ++nrej;
        }
    }
    std::printf("RESULT asm_total accepted=%d rejected=%d\n", nacc, nrej);

    const int ROWS = 1000000;
    int unclassified = 0, badctl = 0, ntested = 0;
    for (int fi = 0; fi < 8; ++fi) {
        const FormN & f = NFORMS[fi];
        const std::vector<Cand> cs = cands_for(f);
        const D x = gen_rows(f, ROWS, 4242 + fi);
        const uint32_t M = out_mask(f.out);
        const int vplain = vid(f.op);
        const std::vector<uint32_t> base = gpu(vplain, x);
        std::vector<std::vector<uint32_t>> cout(cs.size());
        std::vector<int> cand_vs_plain(cs.size());
        for (size_t j = 0; j < cs.size(); ++j) {
            D t = x;
            apply_cand(cs[j], f, t);
            cout[j] = gpu(vplain, t);
            cand_vs_plain[j] = count_ne(cout[j], base, M);
        }
        for (const VarInfo & vi : VARS) {
            const std::string txt = vi.text;
            if (txt.compare(0, std::strlen(f.op) + 1, std::string(f.op) + " ") != 0 || vi.id < 24 || txt == std::string(VARS[vplain].text)) {
                continue;
            }
            const std::string mod = txt.substr(std::strlen(f.op) + 1);
            bool ok;
            const std::string enc = mc(sub_regs(txt), ok);
            const bool dhi = txt.find("op_sel:[0,0,0,1]") != std::string::npos && f.out != 0;
            const std::vector<uint32_t> o = gpu(vi.id, x);
            ++ntested;
            if (dhi) {
                // D written to the high half: its high half must equal the plain result's low half, its low half the old destination
                int hm = 0, lm = 0, pk = 0;
                for (int i = 0; i < ROWS; ++i) {
                    hm += (o[i] >> 16) != (base[i] & 0xFFFFu);
                    lm += (o[i] & 0xFFFFu) != (x.din[i] & 0xFFFFu);
                    pk += (base[i] >> 16) != (x.din[i] >> 16);
                }
                std::printf("RESULT mod form=%s mod=\"%s\" rows=%d asm_enc=%s result_high16_equals_plain_low16_mismatch=%d result_low16_equals_old_dest_low16_mismatch=%d plain_keeps_old_high16_mismatch=%d\n",
                            f.name, mod.c_str(), ROWS, enc.c_str(), hm, lm, pk);
                std::printf("RESULT mod_class form=%s mod=\"%s\" class=%s\n", f.name, mod.c_str(), hm == 0 && lm == 0 ? "writes the result to the high half of D, keeps the low half" : "other");
                if (hm || lm) {
                    ++unclassified;
                }
                if (brk) {
                    ++badctl;
                }
                continue;
            }
            std::string zeros, others;
            int idcnt = -1, minother = -1;
            for (size_t j = 0; j < cs.size(); ++j) {
                const int c = count_ne(o, cout[j], M);
                if (j == 0) {
                    idcnt = c;
                }
                if (c == 0) {
                    zeros += cs[j].name + "(plain_vs_hypothesis_differs_in=" + std::to_string(cand_vs_plain[j]) + ") ";
                } else if (j > 0 && (minother < 0 || c < minother)) {
                    minother = c;
                }
            }
            const bool ignored = idcnt == 0;
            const bool matched = !zeros.empty();
            std::string cls = ignored ? "ignored: output equals the unmodified instruction" : matched ? "equals plain instruction on changed memory data" : "other";
            std::printf("RESULT mod form=%s mod=\"%s\" rows=%d asm_enc=%s differs_from_unmodified_rows=%d zero_mismatch_hypotheses=[%s] smallest_nonzero_other=%d class=\"%s\"\n",
                        f.name, mod.c_str(), ROWS, enc.c_str(), idcnt, zeros.c_str(), minother, cls.c_str());
            if (!matched) {
                ++unclassified;
            }
            // control: the hypothesis must not hold for the unmodified instruction (idcnt>0), unless the bit is ignored
            if (!ignored && (idcnt == 0 || minother == 0)) {
                ++badctl;
            }
            if (brk && !ignored) {
                ++badctl;  // break judges every modifier by the wrong claim "ignored"
            }
        }
    }
    // clamp on the float forms that accept it: fixed values first, then the random rows
    for (int fi = 4; fi < 6; ++fi) {
        const FormN & f = NFORMS[fi];
        const int vc = vid(f.op, "clamp"), vp = vid(f.op);
        D fx;
        const uint32_t one = KD[f.ka].one;
        const uint32_t cv[] = {0x40000000u, 0xC0400000u, 0x3E800000u, 0x3F800000u, 0x3FC00000u, 0x80000000u, 0x00000000u, 0x00000001u, 0x80000001u,
                               0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7F800001u, 0xFFC12345u, 0x3F7FFFFFu, 0x3F800001u};
        for (uint32_t c : cv) {
            fx.push(one, one, c);  // A = (1,0)... products 1*1 + 0: result is 1 + C; C chosen so the sum is easy to read
        }
        const std::vector<uint32_t> fp = gpu(vp, fx), fc = gpu(vc, fx);
        for (size_t i = 0; i < fx.n(); ++i) {
            std::printf("RESULT clamp_fixed form=%s c=%s plain=%s clamp=%s\n", f.name, hx(fx.c[i]).c_str(), hx(fp[i]).c_str(), hx(fc[i]).c_str());
        }
        const D x = gen_rows(f, ROWS, 999 + fi);
        const std::vector<uint32_t> p = gpu(vp, x), c = gpu(vc, x);
        long cnan = 0, cneg = 0, cin = 0, cabove = 0, bad_neg = 0, bad_in = 0, bad_above = 0, same_as_plain = 0;
        std::map<uint32_t, long> nanout;
        for (int i = 0; i < ROWS; ++i) {
            const uint32_t r = p[i];
            same_as_plain += c[i] == r;
            if ((r & 0x7FFFFFFFu) > 0x7F800000u) {
                ++cnan;
                ++nanout[c[i]];
            } else if (r >> 31) {
                ++cneg;
                bad_neg += c[i] != 0;
            } else if (r <= 0x3F800000u) {
                ++cin;
                bad_in += c[i] != r;
            } else {
                ++cabove;
                bad_above += c[i] != 0x3F800000u;
            }
        }
        std::string nw;
        int nn = 0;
        for (auto & kv : nanout) {
            if (nn++ < 6) {
                nw += hx(kv.first) + "x" + std::to_string(kv.second) + " ";
            }
        }
        std::printf("RESULT clamp_equals_plain form=%s rows_equal=%ld of %d (the plain-instruction results fall in: negative %ld, 0..1 %ld, above 1 %ld, NaN %ld, so the [0,1] clamp rule would change %ld rows)\n", f.name,
                    same_as_plain, ROWS, cneg, cin, cabove, cnan, cneg + cabove);
        std::printf("RESULT clamp_random_vs_01_rule form=%s rows=%d nan=%ld neg_or_negzero=%ld(bad %ld, expect 0x00000000) in_0_1=%ld(bad %ld, expect unchanged) above_1=%ld(bad %ld, expect 0x3F800000) nan_outputs(first6, distinct=%zu)=%s\n",
                    f.name, ROWS, cnan, cneg, bad_neg, cin, bad_in, cabove, bad_above, nanout.size(), nw.c_str());
        std::printf("RESULT mod_class form=%s mod=\"clamp\" class=\"%s\"\n", f.name,
                    same_as_plain == ROWS ? "ignored: all rows equal the plain result, including values below 0, above 1 and NaN" : bad_neg + bad_in + bad_above == 0 ? "clamps to [0,1]" : "other");
        if (same_as_plain != ROWS && bad_neg + bad_in + bad_above != 0) {
            ++unclassified;
        }
        if (cneg == 0 || cabove == 0 || cin == 0) {
            ++badctl;  // the control: rows exist where a [0,1] clamp would have changed the result
        }
    }
    // fixed marker rows for op_sel on the 16-bit-output forms: which half of C is read, which half of D is written
    for (int fi = 6; fi < 8; ++fi) {
        const FormN & f = NFORMS[fi];
        const uint32_t one = KD[f.ka].one, two = KD[f.ka].two, four = fi == 6 ? 0x4400u : 0x4080u;
        D d;
        d.push(one, one, two | (four << 16), 0xA5A55A5Au);  // A=(1,0)? A word has only the low half set: 1*1 + C
        const char * ms[3] = {"", "op_sel:[0,0,1]", "op_sel:[0,0,0,1]"};
        for (const char * m : ms) {
            const std::vector<uint32_t> r = gpu(vid(f.op, m), d);
            std::printf("RESULT opsel_marker form=%s mod=\"%s\" a=%s b=%s c=%s(lo=2.0 hi=4.0) dest_before=0xA5A55A5A d=%s (1+2.0 = %s, 1+4.0 = %s)\n", f.name,
                        m, hx(d.a[0]).c_str(), hx(d.b[0]).c_str(), hx(d.c[0]).c_str(), hx(r[0]).c_str(), fi == 6 ? "0x4200" : "0x4040",
                        fi == 6 ? "0x4500" : "0x40A0");
        }
    }
    char t[300];
    if (brk) {
        std::snprintf(t, sizeof t, "break: judging every modifier by the claim 'ignored'; %d modifier(s) disagree", badctl);
        finish("mods", badctl == 0, t);
    }
    std::snprintf(t, sizeof t, "%d GPU modifier variants tested on 10^6 rows; assembler accepted=%d rejected=%d; unclassified=%d, control_failures=%d", ntested,
                  nacc, nrej, unclassified, badctl);
    finish("mods", unclassified == 0 && badctl == 0, t);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: left_dot nan|int|mods [--break]\n");
        return 2;
    }
    bool brk = false;
    for (int i = 2; i < argc; ++i) {
        brk = brk || std::string(argv[i]) == "--break";
    }
    const hipDeviceProp_t prop = require_arch();
    std::printf("RESULT device %s %s\n", prop.name, prop.gcnArchName);
    const std::string m = argv[1];
    if (m == "sample" && argc >= 4) {
        // Debug aid: first rows where a variant differs from the plain instruction (left_dot sample FORM MODS).
        const FormN * f = nullptr;
        for (const FormN & x : NFORMS) {
            f = std::strcmp(x.op, argv[2]) == 0 ? &x : f;
        }
        const D x = gen_rows(*f, 200000, 5);
        const std::vector<uint32_t> p = gpu(vid(f->op), x), o = gpu(vid(f->op, argv[3]), x);
        int shown = 0;
        for (size_t i = 0; i < x.n() && shown < 40; ++i) {
            if ((p[i] ^ o[i]) & out_mask(f->out)) {
                std::printf("a=%s b=%s c=%s plain=%s mod=%s\n", hx(x.a[i]).c_str(), hx(x.b[i]).c_str(), hx(x.c[i]).c_str(), hx(p[i]).c_str(), hx(o[i]).c_str());
                ++shown;
            }
        }
        return 0;
    }
    if (m == "nan") {
        return run_nan(brk);
    } else if (m == "int") {
        return run_int(brk);
    } else if (m == "mods") {
        return run_mods(brk);
    }
    std::fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
