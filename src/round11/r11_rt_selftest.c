// CPU-only test of r11_rt_ref.h (no Vulkan, no card): every rule the probe's controls rest on, and every deliberate break.
// Usage: r11_rt_selftest            all checks must pass, exit 0
//        r11_rt_selftest --break N  breaks control N on purpose (1..5) and still runs everything; must exit non-zero
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "r11_rt_ref.h"

static int bad = 0;
#define T(c, msg)                                           \
    do {                                                    \
        if (!(c)) {                                         \
            printf("selftest FAIL: %s\n", msg);             \
            ++bad;                                          \
        }                                                   \
    } while (0)

// Brute-force answer for each ray of a lattice rate pass, against the real triangles: ties the integer rule to the geometry.
static uint64_t brute_rate(const R11Rate * p, const R11V3 * tri) {
    uint64_t hits = 0;
    for (uint32_t gid = 0; gid < p->threads; ++gid)
        for (uint32_t k = 0; k < p->rpt; ++k) {
            uint32_t id = (uint32_t) (((uint64_t) k * p->threads + gid) % p->total_rays), cx, cy, a, b;
            if (p->flags & 2u) {
                id = r11_hash32(id ^ p->seed) % p->total_rays;
                const uint32_t hx = r11_hash32(id * 2u + 1u + p->seed), hy = r11_hash32(id * 2u + 2u + p->seed);
                cx = (hx >> 4) % p->nx, cy = (hy >> 4) % p->ny, a = hx & 15u, b = hy & 15u;
            } else {
                const uint32_t cell = id % p->ncell, sub = (id / p->ncell) & 255u;
                cx = cell % p->nx, cy = cell / p->nx, a = sub & 15u, b = sub >> 4;
            }
            const double o[3] = {cx + (a * 4 + 1) / 64.0, cy + (b * 4 + 1) / 64.0, 1.0}, d[3] = {0, 0, -1};
            int prim;
            if (r11_brute(tri, p->ncell, o, d, 0.0, p->tmax, &prim) >= 0) ++hits;
        }
    return hits;
}

int main(int argc, char ** argv) {
    int brk = (argc >= 3 && !strcmp(argv[1], "--break")) ? atoi(argv[2]) : 0;

    // hash32: known values of lowbias32 (computed by hand from the three-step definition for input 0 and 1).
    T(r11_hash32(0) == 0, "hash32(0) is 0");
    T(r11_hash32(12345) == 0x912efcf7u, "hash32(12345) is 0x912efcf7 (independent python computation)");
    T(r11_hash32(1) == 0x688990c0u, "hash32(1) is 0x688990c0");

    // S0 analytic count, against brute force over the one triangle.
    {
        const uint32_t g = 64;
        const R11V3 tri[3] = {{0, 0, 0}, {(float) g, 0, 0}, {0, (float) g, 0}};
        uint64_t hits = 0;
        for (uint32_t j = 0; j < g; ++j)
            for (uint32_t i = 0; i < g; ++i) {
                const double o[3] = {i + 0.25, j + 0.25, 1.0}, d[3] = {0, 0, -1};
                int prim;
                if (r11_brute(tri, 1, o, d, 0.0, 10.0, &prim) >= 0) ++hits;
            }
        uint64_t want = r11_s0_expected(g);
        if (brk == 1) want += 1;  // break 1: the hand count is wrong by one; the analytic check must go red
        T(hits == want, "S0: hit count equals the hand-counted area (64 x 65 / 2 = 2080)");
        T(r11_s0_expected(64) == 2080, "S0 expected(64) is 2080");
    }

    // The lattice rule against geometry, for both ray orders, both triangle forms, a partial last row, and a miss floor.
    for (int rot = 0; rot < 2; ++rot)
        for (int shuf = 0; shuf < 2; ++shuf) {
            R11Rate p = {0};
            p.ncell = 61;  // 8 x 8 grid with 3 cells missing
            r11_grid_dims(p.ncell, &p.nx, &p.ny);
            p.rpt = 5, p.threads = 800, p.flags = shuf ? 2u : 0u, p.seed = 0x1234, p.rot180 = rot, p.tmax = 100.0f;
            p.total_rays = shuf ? 1000 : p.ncell * 256;
            R11V3 * tri = malloc(sizeof(R11V3) * 3 * p.ncell);
            for (uint32_t k = 0; k < p.ncell; ++k) r11_cell_tri(k, p.nx, rot, tri + 3 * k);
            uint64_t a = r11_rate_expected_hits(&p), b = brute_rate(&p, tri);
            if (brk == 2 && !rot && !shuf) a += 7;  // break 2: the integer rule disagrees with the geometry
            char m[96];
            snprintf(m, sizeof m, "lattice rule matches brute force (rot=%d shuffled=%d): %llu vs %llu", rot, shuf, (unsigned long long) a, (unsigned long long) b);
            T(a == b, m);
            T(a > 0, "some rays hit");
            p.tmax = 0.5f;
            T(r11_rate_expected_hits(&p) == 0, "tmax 0.5 misses everything (floor cell)");
            free(tri);
        }

    // Deliberate break (1) of the plan: move one triangle on the card only. CPU copy keeps the original.
    {
        const uint32_t n = 4096;
        uint32_t nx, ny;
        r11_grid_dims(n, &nx, &ny);
        R11V3 *cpu = malloc(sizeof(R11V3) * 3 * n), *card = malloc(sizeof(R11V3) * 3 * n);
        for (uint32_t k = 0; k < n; ++k) r11_cell_tri(k, nx, 0, cpu + 3 * k);
        memcpy(card, cpu, sizeof(R11V3) * 3 * n);
        if (brk != 3) for (int v = 0; v < 3; ++v) card[3 * 1234 + v].x += 0.5f;  // break 3 leaves the card scene unmoved: the break must then be undetected
        float * ans = calloc(4 * (size_t) n, sizeof(float));
        double * want = malloc(sizeof(double) * n);
        for (uint32_t k = 0; k < n; ++k) {  // one ray at the centroid of every triangle, as the probe does for N <= 4096
            const double o[3] = {(k % nx) + 0.25, (k / nx) + 0.25, 1.0}, d[3] = {0, 0, -1};
            int prim;
            want[k] = r11_brute(cpu, n, o, d, 0.0, 10.0, &prim);
            const double c = r11_brute(card, n, o, d, 0.0, 10.0, &prim);  // stands in for the card's answer
            ans[4 * k] = (float) c;
        }
        uint32_t first = 0;
        const uint32_t mism = r11_compare(ans, want, n, &first);
        printf("break moved-triangle: %u rays disagree (first %u)\n", mism, first);
        T(mism > 0, "moved triangle: the CPU comparison reports mismatches (a break that is not seen is a probe failure)");
        // and a clean copy compares clean
        for (uint32_t k = 0; k < n; ++k) ans[4 * k] = (float) want[k];
        T(r11_compare(ans, want, n, &first) == 0, "an exact copy compares clean");
        // tolerance edge: 5e-5 relative passes, 3e-4 fails
        ans[0] = (float) (want[0] * (1 + 5e-5));
        T(r11_compare(ans, want, 1, &first) == 0, "t within 1e-4 relative passes");
        ans[0] = (float) (want[0] * (1 + 3e-4));
        T(r11_compare(ans, want, 1, &first) == 1, "t off by 3e-4 relative fails");
        free(cpu), free(card), free(ans), free(want);
    }

    // Deliberate break (3) of the plan: one extra instance. M instances expected, rays aimed at M+1 positions. A scene with exactly M
    // instances must pass the analytic count (positive control); a scene with M+1 must fail it.
    for (int extra = 0; extra < 2; ++extra) {
        const uint32_t M = 256;
        uint32_t nx, ny;
        r11_grid_dims(M + 1, &nx, &ny);
        R11V3 * tri = malloc(sizeof(R11V3) * 3 * (M + 1));
        const uint32_t present = M + (uint32_t) extra;
        for (uint32_t k = 0; k < present; ++k) r11_cell_tri(k, nx, 0, tri + 3 * k);
        uint32_t hits = 0;
        for (uint32_t k = 0; k < M + 1; ++k) {
            const double o[3] = {(k % nx) + 0.25, (k / nx) + 0.25, 1.0}, d[3] = {0, 0, -1};
            int prim;
            if (r11_brute(tri, present, o, d, 0.0, 10.0, &prim) >= 0) ++hits;
        }
        const uint32_t expect = M + (brk == 4 ? 1u : 0u);  // break 4: the hand count of the instances is wrong by one
        const int analytic_ok = hits == expect;
        printf("%s: hits %u, analytic expectation %u -> check %s\n", extra ? "break extra-instance" : "clean instances", hits, expect, analytic_ok ? "PASS" : "FAIL");
        T(extra ? !analytic_ok : analytic_ok, extra ? "extra instance: the analytic count fails (as required)" : "clean instances: the analytic count passes");
        free(tri);
    }

    // Chain check and clock rule, with synthetic numbers.
    T(r11_chain_ok(40000, 4000), "chain 10x the wave passes");
    T(!r11_chain_ok(4100, 4000), "chain about equal to the wave fails (break 2 of the plan: all lanes active)");
    T(r11_clock_flat(2900, 2890) && !r11_clock_flat(2900, 2500), "clock drift rule: 3 %");
    T(!r11_clock_cell_fail(1, 1) && !r11_clock_cell_fail(0, 0), "a flat cell never fails; a drifting cell does not fail without --strict-clock");
    T(r11_clock_cell_fail(0, brk == 6 ? 0 : 1), "break 6: a drifting cell under --strict-clock must fail");
    if (brk == 5) T(r11_chain_ok(4100, 4000), "break 5: forced");

    printf("r11_rt_selftest%s: %d check(s) failed\n", brk ? " (break on)" : "", bad);
    return bad ? 1 : 0;
}
