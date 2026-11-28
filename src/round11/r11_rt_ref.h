// Round 11 item 1: the CPU side of the ray-tracing probe, with no Vulkan and no card. Scenes, the exact hit counts of the rate
// shader (rt_rate.comp), a brute-force ray against every triangle, the comparison with the card's answers, and the decision
// rules of the controls. r11_rt_selftest (CPU only) runs all of it, including the deliberate breaks.
//
// Geometry: every scene lies in the plane z = 0; rays start at z = 1 and go along -z, so a ray is a point (x, y) and a hit is
// "the point is inside the triangle". The rate shader's rays sit on a lattice of 1/16 cell with a 1/64 offset, so no ray lies on
// an edge and the hit rule is integer arithmetic:   cell (cx, cy), sub-position (a, b) in 0..15:
//   lower-left half-cell triangle (0,0),(1,0),(0,1):   hit  <=>  a + b <= 15
//   rotated 180 degrees about the cell centre:         hit  <=>  a + b >= 16
#ifndef R11_RT_REF_H
#define R11_RT_REF_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The shader's hash (rt_common.glsl, hash32): keep identical. The runner's --cpu-checks greps both files for its constants.
static inline uint32_t r11_hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

typedef struct {
    float x, y, z;
} R11V3;

// ---- scenes -------------------------------------------------------------------------------------------------------------
// Grid cells: triangle k at cell (k % nx, k / nx). rot180 = 1 puts the triangle in its rotated form (as an instance transform
// would, here baked into the vertices for the CPU copy).
static inline void r11_grid_dims(uint32_t n, uint32_t * nx, uint32_t * ny) {
    uint32_t g = (uint32_t) ceil(sqrt((double) n));
    *nx = g;
    *ny = (n + g - 1) / g;
}
static inline void r11_cell_tri(uint32_t k, uint32_t nx, int rot180, R11V3 v[3]) {
    const float cx = (float) (k % nx), cy = (float) (k / nx);
    if (!rot180) {
        v[0] = (R11V3){cx, cy, 0}, v[1] = (R11V3){cx + 1, cy, 0}, v[2] = (R11V3){cx, cy + 1, 0};
    } else {
        v[0] = (R11V3){cx + 1, cy + 1, 0}, v[1] = (R11V3){cx, cy + 1, 0}, v[2] = (R11V3){cx + 1, cy, 0};
    }
}
// S2 slivers: a thin triangle along the diagonal of its cell (axis-aligned box = the whole cell, almost all empty), or the
// same sliver laid along the cell's bottom edge (tight box). Width 1/32 of a cell.
static inline void r11_sliver_tri(uint32_t k, uint32_t nx, int along_axis, R11V3 v[3]) {
    const float cx = (float) (k % nx), cy = (float) (k / nx), w = 1.0f / 32.0f;
    if (!along_axis) {
        v[0] = (R11V3){cx, cy, 0}, v[1] = (R11V3){cx + 1 - w, cy + 1, 0}, v[2] = (R11V3){cx + 1, cy + 1 - w, 0};
    } else {
        v[0] = (R11V3){cx, cy, 0}, v[1] = (R11V3){cx + 1, cy, 0}, v[2] = (R11V3){cx + 1, cy + w, 0};
    }
}
// S0: one right triangle with legs g (cells): (0,0),(g,0),(0,g).

// ---- the rate shader, counted on the CPU --------------------------------------------------------------------------------
typedef struct {
    uint32_t nx, ny, ncell, rpt, flags, seed, kind, total_rays;
    uint32_t threads;
    int rot180;
    float tmax;
} R11Rate;

// Exact number of rays of one dispatch that hit, for kind 0 / 2 (the lattice scenes). tmax < 1 misses everything (floor cells).
static inline uint64_t r11_rate_expected_hits(const R11Rate * p) {
    if (p->tmax < 1.0f) return 0;
    uint64_t hits = 0;
    for (uint32_t gid = 0; gid < p->threads; ++gid) {
        for (uint32_t k = 0; k < p->rpt; ++k) {
            uint32_t id = (uint32_t) (((uint64_t) k * p->threads + gid) % p->total_rays);
            uint32_t cx, cy, a, b;
            if (p->flags & 2u) {
                id = r11_hash32(id ^ p->seed) % p->total_rays;
                const uint32_t hx = r11_hash32(id * 2u + 1u + p->seed), hy = r11_hash32(id * 2u + 2u + p->seed);
                cx = (hx >> 4) % p->nx;
                cy = (hy >> 4) % p->ny;
                a = hx & 15u;
                b = hy & 15u;
            } else {
                const uint32_t cell = id % p->ncell, sub = (id / p->ncell) & 255u;
                cx = cell % p->nx;
                cy = cell / p->nx;
                a = sub & 15u;
                b = sub >> 4;
            }
            if (cy * p->nx + cx >= p->ncell) continue;  // no triangle in this cell
            const int hit = p->rot180 ? (a + b >= 16) : (a + b <= 15);
            hits += (uint64_t) hit;
        }
    }
    return hits;
}

// ---- brute force: one ray against every triangle ---------------------------------------------------------------------------
// Moller-Trumbore in double; both faces count (the instance flag disables facing culls). Returns t of the nearest hit, or -1.
static inline double r11_ray_tri(const double o[3], const double d[3], double tmin, double tmax, const R11V3 v[3]) {
    const double e1[3] = {v[1].x - v[0].x, v[1].y - v[0].y, v[1].z - v[0].z};
    const double e2[3] = {v[2].x - v[0].x, v[2].y - v[0].y, v[2].z - v[0].z};
    const double p[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
    const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (fabs(det) < 1e-12) return -1;
    const double inv = 1.0 / det;
    const double s[3] = {o[0] - v[0].x, o[1] - v[0].y, o[2] - v[0].z};
    const double u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv;
    if (u < 0 || u > 1) return -1;
    const double q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
    const double w = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
    if (w < 0 || u + w > 1) return -1;
    const double t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
    return (t >= tmin && t <= tmax) ? t : -1;
}

// Nearest hit over `n` triangles (3 vertices each); *prim gets its index. Cheap 2D box test first (rays are along -z).
static inline double r11_brute(const R11V3 * tri, uint32_t n, const double o[3], const double d[3], double tmin, double tmax, int * prim) {
    double best = -1;
    *prim = -1;
    for (uint32_t i = 0; i < n; ++i) {
        const R11V3 * v = tri + 3 * (size_t) i;
        if (fabs(d[0]) < 1e-12 && fabs(d[1]) < 1e-12) {
            const float minx = fminf(v[0].x, fminf(v[1].x, v[2].x)), maxx = fmaxf(v[0].x, fmaxf(v[1].x, v[2].x));
            const float miny = fminf(v[0].y, fminf(v[1].y, v[2].y)), maxy = fmaxf(v[0].y, fmaxf(v[1].y, v[2].y));
            if (o[0] < minx || o[0] > maxx || o[1] < miny || o[1] > maxy) continue;
        }
        const double t = r11_ray_tri(o, d, tmin, tmax, v);
        if (t >= 0 && (best < 0 || t < best)) {
            best = t;
            *prim = (int) i;
        }
    }
    return best;
}

// ---- comparing the card with the CPU ---------------------------------------------------------------------------------------
// card: (t or -1, primitive, instance, flag) per ray. Hit / miss must agree exactly; t within 1e-4 relative (the plan's value).
// Returns the number of rays that disagree. A ray whose CPU hit lies within 1e-6 of a triangle edge is not asked for here:
// the check rays are centroids and lattice points, which are not on edges.
static inline uint32_t r11_compare(const float * card, const double * cpu_t, uint32_t n, uint32_t * first_bad) {
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const int card_hit = card[4 * (size_t) i] >= 0.0f, cpu_hit = cpu_t[i] >= 0;
        int ok = card_hit == cpu_hit;
        if (ok && card_hit) ok = fabs((double) card[4 * (size_t) i] - cpu_t[i]) <= 1e-4 * fmax(1.0, fabs(cpu_t[i]));
        if (!ok) {
            if (!bad && first_bad) *first_bad = i;
            ++bad;
        }
    }
    return bad;
}

// Control rules (decided here so the CPU test can break them).
// Chain check: the one-lane dependent chain must take at least 3x the time of the same number of independent rays on a whole
// wave. Below that the chain is not serial (the compiler or the hardware overlapped it) and the latency cells are void.
static inline int r11_chain_ok(double chain_ticks, double wave_ticks) { return wave_ticks > 0 && chain_ticks >= 3.0 * wave_ticks; }
// Core-clock drift: first and last repeat of a cell must be within 3 % (the plan); otherwise the cell is flagged.
static inline int r11_clock_flat(double first_mhz, double last_mhz) { return fabs(first_mhz - last_mhz) <= 0.03 * fmax(first_mhz, last_mhz); }
// A cell whose clock drifted counts toward clock_flat_failed; it fails the run only under --strict-clock (the clock is not pinned, so an auto-boost ramp is expected).
static inline int r11_clock_cell_fail(int flat, int strict) { return strict && !flat; }
// Analytic S0: legs of g cells, rays at (i + 1/4, j + 1/4) cell units: hit <=> i + j <= g - 1 -> g(g+1)/2 hits.
static inline uint64_t r11_s0_expected(uint32_t g) { return (uint64_t) g * (g + 1) / 2; }

#endif
