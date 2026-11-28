#!/usr/bin/env python3
"""Generate refined, reproducible publication figures for the RDNA 4 (RX 9070 XT) measurement study.

Every plotted value is sourced verbatim from the committed measurement write-ups under measurements/.
Outputs both high-DPI PNG (220 DPI, opaque white card background for both GitHub light & dark themes)
and vector SVG into figures/.

Usage:
  python3 -I scripts/make_figures.py [--out-dir figures] [--self-test]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Publication palette (colorblind-safe, crisp contrast on light paper surface)
C_BLUE = "#2563EB"
C_TEAL = "#0D9488"
C_AMBER = "#D97706"
C_RED = "#DC2626"
C_VIOLET = "#7C3AED"
C_SLATE = "#475569"
C_INK = "#0F172A"
C_MUTED = "#475569"
C_GRID = "#E2E8F0"
C_BG = "#FFFFFF"
C_PANEL = "#F8FAFC"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Liberation Sans"],
        "font.size": 10.0,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.0,
        "axes.labelcolor": C_INK,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "axes.facecolor": C_BG,
        "figure.facecolor": C_BG,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "grid.color": C_GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "legend.frameon": True,
        "legend.facecolor": C_BG,
        "legend.edgecolor": "#E2E8F0",
        "legend.fontsize": 8.8,
        "savefig.facecolor": C_BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
        "svg.hashsalt": "rdna4-fp8",
    })


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=C_GRID, linewidth=0.75)


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path, metadata={"Date": None})
    plt.close(fig)
    return [png_path, svg_path]


# ---------------------------------------------------------------------------
# Figure 1: Pinned Cache Latency vs MCLK + LDS CU/WGP & Bus Bandwidth
# Sources: measurements/kimi-followups.md, lds-and-special-loads.md,
#          memory-system.md, vulkan-view.md, round10.md
# ---------------------------------------------------------------------------
def fig_01_memory_hierarchy_and_mclk(out_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7), gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.subplots_adjust(wspace=0.32)

    # Panel A: Pinned Cache Latency vs MCLK (r9_lat, zero background load)
    mclk = np.array([456, 875, 1124])
    series = [
        ("GDDR6 1 GiB (random)", [384.04, 289.10, 251.32], C_RED, "o", "-"),
        ("MALL 32 MiB (random)", [345.07, 260.44, 226.88], C_AMBER, "s", "-"),
        ("GDDR6 1 GiB (page)",   [231.68, 191.13, 167.73], C_VIOLET, "o", "--"),
        ("MALL 32 MiB (page)",   [224.84, 183.96, 166.27], C_TEAL, "s", "--"),
        ("L0 16 KiB (1.98 GHz)", [40.62,  40.63,  40.62],  C_BLUE, "^", "-"),
    ]
    for label, vals, color, marker, ls in series:
        ax1.plot(mclk, vals, label=label, color=color, marker=marker, linestyle=ls,
                 linewidth=2.2, markersize=6.5, zorder=3)

    # Direct endpoint annotations (carefully staggered so zero overlap)
    offsets = {
        "GDDR6 1 GiB (random)": (8, 5, "251 ns (-35%)"),
        "MALL 32 MiB (random)": (8, -4, "227 ns (-34%)"),
        "GDDR6 1 GiB (page)":   (8, 7, "168 ns (-28%)"),
        "MALL 32 MiB (page)":   (8, -11, "166 ns (-26%)"),
        "L0 16 KiB (1.98 GHz)": (8, 0, "40.6 ns (80.6 cyc flat)"),
    }
    for label, vals, color, _, _ in series:
        dx, dy, txt = offsets[label]
        ax1.annotate(txt, (mclk[-1], vals[-1]), xytext=(dx, dy),
                     textcoords="offset points", fontsize=8.4, fontweight="semibold",
                     color=color, va="center")
    # Annotate starting values on left for the top two curves
    ax1.annotate("384 ns", (456, 384.04), xytext=(-2, 10), textcoords="offset points", fontsize=8.3, color=C_RED, ha="left")
    ax1.annotate("345 ns", (456, 345.07), xytext=(-2, -17), textcoords="offset points", fontsize=8.3, color=C_AMBER, ha="left")

    ax1.set_title("A. Pinned Cache Latency vs Memory Clock (Zero BG Load)", loc="left", pad=10)
    ax1.set_xlabel("Pinned Memory Clock (MHz, flat across all sampler rows)")
    ax1.set_ylabel("Pointer-Chase Latency (ns)")
    ax1.set_xticks([456, 875, 1124])
    ax1.set_xlim(400, 1470)
    ax1.set_ylim(0, 425)
    style_axes(ax1, "y")
    ax1.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), frameon=True)

    # Panel B: LDS CU vs WGP & Memory Bus Bandwidth (% of Ceiling / Absolute)
    labels = [
        "LDS (CU mode)\n21.9 TB/s",
        "LDS (WGP default)\n11.5 TB/s",
        "GDDR6 Vulkan\n(stock 1258 MHz)",
        "GDDR6 HIP\n(stock 1258 MHz)",
        "ld_mat2 GDDR6\n(pinned 1124 MHz)",
    ]
    # Normalize each row against its own physical ceiling for clear side-by-side comparison
    pcts = [100.0, 11.5 / 21.9 * 100.0, 636.0 / 640.0 * 100.0, 633.0 / 640.0 * 100.0, 571.0 / 575.5 * 100.0]
    val_labels = [
        "21.9 TB/s (1.90x WGP)",
        "11.5 TB/s (52.5% of CU)",
        "636 / 640 GB/s (99.4%)",
        "633 / 640 GB/s (98.9%)",
        "569–573 / 575.5 GB/s (99%)",
    ]
    colors = [C_TEAL, C_AMBER, C_BLUE, C_BLUE, C_VIOLET]
    y = np.arange(len(labels))[::-1]
    bars = ax2.barh(y, pcts, height=0.58, color=colors, edgecolor="none", zorder=3)
    ax2.axvline(100.0, color="#94A3B8", linestyle=":", linewidth=1.2, zorder=2)

    for bar, txt in zip(bars, val_labels):
        w = bar.get_width()
        if w > 72:
            ax2.text(w - 2.5, bar.get_y() + bar.get_height() / 2, txt,
                     va="center", ha="right", fontsize=8.4, fontweight="semibold", color="#FFFFFF")
        else:
            ax2.text(w + 2.0, bar.get_y() + bar.get_height() / 2, txt,
                     va="center", ha="left", fontsize=8.4, fontweight="semibold", color=C_INK)

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8.7)
    ax2.set_xlim(0, 108)
    ax2.set_xlabel("Share of Mode / Bus Ceiling (%)")
    ax2.set_title("B. LDS CU-vs-WGP Cliff & GDDR6 Bus Saturation", loc="left", pad=10)
    style_axes(ax2, "x")

    fig.suptitle("RX 9070 XT (gfx1201): Memory Hierarchy Clock Scaling & Bandwidth Cliffs",
                 fontsize=13, fontweight="bold", color=C_INK, y=1.02)
    fig.text(0.01, -0.04,
             "Sources: measurements/kimi-followups.md (pinned r9_lat, ~3.23 GHz SCLK for MALL/GDDR), lds-and-special-loads.md, memory-system.md, round10.md.",
             fontsize=8.2, color=C_MUTED)
    return save_fig(fig, out_dir, "01_memory_hierarchy_and_mclk")


# ---------------------------------------------------------------------------
# Figure 2: FP8 WMMA + Load Overlap (ld_mat2) — One-Wave Plateau & Whole-Card K=8
# Sources: measurements/round10.md (Slot 2 L0/L2 + clean pinned-slot-3 MALL/GDDR)
# ---------------------------------------------------------------------------
def fig_02_wmma_load_overlap(out_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7), gridspec_kw={"width_ratios": [1.22, 1.0]})
    fig.subplots_adjust(wspace=0.30)

    # Panel A: One-wave K=1 lookahead curve across all 4 levels
    d_vals = [0, 1, 2, 4, 8, 24, 54]
    x = np.arange(len(d_vals))
    curves = [
        ("GDDR6 1 GiB (pinned 1124 MHz)", [620.4, 306.6, 199.3, 160.4, 86.8, 34.3, 34.2], C_RED, "o"),
        ("MALL 32 MiB (pinned 1124 MHz)", [593.5, 298.9, 189.0, 113.7, 65.8, 33.4, 33.7], C_AMBER, "s"),
        ("L2 1 MiB (unpinned Slot 2)",    [268.9, 133.5,  90.1,  53.4, 33.2, 33.1, 33.1], C_TEAL, "D"),
        ("L0 8 KiB (unpinned Slot 2)",    [104.1,  52.1,  34.7,  33.1, 33.1, 33.1, 33.1], C_BLUE, "^"),
    ]
    for label, vals, color, marker in curves:
        ax1.plot(x, vals, label=label, color=color, marker=marker, linewidth=2.2, markersize=6.0, zorder=3)

    ax1.axhline(33.1, color="#94A3B8", linestyle="--", linewidth=1.1, zorder=2)
    ax1.annotate("4-level pipe floor\n33.1–34.2 cyc/WMMA\n(K=8 D=12: 11.15–11.51)",
                 xy=(5.5, 34.0), xytext=(3.65, 115),
                 arrowprops=dict(arrowstyle="->", color=C_SLATE, lw=1.1),
                 fontsize=8.4, fontweight="semibold", color=C_INK,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor=C_PANEL, edgecolor="#CBD5E1"))

    ax1.set_yscale("log")
    ax1.set_yticks([30, 50, 100, 200, 400, 650])
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"D={d}" for d in d_vals])
    ax1.set_xlabel("Load Lookahead Distance D (groups ahead of WMMA)")
    ax1.set_ylabel("Cycles per FP8 WMMA (log scale)")
    ax1.set_title("A. One-Wave Latency Hiding (K=1, R=209)", loc="left", pad=10)
    ax1.set_ylim(24, 820)
    style_axes(ax1, "y")
    ax1.legend(loc="upper right", fontsize=8.3)

    # Panel B: Whole-card K=8 D=max extra cycles over 8.05 ref (R=209 vs R=51)
    levels = ["L0\n(8 KiB)", "L2\n(1 MiB)", "MALL\n(32 MiB)", "GDDR6\n(1 GiB, 1124 MHz)"]
    r209_extra = [0.461, 0.461, 0.554, 11.234]  # L0 uses K=8 L2/L0 ref; note L0 K=8 wasn't separate -> wait! Verify exact round10.md!
    # Wait: let's check round10.md: L0 only ran K=1 (D=0..60), NOT K=8! L2, MALL, GDDR ran K=8!
    # Use ONLY exact measured K=8 cells: L2 (R=209 +0.461, R=51 +0.328), MALL (R=209 +0.554, R=51 +0.431), GDDR (R=209 +11.234, R=51 +11.863)!
    levels = ["L2 (1 MiB)\nunpinned", "MALL (32 MiB)\npinned 1124 MHz", "GDDR6 (1 GiB)\npinned 1124 MHz"]
    r209_extra = [0.461, 0.554, 11.234]   # 8.504-8.043, 8.602-8.048, 19.282-8.048
    r51_extra  = [0.328, 0.431, 11.863]   # 8.382-8.054, 8.487-8.056, 19.919-8.056
    bx = np.arange(len(levels))
    w = 0.34
    b1 = ax2.bar(bx - w / 2, r209_extra, width=w, label="R=209 (7 waves/SIMD, D=14)", color=C_BLUE, zorder=3)
    b2 = ax2.bar(bx + w / 2, r51_extra,  width=w, label="R=51 (16 waves/SIMD, D=6)", color=C_TEAL, zorder=3)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.28, f"+{h:.2f}",
                     ha="center", va="bottom", fontsize=8.6, fontweight="semibold", color=C_INK)

    ax2.annotate("GDDR6 hits physical bus wall\n(569–573 GB/s = 99% of 575.5 GB/s)",
                 xy=(2, 11.6), xytext=(0.12, 8.6),
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2),
                 fontsize=8.4, fontweight="semibold", color=C_RED,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#FECACA"))

    ax2.set_xticks(bx)
    ax2.set_xticklabels(levels, fontsize=8.8)
    ax2.set_ylabel("Extra Cycles / WMMA Above Matrix-Only Ref (~8.05)")
    ax2.set_title("B. Whole-Card GEMM-Like Load Cost (K=8, Longest D)", loc="left", pad=10)
    ax2.set_ylim(0, 14.0)
    style_axes(ax2, "y")
    ax2.legend(loc="upper left", fontsize=8.4)

    fig.suptitle("RX 9070 XT (gfx1201): Loads Beside FP8 Matrix Work (ld_mat2)",
                 fontsize=13, fontweight="bold", color=C_INK, y=1.02)
    fig.text(0.01, -0.04,
             "Sources: measurements/round10.md (Slot 2 L0/L2 + clean pinned-slot-3 MALL/GDDR; GDDR->L0 breaks closed 98.8% one-wave and 99.97% whole-card).",
             fontsize=8.2, color=C_MUTED)
    return save_fig(fig, out_dir, "02_wmma_load_overlap")


# ---------------------------------------------------------------------------
# Figure 3: Matrix Throughput (HIP vs Vulkan), Stalls, Conversions & Hazards
# Sources: measurements/matrix-timing.md, vulkan-view.md, conversions-and-dots.md,
#          compiler-and-hazards.md, ext-review-checks.md
# ---------------------------------------------------------------------------
def fig_03_matrix_rates_and_hazards(out_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(wspace=0.32)

    # Panel A: Whole-card TFLOPS (HIP vs Vulkan)
    cats = ["FP16 / BF16\n(16 cyc/inst)", "FP8 / BF8\n(8 cyc/inst)"]
    hip_vals = [211.0, 420.0]
    vk_vals  = [210.6, 421.0]
    x = np.arange(len(cats))
    w = 0.32
    b1 = ax1.bar(x - w / 2, hip_vals, width=w, label="HIP (wave32)", color=C_BLUE, zorder=3)
    b2 = ax1.bar(x + w / 2, vk_vals,  width=w, label="Vulkan / RADV (wave64)", color=C_TEAL, zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 8, f"{h:.1f}",
                     ha="center", va="bottom", fontsize=9.2, fontweight="semibold", color=C_INK)

    ax1.annotate("Sparse 2:4 forms:\n2x work in same cycles\n(8 cyc FP8, 16 cyc FP16)",
                 xy=(1.0, 421), xytext=(0.02, 330),
                 fontsize=8.4, color=C_INK,
                 bbox=dict(boxstyle="round,pad=0.38", facecolor=C_PANEL, edgecolor="#CBD5E1"))

    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=9.5)
    ax1.set_ylabel("Whole-Card Peak Rate (TFLOPS, Stock Profile)")
    ax1.set_title("A. Register-Fed Matrix Throughput: HIP vs Vulkan", loc="left", pad=10)
    ax1.set_ylim(0, 490)
    style_axes(ax1, "y")
    ax1.legend(loc="upper left", fontsize=8.8)

    # Panel B: Cycle Costs of Key Matrix / FP8 Operations & Hazards
    ops = [
        "WMMA FP8/IU8 issue\n(accumulator chain)",
        "WMMA operand-A/B\nfeed penalty (+3 cyc)",
        "WMMA FP16/BF16\nissue rate",
        "FP8 -> FP32 widen\n(per weight)",
        "FP8 -> FP16 pair\n(per weight)",
        "VALU dependent\nlatency",
        "LDS 32-bit load\nreturn latency",
        "L0 cache hit\nreturn latency",
    ]
    cycs = [8.0, 3.0, 16.0, 1.0, 1.5, 5.0, 32.0, 81.0]
    cols = [C_BLUE, C_RED, C_BLUE, C_TEAL, C_AMBER, C_SLATE, C_VIOLET, C_SLATE]
    y = np.arange(len(ops))[::-1]
    bars = ax2.barh(y, cycs, height=0.62, color=cols, zorder=3)
    for bar, v in zip(bars, cycs):
        label = f"+{v:.0f} cyc" if v == 3.0 else (f"{v:.1f} cyc" if v < 2 else f"{v:.0f} cyc")
        ax2.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2, label,
                 va="center", ha="left", fontsize=8.6, fontweight="semibold", color=C_INK)

    ax2.set_yticks(y)
    ax2.set_yticklabels(ops, fontsize=8.5)
    ax2.set_xlim(0, 96)
    ax2.set_xlabel("Clock-Invariant Cycles (Wave Cycle Counter)")
    ax2.set_title("B. Key Matrix, Conversion & Latency Cycle Costs", loc="left", pad=10)
    style_axes(ax2, "x")

    fig.suptitle("RX 9070 XT (gfx1201): Matrix Rates, Conversion Costs & Pipeline Latencies",
                 fontsize=13, fontweight="bold", color=C_INK, y=1.02)
    fig.text(0.01, -0.04,
             "Sources: measurements/matrix-timing.md, vulkan-view.md, conversions-and-dots.md, execution-core.md, lds-and-special-loads.md, memory-system.md.",
             fontsize=8.2, color=C_MUTED)
    return save_fig(fig, out_dir, "03_matrix_rates_and_hazards")


# ---------------------------------------------------------------------------
# Figure 4: 16-Wave Card-Wide Pause A/B (dpms=On vs dpms=Off) & Shuffle Control
# Sources: measurements/ext-review-checks.md (2026-11-26 same-boot A/B run)
# ---------------------------------------------------------------------------
def fig_04_dpms_pauses_ab(out_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4), gridspec_kw={"width_ratios": [1.1, 1.0]})
    fig.subplots_adjust(wspace=0.30)

    # Panel A: Median events/wave in 0.500 s
    runs = ["ON\nRun 1", "ON\nRun 2", "ON\nRun 3", "OFF\nRun 1", "OFF\nRun 2", "OFF\nRun 3"]
    counts = [683, 1029, 776, 0, 0, 0]
    cols = [C_RED, C_RED, C_RED, C_TEAL, C_TEAL, C_TEAL]
    x = np.arange(len(runs))
    bars = ax1.bar(x, counts, width=0.58, color=cols, zorder=3)
    ax1.axvline(2.5, color="#94A3B8", linestyle="--", linewidth=1.2)
    ax1.text(1.0, 1185, "Displays ON (dpms=On)", ha="center", fontsize=9.0, fontweight="semibold", color=C_RED)
    ax1.text(4.0, 1185, "60 s later: Displays OFF", ha="center", fontsize=9.0, fontweight="semibold", color=C_TEAL)

    for bar, v in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 22, f"{v:,}" if v > 0 else "0",
                 ha="center", va="bottom", fontsize=9.2, fontweight="semibold",
                 color=C_INK if v > 0 else C_TEAL)

    ax1.set_xticks(x)
    ax1.set_xticklabels(runs, fontsize=9.0)
    ax1.set_ylabel("Median Stall Events / Wave (per 0.500 s)")
    ax1.set_title("A. Same-Boot 16-Wave Pause Count (60 s Apart)", loc="left", pad=10)
    ax1.set_ylim(0, 1300)
    style_axes(ax1, "y")

    # Panel B: Real ±1 us Coincidence vs 1,000 Circular Shuffles
    r_idx = np.arange(3)
    same_se = [95.70, 97.46, 96.50]
    diff_se = [95.71, 97.42, 96.77]
    shuf_95 = [0.30, 0.39, 0.38]   # 95% upper bound <= 0.39% (same-SE raw: 0.30, 0.39, 0.38%)
    w = 0.26
    ax2.bar(r_idx - w, same_se, width=w, label="Same SE (±1 µs)", color=C_BLUE, zorder=3)
    ax2.bar(r_idx,     diff_se, width=w, label="Diff SE (±1 µs)", color=C_VIOLET, zorder=3)
    ax2.bar(r_idx + w, shuf_95, width=w, label="1,000 Shuffles (95% upper ≤ 0.39%)", color=C_SLATE, zorder=3)

    for i in range(3):
        ax2.text(r_idx[i] - w / 2, max(same_se[i], diff_se[i]) + 2.0, f"{diff_se[i]:.1f}%",
                 ha="center", va="bottom", fontsize=8.6, fontweight="semibold", color=C_INK)
        ax2.text(r_idx[i] + w, shuf_95[i] + 2.5, "<0.4%",
                 ha="center", va="bottom", fontsize=8.2, color=C_MUTED)

    ax2.set_xticks(r_idx)
    ax2.set_xticklabels(["ON Run 1", "ON Run 2", "ON Run 3"])
    ax2.set_ylabel("Event Coincidence (%)")
    ax2.set_title("B. Cross-SE Coincidence vs Chance (1,000 Shuffles)", loc="left", pad=10)
    ax2.set_ylim(0, 128)
    style_axes(ax2, "y")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=8.1, columnspacing=1.0)

    fig.suptitle("RX 9070 XT (gfx1201): Active Display DPMS Drives Card-Wide Pauses",
                 fontsize=13, fontweight="bold", color=C_INK, y=1.02)
    fig.text(0.01, -0.04,
             "Source: measurements/ext-review-checks.md (2026-11-26 same-boot, same-binary A/B; hw_changed=0 on all events; 0 faults).",
             fontsize=8.2, color=C_MUTED)
    return save_fig(fig, out_dir, "04_dpms_pauses_ab")


# ---------------------------------------------------------------------------
# Figure 5: vLLM 0.32.1 Mode/Graph/AOT Sweep & LLM Decode Weight-Fetch / Cost
# Sources: measurements/kimi-followups.md, measurements/round6.md
# ---------------------------------------------------------------------------
def fig_05_vllm_modes_and_llm_decode(out_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.subplots_adjust(wspace=0.32)

    # Panel A: vLLM 0.32.1 17-start sweep (1-stream decode, individual starts + median bar)
    groups = [
        ("default\n(n=4)",      [154.5, 154.4, 153.3, 154.2], C_BLUE),
        ("no AOT\n(n=4)",       [153.6, 149.8, 153.2, 151.8], C_TEAL),
        ("PIECEWISE\n(n=3)",    [147.2, 147.0, 153.3],        C_AMBER),
        ("eager\n(n=3)",        [147.1, 144.3, 147.1],        C_VIOLET),
        ("CG=NONE\n(n=3)",      [105.7, 105.3, 105.6],        C_RED),
    ]
    x = np.arange(len(groups))
    medians = [float(np.median(vals)) for _, vals, _ in groups]
    colors  = [c for _, _, c in groups]
    bars = ax1.bar(x, medians, width=0.52, color=colors, alpha=0.84, zorder=2)

    # Overlay every individual fresh engine start as a crisp point
    for i, (_, vals, color) in enumerate(groups):
        jit = np.linspace(-0.14, 0.14, len(vals))
        ax1.scatter(np.full(len(vals), i) + jit, vals, color=C_INK, s=26, zorder=4, edgecolors="#FFFFFF", linewidths=0.7)
        ax1.text(i, max(vals) + 4.2, f"{medians[i]:.1f}", ha="center", va="bottom",
                 fontsize=9.0, fontweight="semibold", color=C_INK)

    ax1.axhline(120.0, color="#64748B", linestyle="--", linewidth=1.2, zorder=3)
    ax1.text(2.0, 120.0, "Round 6 slow band (~120 tok/s; 0/11 starts triggered here)",
             fontsize=8.0, color=C_INK, ha="center", va="center", zorder=5,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.95))

    ax1.set_xticks(x)
    ax1.set_xticklabels([g[0] for g in groups], fontsize=8.7)
    ax1.set_ylabel("1-Stream Decode Throughput (tokens/s)")
    ax1.set_title("A. vLLM 0.32.1 FP8 Mode & AOT Sweep (1408 MHz)", loc="left", pad=10)
    ax1.set_ylim(0, 180)
    style_axes(ax1, "y")

    # Panel B: Decode 4-stream & 8-stream scaling across the same 5 vLLM modes
    m_labels = ["default", "no AOT", "PIECEWISE", "eager", "CG=NONE"]
    d4_med = [592.0, 588.1, 585.7, 534.6, 407.7]
    d8_med = [962.7, 958.4, 957.0, 918.0, 704.4]
    bx = np.arange(len(m_labels))
    w = 0.35
    b4 = ax2.bar(bx - w / 2, d4_med, width=w, label="4 streams (median tok/s)", color=C_TEAL, zorder=3)
    b8 = ax2.bar(bx + w / 2, d8_med, width=w, label="8 streams (median tok/s)", color=C_BLUE, zorder=3)
    for bar in b8:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 18, f"{h:.0f}",
                 ha="center", va="bottom", fontsize=8.4, fontweight="semibold", color=C_INK)
    for bar in b4:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 18, f"{h:.0f}",
                 ha="center", va="bottom", fontsize=8.1, color=C_MUTED)

    ax2.set_xticks(bx)
    ax2.set_xticklabels(m_labels, fontsize=8.8)
    ax2.set_ylabel("Aggregate Decode Throughput (tokens/s)")
    ax2.set_title("B. Multi-Stream Scaling Across vLLM Modes", loc="left", pad=10)
    ax2.set_ylim(0, 1260)
    style_axes(ax2, "y")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=8.3)

    fig.suptitle("RX 9070 XT (gfx1201): vLLM 0.32.1 FP8 (Llama 3.2 3B) Graph & AOT Modes",
                 fontsize=13, fontweight="bold", color=C_INK, y=1.02)
    fig.text(0.01, -0.04,
             "Source: measurements/kimi-followups.md (17 fresh starts, ROCm 10.2, overclock profile, 1408 MHz MCLK, dpms=Off, 0 faults).",
             fontsize=8.2, color=C_MUTED)
    return save_fig(fig, out_dir, "05_vllm_modes_and_llm_decode")


# ---------------------------------------------------------------------------
# Figure 6: Paused-falcond FSR 4.1.2 MCLK Scaling + HIP vs Vulkan Async Overlap
# Sources: measurements/kimi-followups.md Part B, measurements/blindspot-followups.md §2
# ---------------------------------------------------------------------------
def fig_06_fsr4_mclk_and_async_overlap(out_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7), gridspec_kw={"width_ratios": [1.08, 1.12]})
    fig.subplots_adjust(wspace=0.30)

    # Panel A: Paused-falcond ABBA FSR 4.1.2 MCLK scaling (matched ~3.44-3.47 GHz SCLK)
    mclks = ["auto\n(1408 MHz)", "L5 pinned\n(1124 MHz)", "L3 pinned\n(875 MHz)"]
    x = np.arange(len(mclks))
    fsr_4k  = [1.6670, 1.9201, 2.3003]
    gb_4k   = [1.1966, 1.4257, 1.7441]
    sh_4k   = [1.6124, 1.6616, 1.7703]
    fsr_1080 = [0.3434, 0.3763, 0.4301]

    ax1.plot(x, fsr_4k,   marker="o", linewidth=2.4, markersize=6.8, color=C_RED,   label="4K FSR 4.1.2 Upscaler (+38.0%)", zorder=4)
    ax1.plot(x, gb_4k,    marker="s", linewidth=2.1, markersize=6.0, color=C_AMBER, label="4K GBuffer control (+45.8%)", zorder=3)
    ax1.plot(x, sh_4k,    marker="^", linewidth=2.0, markersize=6.0, color=C_SLATE, linestyle="--", label="4K RasterShadow (+9.8%)", zorder=3)
    ax1.plot(x, fsr_1080, marker="D", linewidth=2.2, markersize=6.0, color=C_BLUE,  label="1080p FSR 4.1.2 (+25.2%)", zorder=3)

    # Endpoint annotations (staggered vertically, zero overlap)
    ax1.annotate("2.30 ms\n(+38%)", (2, 2.3003), xytext=(7, 2), textcoords="offset points", fontsize=8.4, fontweight="semibold", color=C_RED, va="center")
    ax1.annotate("1.77 ms\n(+10%)", (2, 1.7703), xytext=(7, 9), textcoords="offset points", fontsize=8.1, color=C_SLATE, va="center")
    ax1.annotate("1.74 ms\n(+46%)", (2, 1.7441), xytext=(7, -14), textcoords="offset points", fontsize=8.1, fontweight="semibold", color=C_AMBER, va="center")
    ax1.annotate("0.43 ms (+25%)", (2, 0.4301), xytext=(7, 0), textcoords="offset points", fontsize=8.3, fontweight="semibold", color=C_BLUE, va="center")
    ax1.annotate("1.67 ms", (0, 1.6670), xytext=(-4, 9), textcoords="offset points", fontsize=8.2, color=C_RED, ha="left")
    ax1.annotate("1.20 ms", (0, 1.1966), xytext=(-4, -14), textcoords="offset points", fontsize=8.0, color=C_AMBER, ha="left")

    ax1.set_xticks(x)
    ax1.set_xticklabels(mclks, fontsize=8.9)
    ax1.set_xlim(-0.22, 2.68)
    ax1.set_ylim(0, 2.68)
    ax1.set_ylabel("Average GPU Pass Time (ms, ABBA Mean)")
    ax1.set_title("A. Paused-falcond FSR 4.1.2 MCLK Scaling (~3.45 GHz)", loc="left", pad=10)
    style_axes(ax1, "y")
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.95)

    # Panel B: HIP FP8 WMMA beside Vulkan 1440p Graphics — Fix-Pass Rerun (HWq prio 2 vs prio 1) + First-Run (runv2)
    regimes = [
        "Alone\nc80\n(rerun)",
        "SAT mono\nc1\n(r2/rerun)",
        "PACE-60\nc80\n(rerun)",
        "SAT c80\n(rerun\nABBA)",
        "SAT c80\n(first run\nprio ign.*)",
    ]
    # Exact verbatim values from measurements/blindspot-followups.md §2:
    # prio 1 (default ROCm DEBUG_HIP_IGNORE_STREAM_PRIORITY=1):
    #   Alone c80 (r3 A1) 6.92, SAT mono (r2 norm) 11.93, PACE-60 (r3 A1) 12.61,
    #   SAT c80 (r3 A1 starved, A2=76.4) 85.06, SAT c80 (r2 norm, 10.8ms alone) 147.9
    # prio 2 (r3 DEBUG_HIP_IGNORE_STREAM_PRIORITY=0, HWq priority 2) except last bar = r2 high* (still prio 1):
    #   Alone c80 (r3 B1) 6.98, SAT mono (r3 true_p2) 10.30, PACE-60 (r3 B1) 12.16,
    #   SAT c80 (r3 B1 3/3 full, B2=14.0-48.3) 21.49, SAT c80 (r2 high*, ignored) 150.2
    p1_ms = [6.92, 11.93, 12.61, 85.06, 147.9]
    p2_ms = [6.98, 10.30, 12.16, 21.49, 150.2]
    bx = np.arange(len(regimes))
    w = 0.35
    b1 = ax2.bar(bx - w / 2, p1_ms, width=w, label="HWq prio 1 (default IGNORE_PRIO=1)", color=C_BLUE, zorder=3)
    b2 = ax2.bar(bx[:4] + w / 2, p2_ms[:4], width=w, label="HWq prio 2 (IGNORE_PRIO=0, rerun)", color=C_TEAL, zorder=3)
    b2_r2 = ax2.bar(bx[4] + w / 2, p2_ms[4], width=w, label="First run high* (prio ignored = p1)",
                    color="#94A3B8", hatch="//", edgecolor=C_SLATE, linewidth=0.7, zorder=3)

    # Bar value labels
    ax2.text(bx[0], 6.98 * 1.12, "6.9 / 7.0", ha="center", va="bottom", fontsize=7.9, fontweight="semibold", color=C_INK)
    ax2.text(bx[1] - w / 2, 11.93 * 1.11, "11.9", ha="center", va="bottom", fontsize=7.8, fontweight="semibold", color=C_BLUE)
    ax2.text(bx[1] + w / 2, 10.30 * 1.11, "10.3", ha="center", va="bottom", fontsize=7.8, fontweight="semibold", color=C_TEAL)
    ax2.text(bx[2], 12.61 * 1.12, "12.6 / 12.2", ha="center", va="bottom", fontsize=7.9, fontweight="semibold", color=C_INK)
    ax2.text(bx[3] - w / 2, 85.06 * 1.08, "85.1\n(A2:76.4)", ha="center", va="bottom", fontsize=7.3, fontweight="semibold", color=C_RED)
    ax2.text(bx[3] + w / 2 + 0.03, 21.49 * 1.10, "21.5\n(3/3)", ha="center", va="bottom", fontsize=7.4, fontweight="semibold", color=C_TEAL)
    ax2.text(bx[4], 150.2 * 1.09, "148 / 150\n(both p1*)", ha="center", va="bottom", fontsize=7.5, fontweight="semibold", color=C_SLATE)

    # Graphics cadence tradeoff summary box in clean upper-left whitespace (zero arrow crossing)
    ax2.text(-0.36, 98,
             "SAT GFX cadence (vs 3.48 ms):\n"
             "  base_p1_c80: 3.61 ms (+4%)\n"
             "  true_p2_c80: 4.95 ms (+42%)\n"
             "  true_p2_mono: 7.51 ms (2.16x)",
             ha="left", va="top", fontsize=7.3, color=C_INK,
             bbox=dict(boxstyle="round,pad=0.34", facecolor=C_PANEL, edgecolor="#CBD5E1"))

    ax2.set_yscale("log")
    ax2.set_yticks([5, 10, 20, 50, 100, 200])
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xticks(bx)
    ax2.set_xticklabels(regimes, fontsize=8.0)
    ax2.set_ylabel("HIP FP8 WMMA Job Time (ms, log scale)")
    ax2.set_title("B. Async Overlap: HWq Priority 2 vs Priority 1 & Fusion", loc="left", pad=10)
    ax2.set_ylim(5, 420)
    style_axes(ax2, "y")
    ax2.legend(loc="upper left", fontsize=7.4, framealpha=0.95)

    fig.suptitle("RX 9070 XT (gfx1201): FSR 4.1.2 Memory-Clock Scaling & Async-Compute Scheduling",
                 fontsize=13, fontweight="bold", color=C_INK, y=1.02)
    fig.text(0.01, -0.05,
             "Sources: measurements/kimi-followups.md Part B (12 ABBA cells, flat pinned MCLK) and blindspot-followups.md §2 (first run runv2 + 2026-11-27 rerun runv3-sched, unpinned SCLK, 92 C thermal stop, 0 faults; B2 repeat 14.0–48.3 ms).",
             fontsize=7.9, color=C_MUTED)
    return save_fig(fig, out_dir, "06_fsr4_mclk_and_async_overlap")


# ---------------------------------------------------------------------------
# Figure 7: Blind Spot #3 — Store-Hint MALL Allocation/Eviction & Cross-Wave 32 B Write Combining
# Source: measurements/blindspot-followups.md §3 (store_mall)
# ---------------------------------------------------------------------------
def fig_07_store_hints_and_write_combine(out_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.subplots_adjust(wspace=0.33)

    # Panel A: All 9 valid gfx1201 store hints at 32 MiB (consumer read latency vs victim eviction)
    hints = ["rt", "ht", "nt_rt", "nt_ht", "rt_sys", "wb*", "nt", "rt_nt", "nt_sys"]
    alloc_cyc = [493.5, 505.8, 493.1, 496.5, 505.4, 516.2, 799.8, 796.5, 807.0]
    evict_cyc = [804.3, 804.1, 796.6, 813.6, 803.7, 791.6, 505.9, 503.1, 499.1]
    x = np.arange(len(hints))
    w = 0.36
    ax1.bar(x - w / 2, alloc_cyc, width=w, label="Consumer read (low = MALL hit)", color=C_BLUE, zorder=3)
    ax1.bar(x + w / 2, evict_cyc, width=w, label="Resident victim (high = evicted)", color=C_AMBER, zorder=3)

    ax1.axhline(489.5, color=C_TEAL, linestyle="--", linewidth=1.2, label="Warm ref (490 cyc)", zorder=2)
    ax1.axhline(785.1, color=C_RED,  linestyle=":",  linewidth=1.2, label="Cold ref (785 cyc)", zorder=2)
    ax1.axvline(5.5, color="#94A3B8", linestyle="-.", linewidth=1.1)

    ax1.text(2.5, 950, "Allocates & Evicts MALL\n(1,151–1,197 GB/s; *wb = 553 GB/s)",
             ha="center", va="top", fontsize=7.9, fontweight="semibold", color=C_BLUE,
             bbox=dict(boxstyle="round,pad=0.28", facecolor=C_PANEL, edgecolor="#CBD5E1"))
    ax1.text(7.0, 950, "Bypasses MALL\n(598–602 GB/s;\npreserves victim)",
             ha="center", va="top", fontsize=7.9, fontweight="semibold", color=C_RED,
             bbox=dict(boxstyle="round,pad=0.28", facecolor="#FEF2F2", edgecolor="#FECACA"))

    ax1.set_xticks(x)
    ax1.set_xticklabels(hints, fontsize=8.6)
    ax1.set_ylabel("Pointer-Chase Latency (Cycles, 32 MiB)")
    ax1.set_title("A. Store-Hint MALL Allocation vs Victim Eviction (32 MiB)", loc="left", pad=10)
    ax1.set_ylim(0, 1240)
    style_axes(ax1, "y")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, fontsize=7.3, columnspacing=0.9, framealpha=0.96)

    # Panel B: Cross-wave 32 B write combining (512 MiB streaming stores)
    pats = [
        "full_b128\n(128 B/wave)",
        "same_wave_4x32B\n(4 quarters, 1 wave)",
        "full_b32\n(32-bit full line)",
        "xwave_diff_wg_32B\n(4 WGs x 32 B)",
        "xwave_same_wg_32B\n(4 waves/WG x 32 B)",
        "quarter_32B_only\n(32 B / 128 B line)",
        "eighth_16B_only\n(16 B / 128 B line)",
    ]
    bws = [672.9, 667.3, 624.0, 545.3, 515.7, 294.6, 154.8]
    pcols = [C_TEAL, C_TEAL, C_BLUE, C_AMBER, C_RED, C_VIOLET, C_SLATE]
    y = np.arange(len(pats))[::-1]
    bars = ax2.barh(y, bws, height=0.60, color=pcols, zorder=3)
    for bar, v, idx in zip(bars, bws, range(len(bws))):
        note = ""
        if idx == 1: note = " (-0.8%)"
        elif idx == 3: note = " (-19.0%)"
        elif idx == 4: note = " (-23.4%)"
        elif idx == 5: note = " (9.2 G lines/s)"
        elif idx == 6: note = " (9.7 G lines/s)"
        txt = f"{v:.1f} GB/s{note}"
        if v > 460:
            ax2.text(v - 12, bar.get_y() + bar.get_height() / 2, txt,
                     va="center", ha="right", fontsize=8.2, fontweight="semibold", color="#FFFFFF")
        else:
            ax2.text(v + 12, bar.get_y() + bar.get_height() / 2, txt,
                     va="center", ha="left", fontsize=8.2, fontweight="semibold", color=C_INK)

    ax2.set_yticks(y)
    ax2.set_yticklabels(pats, fontsize=8.3)
    ax2.set_xlim(0, 745)
    ax2.set_xlabel("Logical Store Bandwidth (GB/s)")
    ax2.set_title("B. Cross-Wave 32 B Write Combining & Partial-Line Rate", loc="left", pad=10)
    style_axes(ax2, "x")

    fig.suptitle("RX 9070 XT (gfx1201): Store-Hint MALL Residency & Cross-Wave 32 B Write Combining",
                 fontsize=13, fontweight="bold", color=C_INK, y=1.02)
    fig.text(0.01, -0.04,
             "Source: measurements/blindspot-followups.md §3 (store_mall, first run, unpinned clocks; 9 tested gfx1201 store hints verified in asm; 0 faults; --break exits FAIL).",
             fontsize=8.0, color=C_MUTED)
    return save_fig(fig, out_dir, "07_store_hints_and_write_combine")


# ---------------------------------------------------------------------------
# Figure 8: Blind Spot #4 (Load Width & Clustering) + Blind Spot #10 (RADV w32 vs w64)
# Sources: measurements/blindspot-followups.md §4 & §5 (2026-11-27 fix-pass reruns)
# ---------------------------------------------------------------------------
def fig_08_load_width_clustering_and_radv_waves(out_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9), gridspec_kw={"width_ratios": [1.15, 1.05]})
    fig.subplots_adjust(wspace=0.28)

    # Panel A: HIP load_cluster (2026-11-27 rerun round12-run3-cluster, matched 8w/SIMD, pin_accs4 true clumping)
    pats = [
        "ref\n(0 B)",
        "b128\nclump\n(16 B)",
        "b128\nspread\n(16 B)",
        "b64*\nclump\n(8 B)",
        "2x b64\nclump\n(16 B)",
        "4x b32\nspread\n(16 B)",
        "4x b32\nK=4\n(4 B)",
    ]
    w32_l0 = [8.94, 11.33, 12.18, 12.18, 16.02, 30.02, 12.08]
    w64_l0 = [8.57, 16.05, 16.01, 16.01, 32.01, 60.83, 15.01]
    x = np.arange(len(pats))
    w = 0.36
    b1 = ax1.bar(x - w / 2, w32_l0, width=w, label="HIP wave32 card_8w (8 w/SIMD)", color=C_BLUE, zorder=3)
    b2 = ax1.bar(x + w / 2, w64_l0, width=w, label="HIP wave64 card_8w (matched 8 w/SIMD)", color=C_AMBER, zorder=3)

    for idx, bar in enumerate(b1):
        h = bar.get_height()
        dx = -0.04 if idx == 5 else 0.0
        ax1.text(bar.get_x() + bar.get_width() / 2 + dx, h + 1.1, f"{h:.1f}",
                 ha="center", va="bottom", fontsize=7.4, fontweight="semibold", color=C_BLUE)
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.1, f"{h:.1f}",
                 ha="center", va="bottom", fontsize=7.4, fontweight="semibold", color=C_AMBER)

    ax1.annotate("Per-load floor persists at 8w/SIMD (mechanism inferred):\n~8 cyc/load (w32), ~16 cyc/load (w64)\n*w64 b64 = 512 B/WMMA (same as w32 b128)\n1-wave L0 clump (pin_accs4): -37.4%",
                 xy=(5.18, 60.8), xytext=(0.15, 42.5),
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.1),
                 fontsize=7.6, fontweight="semibold", color=C_INK,
                 bbox=dict(boxstyle="round,pad=0.32", facecolor=C_PANEL, edgecolor="#CBD5E1"))

    ax1.set_xticks(x)
    ax1.set_xticklabels(pats, fontsize=8.0)
    ax1.set_ylabel("Cycles / FP8 WMMA (Whole Card, L0 8 KiB)")
    ax1.set_title("A. Load Width & True Clumping (Rerun, Matched 8w/SIMD)", loc="left", pad=10)
    ax1.set_ylim(0, 75)
    style_axes(ax1, "y")
    ax1.legend(loc="upper left", fontsize=7.8)

    # Panel B: Vulkan RADV vk_ld_mat2 (2026-11-27 rerun round12-run3-vk, warmed 2.99-3.24 GHz, 3-way decomposition)
    vk_cats = ["ref\n(reg)", "K8_L0\n(8 KiB)", "K1_L0\n(8 KiB)", "K1_L2\n(1 MiB)", "K8_L2\n(1 MiB)"]
    sg32_8w = [8.33, 13.21, 15.77, 27.07, 40.21]
    sg64_8w = [8.38, 14.34, 20.44, 34.41, 40.79]
    sg64_4w = [10.81, 16.38, 24.30, 53.31, 80.38]
    vx = np.arange(len(vk_cats))
    vw = 0.26
    ax2.bar(vx - vw, sg32_8w, width=vw, label="RADV sg=32 card_8w (8 w/SIMD)", color=C_TEAL, zorder=3)
    ax2.bar(vx,      sg64_8w, width=vw, label="RADV sg=64 card_8w (matched 8 w/SIMD)", color=C_AMBER, zorder=3)
    ax2.bar(vx + vw, sg64_4w, width=vw, label="RADV sg=64 card_4w (eq threads, 4 w/SIMD)", color=C_RED, zorder=3)

    for i in range(len(vk_cats)):
        dx32 = -0.04 if i == 4 else 0.0
        dy64_8 = 2.6 if i == 4 else 1.3
        ax2.text(vx[i] - vw + dx32, sg32_8w[i] + 1.3, f"{sg32_8w[i]:.1f}", ha="center", va="bottom",
                 fontsize=7.1, fontweight="semibold", color=C_TEAL)
        ax2.text(vx[i], sg64_8w[i] + dy64_8, f"{sg64_8w[i]:.1f}", ha="center", va="bottom",
                 fontsize=7.1, fontweight="semibold", color=C_AMBER)
        r4 = sg64_4w[i] / sg32_8w[i]
        lbl4 = f"{sg64_4w[i]:.1f}\n({r4:.2f}x)" if i >= 3 else f"{sg64_4w[i]:.1f}"
        ax2.text(vx[i] + vw, sg64_4w[i] + 1.3, lbl4, ha="center", va="bottom",
                 fontsize=7.1, fontweight="semibold", color=C_RED)

    ax2.text(-0.36, 71.5,
             "sg=64 / sg=32 ratio (8w occ vs 4w):\n"
             "  K8_L0: 1.09x (8w) vs 1.24x (4w)\n"
             "  K1_L0: 1.30x (8w) vs 1.54x (4w)\n"
             "  K1_L2: 1.27x (8w) vs 1.97x (4w)\n"
             "  K8_L2: 1.01x (8w) vs 2.00x (4w)",
             ha="left", va="top", fontsize=7.2, color=C_INK,
             bbox=dict(boxstyle="round,pad=0.34", facecolor=C_PANEL, edgecolor="#CBD5E1"))

    ax2.set_xticks(vx)
    ax2.set_xticklabels(vk_cats, fontsize=8.3)
    ax2.set_ylabel("Whole-Card Dispatch Cycles / FP8 WMMA")
    ax2.set_title("B. RADV Wave32 vs Wave64 (Rerun, Boosted ~3.1 GHz)", loc="left", pad=10)
    ax2.set_ylim(0, 102)
    style_axes(ax2, "y")
    ax2.legend(loc="upper left", fontsize=7.4, framealpha=0.95)

    fig.suptitle("RX 9070 XT (gfx1201): Load Width/Clustering Law & RADV Wave32 vs Wave64 Decomposition",
                 fontsize=13, fontweight="bold", color=C_INK, y=1.02)
    fig.text(0.01, -0.05,
             "Sources: measurements/blindspot-followups.md §4 (2026-11-27 rerun round12-run3-cluster, unpinned 2.69–3.23 GHz, 22/22 FIFO+clump clean, PASS) and §5 (round12-run3-vk, warmed 2.99–3.24 GHz, unpinned, PASS; first-run 1.87 GHz card_4w = 1.35–2.07x).",
             fontsize=7.9, color=C_MUTED)
    return save_fig(fig, out_dir, "08_load_width_clustering_and_radv_waves")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build publication figures for rdna4-fp8.")
    ap.add_argument("--out-dir", default="figures", help="Output directory (default: figures)")
    ap.add_argument("--self-test", action="store_true", help="Write to temp dir and verify all PNG/SVG outputs")
    args = ap.parse_args()

    configure_style()
    out_dir = Path(args.out_dir)
    builders = (
        fig_01_memory_hierarchy_and_mclk,
        fig_02_wmma_load_overlap,
        fig_03_matrix_rates_and_hazards,
        fig_04_dpms_pauses_ab,
        fig_05_vllm_modes_and_llm_decode,
        fig_06_fsr4_mclk_and_async_overlap,
        fig_07_store_hints_and_write_combine,
        fig_08_load_width_clustering_and_radv_waves,
    )
    if args.self_test:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="r4fig-") as td:
            p = Path(td)
            files = []
            for fn in builders:
                files.extend(fn(p))
            assert len(files) == 16, f"expected 16 outputs, got {len(files)}"
            for f in files:
                assert f.exists() and f.stat().st_size > 2000, f"empty output: {f}"
            print(f"VERDICT make_figures selftest PASS ({len(files)} files verified)")
            return 0

    all_files = []
    for fn in builders:
        all_files.extend(fn(out_dir))
    for f in all_files:
        print(f"WROTE {f} ({f.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
