#!/usr/bin/env python3
"""
Amber24 Benchmark Plot Generator
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = Path(".")    
OUT_DIR  = Path(".")     

# Colour palette
DARK   = "#0f1117"
PANEL  = "#1a1d27"
BLUE   = "#00aaff"   # Saga CPU
GREEN  = "#76b900"   # Saga GPU (NVIDIA green)
AMBER  = "#ffaa00"   # Olivia CPU
PURPLE = "#9b59b6"   # Olivia GPU (Grace Hopper)
WHITE  = "#e8eaf0"
GRID   = "#2a2d3a"

# Labels
LABEL_SAGA_CPU    = "Saga - Intel Xeon CPU"
LABEL_SAGA_GPU    = "Saga - NVIDIA A100 GPU"
LABEL_OLIVIA_CPU  = "Olivia - AMD EPYC Turin CPU"
LABEL_OLIVIA_GPU  = "Olivia - NVIDIA Grace Hopper GPU"

COLORS = {
    LABEL_SAGA_CPU:   BLUE,
    LABEL_SAGA_GPU:   GREEN,
    LABEL_OLIVIA_CPU: AMBER,
    LABEL_OLIVIA_GPU: PURPLE,
}

# Preferred bar order
PREFERRED_ORDER = [LABEL_SAGA_CPU, LABEL_OLIVIA_CPU, LABEL_SAGA_GPU, LABEL_OLIVIA_GPU]

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_out(path: Path) -> dict:
    data = {k: [] for k in ("nstep", "time_ps", "temp", "press",
                             "etot", "ektot", "eptot", "density", "volume")}
    block      = {}
    in_block   = False
    skip_block = False

    with open(path) as fh:
        for line in fh:
            if re.search(r'A V E R A G E S|R M S  F L U C T U A T I O N S', line):
                skip_block = True

            m = re.match(
                r'\s+NSTEP\s+=\s+(\d+)\s+TIME\(PS\)\s+=\s+([\d.]+)'
                r'\s+TEMP\(K\)\s+=\s+([\d.]+)\s+PRESS\s+=\s+([\d.-]+)', line)
            if m:
                if not skip_block:
                    block = {"nstep": int(m[1]), "time_ps": float(m[2]),
                             "temp":  float(m[3]), "press":  float(m[4])}
                    in_block = True
                continue

            if in_block and not skip_block:
                m = re.match(
                    r'\s+Etot\s+=\s+([\d.-]+)\s+EKtot\s+=\s+([\d.-]+)'
                    r'\s+EPtot\s+=\s+([\d.-]+)', line)
                if m:
                    block.update(etot=float(m[1]), ektot=float(m[2]),
                                 eptot=float(m[3]))

                m = re.search(r'VOLUME\s+=\s+([\d.]+)', line)
                if m:
                    block["volume"] = float(m[1])
                m = re.search(r'Density\s+=\s+([\d.]+)', line)
                if m:
                    block["density"] = float(m[1])

                if re.match(r'\s*-{20,}', line):
                    if "etot" in block:
                        for k in data:
                            if k in block:
                                data[k].append(block[k])
                    block      = {}
                    in_block   = False
                    skip_block = False

    return {k: np.array(v) for k, v in data.items()}


def parse_performance(path: Path):
    ns_day = None
    pattern = re.compile(r'ns/day\s+=\s+([\d.]+)')
    with open(path) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                ns_day = float(m[1])
    return ns_day


# ── Plotting helpers ──────────────────────────────────────────────────────────

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.6, linestyle="--", alpha=0.7)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)


# ── Figure 1 : Energy & Temperature ───────────────────────────────────────────

def fig_energy_temp(system_groups: dict):
    sizes = list(system_groups.keys())
    nrows = len(sizes)

    fig = plt.figure(figsize=(18, 7.5), facecolor=DARK)
    fig.suptitle("Amber24 Benchmark — Energy & Temperature", color=WHITE,
                 fontsize=16, fontweight="bold", y=0.98)

    gs = GridSpec(nrows, 4, figure=fig, hspace=0.75, wspace=0.38,
                  left=0.15, right=0.98, top=0.91, bottom=0.07)

    col_titles = ["Temperature (K)", "Total Energy (kcal/mol)",
                  "Potential Energy (kcal/mol)", "Density (g/cm³)"]
    col_keys   = ["temp", "etot", "eptot", "density"]
    col_ylabs  = ["T (K)", "E$_{tot}$ (kcal/mol)",
                  "E$_{pot}$ (kcal/mol)", "ρ (g/cm³)"]

    all_handles = []
    all_labels = []
    label_seen = set()

    for row, size in enumerate(sizes):
        runs = system_groups[size]

        for col in range(4):
            ax = fig.add_subplot(gs[row, col])
            style_ax(ax, col_titles[col], "Simulation time (ps)", col_ylabs[col])

            if col == 0:
                ax.text(-0.30, 0.5, size, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=13, fontweight="bold", color=WHITE)

            for run_label, info in runs.items():
                d = info["data"]
                c = info["color"]
                key = col_keys[col]
                if len(d["time_ps"]) == 0 or key not in d or len(d[key]) < 2:
                    continue
                t = d["time_ps"] - d["time_ps"][0]
                y = d[key]

                window = max(1, min(15, len(y) // 5))
                if window % 2 == 0:
                    window += 1
                pad = window // 2
                y_padded = np.pad(y, pad, mode='edge')
                y_smooth = np.convolve(y_padded, np.ones(window)/window, mode='valid')

                line, = ax.plot(t, y_smooth, color=c, linewidth=3.5,
                                label=run_label)

                if run_label not in label_seen:
                    all_handles.append(line)
                    all_labels.append(run_label)
                    label_seen.add(run_label)

            if col == 0:
                ax.axhline(300, color=WHITE, linewidth=0.9, linestyle=":", alpha=0.6)

    # Legend
    fig.legend(all_handles, all_labels, loc='center',
               bbox_to_anchor=(0.5, 0.48), ncol=len(all_labels),
               fontsize=10, facecolor=PANEL, edgecolor=GRID,
               labelcolor=WHITE, framealpha=0.9)

    fig.subplots_adjust(left=0.15, right=0.98, top=0.91, bottom=0.07, hspace=0.75, wspace=0.38)

    fig.savefig(OUT_DIR / "amber_energy_temp.png", dpi=200,
                bbox_inches="tight", facecolor=DARK)
    print("Saved: amber_energy_temp.png")
    plt.close(fig)


# ── Figure 2 : Performance (inchangé) ─────────────────────────────────────────

def fig_performance(perf: dict):
    systems = list(perf.keys())

    all_configs = set()
    for system_perf in perf.values():
        for config, value in system_perf.items():
            if value is not None:
                all_configs.add(config)

    present_configs = [c for c in PREFERRED_ORDER if c in all_configs]
    present_colors = [COLORS[c] for c in present_configs]

    if not present_configs:
        print("No performance data to plot.")
        return

    fig, (ax_abs, ax_speedup) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor=DARK)
    fig.suptitle("Amber24 Benchmark — Performance on Saga & Olivia HPC",
                 color=WHITE, fontsize=16, fontweight="bold", y=0.98)

    fig.text(0.5, 0.88, "CPU runs: 32 MPI processes | GPU runs: single GPU",
             ha="center", va="center", color=WHITE, fontsize=11, alpha=0.9)

    # Throughput
    style_ax(ax_abs, "Throughput", "System size", "ns / day")

    num_configs = len(present_configs)
    width = 0.8 / num_configs
    x = np.arange(len(systems))

    all_vals = [v for system_perf in perf.values() for v in system_perf.values() if v is not None]
    max_val = max(all_vals) if all_vals else 1
    label_offset = max_val * 0.02

    for i, conf in enumerate(present_configs):
        vals = [perf[s].get(conf) or 0 for s in systems]
        offset = width * (i - (num_configs - 1)/2)
        bars = ax_abs.bar(x + offset, vals, width, color=present_colors[i], zorder=3)

        for bar, val in zip(bars, vals):
            if val > 0:
                ax_abs.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_offset,
                            f"{val:.1f}", ha="center", va="bottom",
                            color=WHITE, fontsize=9, fontweight="bold")

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(systems, color=WHITE)
    ax_abs.set_ylim(0, max_val * 1.18)

    # Speedup
    style_ax(ax_speedup, "Speedup relative to Saga Intel Xeon CPU", "System size", "× faster")

    speedup_configs = [c for c in present_configs if c != LABEL_SAGA_CPU]
    speedup_colors = [COLORS[c] for c in speedup_configs]

    if speedup_configs:
        width_sp = 0.8 / len(speedup_configs)
        x_sp = np.arange(len(systems))

        max_sp = 1.0
        for i, conf in enumerate(speedup_configs):
            ratios = []
            display_ratios = []
            for s in systems:
                saga_cpu_val = perf[s].get(LABEL_SAGA_CPU)
                conf_val = perf[s].get(conf)
                if saga_cpu_val and conf_val and saga_cpu_val > 0:
                    ratio = conf_val / saga_cpu_val
                    ratios.append(ratio)
                    display_ratios.append(ratio)
                else:
                    ratios.append(None)
                    display_ratios.append(0)

            if not any(r is not None for r in ratios):
                continue

            offset = width_sp * (i - (len(speedup_configs) - 1)/2)
            bars = ax_speedup.bar(x_sp + offset, display_ratios, width_sp,
                                  color=speedup_colors[i], zorder=3)

            for bar, r in zip(bars, ratios):
                if r is not None:
                    ax_speedup.text(bar.get_x() + bar.get_width()/2,
                                    bar.get_height() + max_sp*0.03,
                                    f"{r:.1f}×", ha="center", va="bottom",
                                    color=WHITE, fontsize=11, fontweight="bold")
            current_max = max(r for r in ratios if r is not None)
            max_sp = max(max_sp, current_max)

        ax_speedup.set_xticks(x_sp)
        ax_speedup.set_xticklabels(systems, color=WHITE)
        ax_speedup.set_ylim(0, max_sp * 1.25)
        ax_speedup.axhline(1, color=WHITE, linewidth=0.9, linestyle=":", alpha=0.6)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=COLORS[c], label=c) for c in present_configs]
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.82), ncol=len(present_configs),
               fontsize=10, facecolor=PANEL, edgecolor=GRID,
               labelcolor=WHITE, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.80])
    fig.savefig(OUT_DIR / "amber_performance.png", dpi=200, bbox_inches="tight",
                facecolor=DARK)
    print("Saved: amber_performance.png")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # File paths
    p20_saga_cpu_out = BASE_DIR / "20k-atoms" / "benchmark_cpu.out"
    p61_saga_cpu_out = BASE_DIR / "61k-atoms" / "benchmark_cpu.out"
    p20_saga_cpu_inf = BASE_DIR / "20k-atoms" / "benchmark_cpu_inf"
    p61_saga_cpu_inf = BASE_DIR / "61k-atoms" / "benchmark_cpu_inf"

    p20_saga_gpu_out = BASE_DIR / "20k-atoms" / "benchmark_gpu.out"
    p61_saga_gpu_out = BASE_DIR / "61k-atoms" / "benchmark_gpu.out"
    p20_saga_gpu_inf = BASE_DIR / "20k-atoms" / "benchmark_gpu_inf"
    p61_saga_gpu_inf = BASE_DIR / "61k-atoms" / "benchmark_gpu_inf"

    p20_olivia_cpu_out = BASE_DIR / "20k-atoms" / "benchmark_olivia_cpu.out"
    p61_olivia_cpu_out = BASE_DIR / "61k-atoms" / "benchmark_olivia_cpu.out"
    p20_olivia_cpu_inf = BASE_DIR / "20k-atoms" / "benchmark_olivia_cpu_inf"
    p61_olivia_cpu_inf = BASE_DIR / "61k-atoms" / "benchmark_olivia_cpu_inf"

    p20_olivia_gpu_out = BASE_DIR / "20k-atoms" / "benchmark_olivia_gpu.out"
    p61_olivia_gpu_out = BASE_DIR / "61k-atoms" / "benchmark_olivia_gpu.out"
    p20_olivia_gpu_inf = BASE_DIR / "20k-atoms" / "benchmark_olivia_gpu_inf"
    p61_olivia_gpu_inf = BASE_DIR / "61k-atoms" / "benchmark_olivia_gpu_inf"

    # Parse energy/temp data
    system_groups = {}

    for size in ["20k atoms", "61k atoms"]:
        grp = {}

        saga_cpu_out = p20_saga_cpu_out if size == "20k atoms" else p61_saga_cpu_out
        if saga_cpu_out.exists():
            grp[LABEL_SAGA_CPU] = {"data": parse_out(saga_cpu_out), "color": COLORS[LABEL_SAGA_CPU]}

        saga_gpu_out = p20_saga_gpu_out if size == "20k atoms" else p61_saga_gpu_out
        if saga_gpu_out.exists():
            grp[LABEL_SAGA_GPU] = {"data": parse_out(saga_gpu_out), "color": COLORS[LABEL_SAGA_GPU]}

        olivia_cpu_out = p20_olivia_cpu_out if size == "20k atoms" else p61_olivia_cpu_out
        if olivia_cpu_out.exists():
            grp[LABEL_OLIVIA_CPU] = {"data": parse_out(olivia_cpu_out), "color": COLORS[LABEL_OLIVIA_CPU]}

        olivia_gpu_out = p20_olivia_gpu_out if size == "20k atoms" else p61_olivia_gpu_out
        if olivia_gpu_out.exists():
            grp[LABEL_OLIVIA_GPU] = {"data": parse_out(olivia_gpu_out), "color": COLORS[LABEL_OLIVIA_GPU]}

        if grp:
            system_groups[size] = grp

    if system_groups:
        fig_energy_temp(system_groups)
    else:
        print("No .out files found — skipping energy/temp plots.")

    # Parse performance
    def get_ns(inf_path, out_path):
        if inf_path.exists():
            return parse_performance(inf_path)
        if out_path.exists():
            return parse_performance(out_path)
        return None

    perf = {
        "20k atoms": {
            LABEL_SAGA_CPU:    get_ns(p20_saga_cpu_inf, p20_saga_cpu_out),
            LABEL_SAGA_GPU:    get_ns(p20_saga_gpu_inf, p20_saga_gpu_out),
            LABEL_OLIVIA_CPU:  get_ns(p20_olivia_cpu_inf, p20_olivia_cpu_out),
            LABEL_OLIVIA_GPU:  get_ns(p20_olivia_gpu_inf, p20_olivia_gpu_out),
        },
        "61k atoms": {
            LABEL_SAGA_CPU:    get_ns(p61_saga_cpu_inf, p61_saga_cpu_out),
            LABEL_SAGA_GPU:    get_ns(p61_saga_gpu_inf, p61_saga_gpu_out),
            LABEL_OLIVIA_CPU:  get_ns(p61_olivia_cpu_inf, p61_olivia_cpu_out),
            LABEL_OLIVIA_GPU:  get_ns(p61_olivia_gpu_inf, p61_olivia_gpu_out),
        },
    }

    for d in perf.values():
        to_remove = [k for k, v in d.items() if v is None]
        for k in to_remove:
            del d[k]

    if any(perf[s] for s in perf):
        fig_performance(perf)
    else:
        print("No performance data found — skipping performance plot.")

    print("Done.")


if __name__ == "__main__":
    main()
