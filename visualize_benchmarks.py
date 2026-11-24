
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "benchmark_results.csv"
OUTPUT_DIR = "benchmark_charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "sequential": "#64748b",
    "hybrid": "#2563eb",
    "speedup": "#0891b2",
    "efficiency": "#7c3aed",
    "accent": "#f59e0b",
    "ideal": "#94a3b8",
    "background": "#f8fafc",
    "grid": "#e2e8f0",
    "text": "#1e293b"
}

# ============================================================
# GLOBAL CHART STYLE
# ============================================================

plt.rcParams.update({
    "figure.figsize": (11, 6),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "text.color": COLORS["text"],
    "axes.facecolor": "white",
    "figure.facecolor": COLORS["background"],
    "axes.edgecolor": COLORS["grid"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.linestyle": "--",
    "grid.alpha": 0.65,
    "axes.axisbelow": True
})


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

print("\nLoading benchmark results...")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "Implementation",
    "MPI_Processes",
    "OpenMP_Threads",
    "Workers",
    "Run",
    "Iterations",
    "Execution_Time_Seconds"
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

numeric_columns = [
    "MPI_Processes",
    "OpenMP_Threads",
    "Workers",
    "Run",
    "Iterations",
    "Execution_Time_Seconds"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=required_columns)

if df.empty:
    raise ValueError("No valid benchmark records found.")

if (df["Execution_Time_Seconds"] <= 0).any():
    raise ValueError("Execution times must be positive.")

df["Implementation"] = (
    df["Implementation"].astype(str).str.strip()
)

sequential = df[
    df["Implementation"].str.lower() == "sequential"
]

hybrid = df[
    df["Implementation"].str.lower() == "hybrid"
]

if sequential.empty or hybrid.empty:
    raise ValueError(
        "CSV must contain both Sequential and Hybrid results."
    )


# ============================================================
# CALCULATE STATISTICS
# ============================================================

# Raw times are stored in seconds in the original CSV.
# Convert to milliseconds for ALL analysis and plotting.

sequential["Execution_Time_ms"] = (
    sequential["Execution_Time_Seconds"] * 1000
)

hybrid["Execution_Time_ms"] = (
    hybrid["Execution_Time_Seconds"] * 1000
)

seq_mean = sequential["Execution_Time_ms"].mean()
seq_std = sequential["Execution_Time_ms"].std(ddof=1)

if not np.isfinite(seq_std):
    seq_std = 0.0

summary = (
    hybrid.groupby(
        ["MPI_Processes", "OpenMP_Threads", "Workers"]
    )
    .agg(
        Mean_Time_ms=("Execution_Time_ms", "mean"),
        Std_Time_ms=("Execution_Time_ms", "std"),
        Min_Time_ms=("Execution_Time_ms", "min"),
        Max_Time_ms=("Execution_Time_ms", "max"),
        Mean_Iterations=("Iterations", "mean"),
        Std_Iterations=("Iterations", "std"),
        Runs=("Run", "count")
    )
    .reset_index()
    .sort_values(["Workers", "MPI_Processes", "OpenMP_Threads"])
)

summary["Std_Time_ms"] = summary["Std_Time_ms"].fillna(0)
summary["Std_Iterations"] = summary["Std_Iterations"].fillna(0)

# Speedup is unit-independent
summary["Speedup"] = seq_mean / summary["Mean_Time_ms"]

# Parallel efficiency
summary["Efficiency_Percent"] = (
    summary["Speedup"] / summary["Workers"] * 100
)

summary["Configuration"] = (
    summary["MPI_Processes"].astype(int).astype(str)
    + " MPI × "
    + summary["OpenMP_Threads"].astype(int).astype(str)
    + " OMP"
)

# Save summary statistics
summary.to_csv(
    os.path.join(OUTPUT_DIR, "benchmark_summary.csv"),
    index=False
)


# ============================================================
# PRINT STATISTICS
# ============================================================

print("\n========== BENCHMARK STATISTICS ==========")

print(f"Sequential average: {seq_mean:.3f} ms")
print(f"Sequential std dev: {seq_std:.3f} ms")

print("\nHybrid configurations:")

print(
    summary[
        [
            "Configuration",
            "Workers",
            "Mean_Time_ms",
            "Std_Time_ms",
            "Speedup",
            "Efficiency_Percent"
        ]
    ].round(3).to_string(index=False)
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_chart(fig, filename):

    path = os.path.join(OUTPUT_DIR, filename)

    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print(f"Generated: {path}")


def add_labels(ax, bars, suffix="", decimals=2):

    for bar in bars:

        height = bar.get_height()

        ax.annotate(
            f"{height:.{decimals}f}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )


# ============================================================
# 1. EXECUTION TIME COMPARISON
# ============================================================

print("\nGenerating execution time chart...")

labels = ["Sequential"] + summary["Configuration"].tolist()

times = [seq_mean] + summary["Mean_Time_ms"].tolist()

errors = [seq_std] + summary["Std_Time_ms"].tolist()

colors = (
    [COLORS["sequential"]]
    + [COLORS["hybrid"]] * len(summary)
)

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(
    labels,
    times,
    color=colors,
    width=0.62,
    yerr=errors,
    capsize=5,
    error_kw={
        "elinewidth": 1.5,
        "capthick": 1.5
    }
)

add_labels(ax, bars, " ms", 2)

ax.set_title(
    "Execution Time Comparison",
    loc="left",
    pad=18
)

ax.set_ylabel("Average Execution Time (ms)")
ax.set_xlabel("Implementation Configuration")

ax.tick_params(axis="x", rotation=20)

ax.set_ylim(0, max(times) * 1.25)

fig.tight_layout()

save_chart(fig, "01_execution_time.png")


# ============================================================
# 2. SPEEDUP ANALYSIS
# ============================================================

print("\nGenerating speedup chart...")

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(
    summary["Configuration"],
    summary["Speedup"],
    color=COLORS["speedup"],
    width=0.58
)

add_labels(ax, bars, "×", 2)

ax.axhline(
    y=1,
    color=COLORS["ideal"],
    linestyle="--",
    linewidth=1.5,
    label="Sequential baseline (1×)"
)

ax.set_title(
    "Parallel Speedup",
    loc="left",
    pad=18
)

ax.set_ylabel("Speedup (Sequential / Parallel)")
ax.set_xlabel("MPI × OpenMP Configuration")

ax.tick_params(axis="x", rotation=20)

ax.legend(frameon=False)

ax.set_ylim(0, summary["Speedup"].max() * 1.22)

fig.tight_layout()

save_chart(fig, "02_speedup.png")


# ============================================================
# 3. PARALLEL EFFICIENCY
# ============================================================

print("\nGenerating parallel efficiency chart...")

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(
    summary["Configuration"],
    summary["Efficiency_Percent"],
    color=COLORS["efficiency"],
    width=0.58
)

add_labels(ax, bars, "%", 1)

ax.axhline(
    y=100,
    color=COLORS["ideal"],
    linestyle="--",
    linewidth=1.5,
    label="Ideal efficiency (100%)"
)

ax.set_title(
    "Parallel Efficiency",
    loc="left",
    pad=18
)

ax.set_ylabel("Efficiency (%)")
ax.set_xlabel("MPI × OpenMP Configuration")

ax.tick_params(axis="x", rotation=20)

ax.legend(frameon=False)

ax.set_ylim(
    0,
    max(110, summary["Efficiency_Percent"].max() * 1.2)
)

fig.tight_layout()

save_chart(fig, "03_parallel_efficiency.png")


# ============================================================
# 4. RUNTIME DISTRIBUTION (BOX PLOT)
# ============================================================

print("\nGenerating runtime distribution chart...")

box_data = []
box_labels = []

for _, row in summary.iterrows():

    config_data = hybrid[
        (hybrid["MPI_Processes"] == row["MPI_Processes"])
        & (hybrid["OpenMP_Threads"] == row["OpenMP_Threads"])
    ]["Execution_Time_ms"]

    box_data.append(config_data.values)
    box_labels.append(row["Configuration"])

# Add sequential measurements
box_data.insert(
    0,
    sequential["Execution_Time_ms"].values
)

box_labels.insert(0, "Sequential")

fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(
    box_data,
    tick_labels=box_labels,
    patch_artist=True,
    widths=0.55,
    showmeans=True
)

box_colors = (
    [COLORS["sequential"]]
    + [COLORS["hybrid"]] * len(summary)
)

for patch, color in zip(bp["boxes"], box_colors):

    patch.set_facecolor(color)
    patch.set_alpha(0.75)

for median in bp["medians"]:

    median.set_color("#0f172a")
    median.set_linewidth(2)

ax.set_title(
    "Execution Time Distribution Across Runs",
    loc="left",
    pad=18
)

ax.set_ylabel("Execution Time (ms)")
ax.set_xlabel("Configuration")

ax.tick_params(axis="x", rotation=20)

fig.tight_layout()

save_chart(fig, "04_runtime_distribution.png")


# ============================================================
# 5. MPI VS OPENMP COMPARISON
# ============================================================

print("\nGenerating MPI vs OpenMP comparison...")

worker_counts = sorted(summary["Workers"].unique())

for workers in worker_counts:

    subset = summary[summary["Workers"] == workers]

    if len(subset) < 2:
        continue

    fig, ax = plt.subplots(figsize=(9, 5))

    comparison_colors = [
        COLORS["hybrid"],
        COLORS["accent"],
        COLORS["speedup"],
        COLORS["efficiency"]
    ]

    bars = ax.bar(
        subset["Configuration"],
        subset["Mean_Time_ms"],
        color=comparison_colors[:len(subset)],
        width=0.55,
        yerr=subset["Std_Time_ms"],
        capsize=5
    )

    add_labels(ax, bars, " ms", 2)

    ax.set_title(
        f"MPI vs OpenMP — {workers} Total Workers",
        loc="left",
        pad=18
    )

    ax.set_ylabel("Average Execution Time (ms)")
    ax.set_xlabel("Configuration")

    ax.set_ylim(0, subset["Mean_Time_ms"].max() * 1.3)

    fig.tight_layout()

    save_chart(
        fig,
        f"05_worker_comparison_{workers}.png"
    )


# ============================================================
# 6. ITERATION COMPARISON
# ============================================================

print("\nGenerating iteration comparison...")

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(
    summary["Configuration"],
    summary["Mean_Iterations"],
    color=COLORS["accent"],
    width=0.58,
    yerr=summary["Std_Iterations"],
    capsize=5
)

add_labels(ax, bars, "", 2)

ax.set_title(
    "Convergence Iteration Comparison",
    loc="left",
    pad=18
)

ax.set_ylabel("Average Number of Iterations")
ax.set_xlabel("MPI × OpenMP Configuration")

ax.tick_params(axis="x", rotation=20)

fig.tight_layout()

save_chart(fig, "06_iterations.png")


# ============================================================
# 7. SCALABILITY CHART
# ============================================================

print("\nGenerating scalability chart...")

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    summary["Workers"],
    summary["Mean_Time_ms"],
    marker="o",
    markersize=9,
    linewidth=2.5,
    color=COLORS["hybrid"],
    label="Measured execution time"
)

for _, row in summary.iterrows():

    ax.annotate(
        row["Configuration"],
        (row["Workers"], row["Mean_Time_ms"]),
        xytext=(7, 9),
        textcoords="offset points",
        fontsize=8
    )

ax.set_title(
    "Execution Time vs Number of Workers",
    loc="left",
    pad=18
)

ax.set_xlabel("Total MPI × OpenMP Workers")
ax.set_ylabel("Average Execution Time (ms)")

ax.legend(frameon=False)

fig.tight_layout()

save_chart(fig, "07_scalability.png")


# ============================================================
# 8. BEST CONFIGURATION SUMMARY
# ============================================================

best = summary.loc[summary["Mean_Time_ms"].idxmin()]

print("\n========== BEST CONFIGURATION ==========")

print(f"Configuration : {best['Configuration']}")
print(f"Workers       : {int(best['Workers'])}")
print(f"Average Time  : {best['Mean_Time_ms']:.3f} ms")
print(f"Std Dev       : {best['Std_Time_ms']:.3f} ms")
print(f"Speedup       : {best['Speedup']:.2f}×")
print(f"Efficiency     : {best['Efficiency_Percent']:.2f}%")

print("\n========== OUTPUT ==========")

print(f"Charts folder : {OUTPUT_DIR}/")
print(f"Summary CSV   : {OUTPUT_DIR}/benchmark_summary.csv")

print("\nAll visualizations generated successfully!")
