"""Render the documented October 2026 study from its public aggregate JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

repo = Path(__file__).resolve().parents[2]
data = json.loads((repo / "examples/recorded-measurement.json").read_text(encoding="utf-8-sig"))
rows = data["per_seed"]
mean = data["across_seeds"]["balanced_accuracy"]["mean"] * 100
sd = data["across_seeds"]["balanced_accuracy"]["sample_sd"] * 100
delta = data["across_seeds"]["paired_delta_vs_fixed"]["mean"] * 100
ci = data["group_bootstrap"]["intervals"]["mean_paired_delta_vs_fixed"]
lo, hi = ci["lower"] * 100, ci["upper"] * 100
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "thyroid-recorded-study-20261003",
        "axes.labelcolor": "#334155",
        "text.color": "#142A3D",
    }
)
fig = plt.figure(figsize=(12.2, 6.8), facecolor="#FFFFFF")
ax1 = fig.add_axes([0.095, 0.29, 0.43, 0.47])
ax2 = fig.add_axes([0.63, 0.32, 0.31, 0.36])
fig.text(0.07, 0.91, "A measured baseline, with uncertainty visible", fontsize=21, weight="bold")
fig.text(
    0.07,
    0.855,
    "Source-TIRADS categories | Five AlexNet-style fits | One fixed held-out split",
    fontsize=12,
    color="#475569",
)
ax1.axvline(50, color="#94A3B8", linestyle=(0, (4, 4)), linewidth=1.3, zorder=1)
ax1.axvline(mean, color="#176B8D", alpha=0.23, linewidth=1.3, zorder=1)
y = list(range(len(rows)))[::-1]
values = [row["balanced_accuracy"] * 100 for row in rows]
ax1.scatter(values, y, s=76, color="#176B8D", zorder=3)
for val, pos in zip(values, y):
    ax1.text(val + 0.7, pos, f"{val:.2f}%", va="center", fontsize=10.5)
ax1.set_yticks(y, [f"Seed {row['seed']}" for row in rows])
ax1.set_xlim(47, 71)
ax1.set_ylim(-0.8, 4.8)
ax1.set_xticks([50, 55, 60, 65, 70])
ax1.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax1.set_xlabel("Held-out balanced accuracy", labelpad=11)
ax1.set_title("Every planned training run", loc="left", fontsize=13, weight="bold", pad=18)
ax1.text(50, -0.70, "Fixed 0.5 threshold: 50%", ha="left", fontsize=9, color="#64748B")
ax1.grid(axis="x", color="#E8EDF2", linewidth=0.7)
ax1.set_axisbelow(True)
ax2.axvline(0, color="#64748B", linestyle=(0, (4, 4)), linewidth=1.3)
ax2.errorbar(
    delta,
    0,
    xerr=[[delta - lo], [hi - delta]],
    fmt="o",
    markersize=8,
    color="#176B8D",
    ecolor="#176B8D",
    capsize=6,
    linewidth=2.8,
)
ax2.set_xlim(-2.5, 15)
ax2.set_ylim(-1, 1)
ax2.set_yticks([])
ax2.set_xticks([0, 5, 10, 15])
ax2.set_xlabel("Selected minus fixed threshold (pp)", labelpad=11, fontsize=10.5)
ax2.set_title("Paired threshold comparison", loc="left", fontsize=13, weight="bold", pad=42)
ax2.text(delta, 0.30, f"+{delta:.2f} pp", ha="center", fontsize=15, weight="bold")
ax2.text(delta, -0.32, f"95% interval: {lo:+.2f} to {hi:+.2f} pp", ha="center", fontsize=10)
ax2.grid(axis="x", color="#E8EDF2", linewidth=0.7)
for ax in (ax1, ax2):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="both", length=0, pad=8, labelcolor="#475569")
fig.text(
    0.095,
    0.17,
    f"Mean: {mean:.2f}% | Across-seed sample SD: {sd:.2f} pp",
    fontsize=11,
    weight="bold",
)
fig.text(
    0.63,
    0.17,
    "Interval includes zero; a reliably positive\nimprovement was not established.",
    fontsize=10.5,
    linespacing=1.5,
)
fig.text(
    0.07,
    0.085,
    "52 test images in 44 declared groups. The paired group-bootstrap interval is approximate and conditional on the fitted",
    fontsize=9.4,
    color="#64748B",
)
fig.text(
    0.07,
    0.055,
    "models and thresholds. Seed variation is separate. Source annotations are not confirmed cancer diagnoses.",
    fontsize=9.4,
    color="#64748B",
)
out = repo / "docs/visuals"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "measured-results.png", dpi=180, facecolor="white")
fig.savefig(out / "measured-results.svg", facecolor="white", metadata={"Date": None})
# Keep the vector artifact deterministic across Windows and Unix checkouts.
svg_path = out / "measured-results.svg"
svg_path.write_text(
    "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
    encoding="utf-8",
    newline="\n",
)
print(
    json.dumps(
        {
            "mean_percent": mean,
            "sample_sd_pp": sd,
            "delta_pp": delta,
            "interval_pp": [lo, hi],
            "png": str(out / "measured-results.png"),
        }
    )
)
