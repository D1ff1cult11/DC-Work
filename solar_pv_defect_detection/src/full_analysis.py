"""
Comprehensive Failure Analysis, Confusion Matrices, and Ablation Visualization Suite.
Generates:
  1. Confusion matrix heatmaps for every model
  2. Ablation study table (markdown + figure)
  3. Per-class failure bar chart
  4. Class-wise improvement delta chart (Ours vs Baselines)
  5. Training convergence comparison for all runs
  6. Pareto frontier (FPS vs F1) with ALL models
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "benchmark_results")
FIGURES_DIR = os.path.join(BENCHMARK_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.append(PROJECT_ROOT)
from src.dataset import CLASSES


def load_all_results():
    """Scan benchmark_results for all test_metrics.json files."""
    results = {}
    histories = {}
    for entry in sorted(os.listdir(BENCHMARK_DIR)):
        entry_path = os.path.join(BENCHMARK_DIR, entry)
        if not os.path.isdir(entry_path):
            continue
        metrics_file = os.path.join(entry_path, "test_metrics.json")
        history_file = os.path.join(entry_path, "train_history.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                results[entry] = json.load(f)
        if os.path.exists(history_file):
            with open(history_file) as f:
                histories[entry] = json.load(f)
    return results, histories


def plot_confusion_matrix(cm, class_names, title, out_path):
    """Plot a single confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
    cm_np = np.array(cm, dtype=float)

    # Normalize by row (true labels)
    row_sums = cm_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_np / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (Row-Normalized)")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(class_names, fontsize=8.5)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Annotate cells with count + percentage
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = int(cm_np[i, j])
            pct = cm_norm[i, j] * 100
            color = "white" if pct > 50 else "black"
            if count > 0:
                ax.text(j, i, f"{count}\n{pct:.0f}%", ha="center", va="center", fontsize=6.5, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  [+] Confusion matrix: {out_path}")


def plot_all_confusion_matrices(results):
    """Generate confusion matrix heatmap for every model."""
    cm_dir = os.path.join(FIGURES_DIR, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    for run_name, m in results.items():
        if "confusion_matrix" in m:
            title = m.get("run_name", run_name)
            acc = m["accuracy"] * 100
            f1 = m["macro_f1"] * 100
            full_title = f"{title}\nAcc: {acc:.1f}% | Macro-F1: {f1:.1f}%"
            out_path = os.path.join(cm_dir, f"cm_{run_name}.png")
            plot_confusion_matrix(m["confusion_matrix"], CLASSES, full_title, out_path)


def plot_ablation_barchart(results):
    """Bar chart comparing Macro-F1 across all ablation configurations."""
    names = []
    f1_scores = []
    colors = []

    color_map = {
        "01_": "#9E9E9E",   # grey - old runs
        "02_": "#9E9E9E",
        "03_": "#9E9E9E",
        "A1": "#4A90E2",    # blue - baselines
        "A2": "#42A5F5",
        "A3": "#F5A623",    # orange - paper
        "B1": "#FF7043",    # red-ish - ablation
        "B2": "#2E7D32",    # green - our full proposal
        "B3": "#66BB6A",
        "B4": "#8BC34A",
        "B5": "#1B5E20",    # dark green - focal
        "B6": "#4CAF50",
        "B7": "#AB47BC",    # purple - FSFOC + SegFormer 2x2 factorial
    }

    for run_name, m in sorted(results.items()):
        label = m.get("run_name", run_name)
        names.append(label)
        f1_scores.append(m["macro_f1"] * 100)
        c = "#888888"
        for prefix, col in color_map.items():
            if run_name.startswith(prefix):
                c = col
                break
        colors.append(c)

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.2), 6), dpi=200)
    x = np.arange(len(names))
    bars = ax.bar(x, f1_scores, color=colors, edgecolor="black", linewidth=0.5, alpha=0.88)

    # Value labels on bars
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Macro-F1 Score (%)", fontsize=11)
    ax.set_title("Ablation Study: Macro-F1 Comparison Across All Configurations", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(0, max(f1_scores) * 1.15 if f1_scores else 100)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "ablation_macro_f1_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [+] Ablation bar chart: {out}")


def plot_class_improvement_delta(results):
    """Show per-class F1 improvement of best model vs paper baseline."""
    # Find paper baseline and best proposal
    paper_key = None
    best_key = None
    best_f1 = 0

    for k, m in results.items():
        if "segformer" in k.lower() and "01_" in k:
            paper_key = k
        if "A3" in k:
            paper_key = k  # prefer weighted version
        f1 = m["macro_f1"]
        if ("B2" in k or "B5" in k or "fsfoc" in k.lower()) and f1 > best_f1:
            best_f1 = f1
            best_key = k

    if not paper_key or not best_key:
        print("  [!] Cannot compute deltas: missing paper or best model")
        return

    paper_m = results[paper_key]
    best_m = results[best_key]

    deltas = []
    class_names = []
    for cls in CLASSES:
        p_f1 = paper_m.get("per_class", {}).get(cls, {}).get("f1_score", 0) * 100
        b_f1 = best_m.get("per_class", {}).get(cls, {}).get("f1_score", 0) * 100
        deltas.append(b_f1 - p_f1)
        class_names.append(cls)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=200)
    x = np.arange(len(class_names))
    colors_delta = ["#2E7D32" if d >= 0 else "#D32F2F" for d in deltas]
    bars = ax.bar(x, deltas, color=colors_delta, edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars, deltas):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + (0.5 if y >= 0 else -2.5),
                f"{val:+.1f}%", ha="center", va="bottom" if y >= 0 else "top",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("F1 Delta (%)", fontsize=11)
    ax.axhline(y=0, color="black", linewidth=0.8)
    paper_name = results[paper_key].get("run_name", paper_key)
    best_name = results[best_key].get("run_name", best_key)
    ax.set_title(f"Per-Class F1 Improvement: {best_name} vs {paper_name}", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "class_improvement_delta.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [+] Class improvement delta: {out}")


def plot_convergence_comparison(histories):
    """Training/validation F1 curves for ALL runs overlaid."""
    if not histories:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
    cmap = plt.get_cmap("tab10")

    for idx, (name, h) in enumerate(sorted(histories.items())):
        color = cmap(idx % 10)
        epochs = range(1, len(h["train_loss"]) + 1)
        label = name.replace("_", " ")[:30]
        ax1.plot(epochs, h["train_loss"], color=color, linewidth=1.8, label=label, alpha=0.8)
        ax2.plot(epochs, [v*100 for v in h["val_f1"]], color=color, linewidth=1.8, label=label, alpha=0.8, marker="o", markersize=3)

    ax1.set_title("Training Loss Convergence (All Configurations)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.legend(fontsize=7, frameon=True, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.set_title("Validation Macro-F1 Progression (All Configurations)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Macro-F1 (%)")
    ax2.legend(fontsize=7, frameon=True, loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "all_convergence_curves.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [+] Convergence curves: {out}")


def plot_pareto_frontier(results):
    """FPS vs Macro-F1 scatter for ALL models."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    cmap = plt.get_cmap("tab10")

    for idx, (name, m) in enumerate(sorted(results.items())):
        fps = m.get("inference_fps", 0)
        f1 = m["macro_f1"] * 100
        params = m.get("parameters", 1e6) / 1e6
        color = cmap(idx % 10)
        size = max(60, min(600, params * 60))
        ax.scatter(fps, f1, s=size, color=color, alpha=0.7, edgecolors="black", linewidth=1.2,
                   label=f"{m.get('run_name', name)[:25]} ({params:.1f}M)", zorder=3)

    ax.set_xlabel("Inference Throughput (FPS)", fontsize=11)
    ax.set_ylabel("Macro-F1 Score (%)", fontsize=11)
    ax.set_title("Efficiency Pareto Frontier: All Configurations", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=True, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "pareto_all_models.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [+] Pareto frontier: {out}")


def generate_ablation_report(results):
    """Generate comprehensive ablation markdown report."""
    report_lines = []
    report_lines.append("# Comprehensive Ablation Study & Failure Analysis Report")
    report_lines.append("## Solar PV Thermal Defect Detection (LWIR 8-14 um)\n")
    report_lines.append(f"> Total configurations evaluated: **{len(results)}**\n")

    # Master table
    report_lines.append("### Ablation Results Matrix\n")
    report_lines.append("| Config | Architecture | Front-End | Loss | Acc | Macro-F1 | Precision | Recall | Kappa | Params | FPS |")
    report_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for name, m in sorted(results.items()):
        rn = m.get("run_name", name)
        acc = f"{m['accuracy']*100:.2f}%"
        f1 = f"{m['macro_f1']*100:.2f}%"
        prec = f"{m['macro_precision']*100:.2f}%"
        rec = f"{m['macro_recall']*100:.2f}%"
        kap = f"{m['cohens_kappa']:.4f}"
        params = f"{m.get('parameters',0)/1e6:.2f}M"
        fps = f"{m.get('inference_fps',0):.1f}"

        # Infer architecture and front-end from name
        if "mobilenet" in name.lower():
            arch, fe, loss = "MobileNetV3", "None", "WCE" if "weight" in name.lower() else "CE"
        elif "resnet" in name.lower():
            arch, fe, loss = "ResNet18", "None", "WCE" if "weight" in name.lower() else "CE"
        elif "segformer" in name.lower():
            arch, fe, loss = "SegFormer MiT-B0", "None", "WCE" if "weight" in name.lower() else "CE"
        elif "segnext_only" in name.lower() or "B1" in name:
            arch, fe, loss = "SegNeXt MSCA", "None", "WCE"
        elif "cfog" in name.lower():
            arch, fe, loss = "SegNeXt MSCA", "CFOG-8", "WCE"
        elif "sfoc" in name.lower() and "fsfoc" not in name.lower():
            arch, fe, loss = "SegNeXt MSCA", "SFOC-2D", "WCE"
        elif "fsfoc8" in name.lower():
            arch, fe, loss = "SegNeXt MSCA", "FSFOC-8", "WCE"
        elif "B7" in name or ("fsfoc" in name.lower() and "segformer" in name.lower()):
            arch, fe, loss = "SegFormer MiT-B0", "FSFOC-4", "WCE"
        elif "focal" in name.lower():
            arch, fe, loss = "SegNeXt MSCA", "FSFOC-4", "Focal"
        elif "fsfoc" in name.lower():
            arch, fe, loss = "SegNeXt MSCA", "FSFOC-4", "WCE"
        else:
            arch, fe, loss = "?", "?", "?"

        report_lines.append(f"| **{rn}** | {arch} | {fe} | {loss} | {acc} | **{f1}** | {prec} | {rec} | {kap} | {params} | {fps} |")

    report_lines.append("\n---\n")

    # Per-class breakdown for top 3
    sorted_by_f1 = sorted(results.items(), key=lambda x: x[1]["macro_f1"], reverse=True)[:3]
    report_lines.append("### Per-Class F1 Breakdown (Top 3 Models)\n")
    header = "| Class | Support | " + " | ".join([f"**{r.get('run_name', n)[:20]}**" for n, r in sorted_by_f1]) + " |"
    sep = "| :--- | :---: | " + " | ".join([":---:" for _ in sorted_by_f1]) + " |"
    report_lines.append(header)
    report_lines.append(sep)

    for cls in CLASSES:
        vals = []
        support = "?"
        for n, r in sorted_by_f1:
            pc = r.get("per_class", {}).get(cls, {})
            f1_val = pc.get("f1_score", 0) * 100
            support = pc.get("support", "?")
            vals.append(f"{f1_val:.1f}%")
        report_lines.append(f"| `{cls}` | {support} | " + " | ".join(vals) + " |")

    report_lines.append("\n---\n")

    # Failure analysis section
    report_lines.append("### Key Ablation Insights\n")
    report_lines.append("1. **FSFOC Front-End Impact (B1 vs B2):** Comparing SegNeXt alone vs FSFOC+SegNeXt quantifies the exact contribution of physics-informed steerable gradient decomposition.")
    report_lines.append("2. **SFOC vs FSFOC (B2 vs B3):** Full 2D steerable kernels vs separable 1D fast approximation — trades accuracy for edge deployability.")
    report_lines.append("3. **CFOG Alternative (B4):** Channel Features of Oriented Gradients as an alternative orientation descriptor.")
    report_lines.append("4. **Focal Loss Impact (B2 vs B5):** Proves whether down-weighting easy majority-class samples helps rare defect detection beyond class-weighted CE.")
    report_lines.append("5. **Orientation Granularity (B2 vs B6):** 4 vs 8 steerable orientations — tests whether finer angular resolution improves detection of angled cracks and diode strings.")

    out_path = os.path.join(PROJECT_ROOT, "ablation_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[+] Ablation report: {out_path}")


def main():
    print(f"\n{'='*80}")
    print(f"  COMPREHENSIVE ANALYSIS & VISUALIZATION SUITE")
    print(f"{'='*80}\n")

    results, histories = load_all_results()
    print(f"[*] Found {len(results)} completed benchmark runs and {len(histories)} training histories.\n")

    if not results:
        print("[!] No results found. Run benchmarks first.")
        return

    print("[1/6] Generating confusion matrix heatmaps...")
    plot_all_confusion_matrices(results)

    print("[2/6] Generating ablation bar chart...")
    plot_ablation_barchart(results)

    print("[3/6] Generating per-class improvement delta chart...")
    plot_class_improvement_delta(results)

    print("[4/6] Generating convergence comparison curves...")
    plot_convergence_comparison(histories)

    print("[5/6] Generating Pareto frontier plot...")
    plot_pareto_frontier(results)

    print("[6/6] Generating ablation markdown report...")
    generate_ablation_report(results)

    print(f"\n{'='*80}")
    print(f"  ALL ANALYSIS COMPLETE")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
