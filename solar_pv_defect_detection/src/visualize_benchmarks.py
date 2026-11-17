"""
Publication Figure Generator for Benchmarks.
Generates:
1. training_loss_curves.png: Convergence curves across pipelines.
2. validation_f1_curves.png: Validation Macro-F1 trajectory.
3. class_f1_comparison_barchart.png: 12-class comparative breakdown.
4. tradeoff_fps_vs_f1.png: Latency (FPS) vs Accuracy/F1 Pareto frontier.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "benchmark_results")
FIGURES_DIR = os.path.join(BENCHMARK_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_all_benchmarks():
    p2_dir = os.path.join(BENCHMARK_DIR, "02_baseline_mobilenet_v3_small")
    p1_dir = os.path.join(BENCHMARK_DIR, "01_paper_segformer")
    p3_dir = os.path.join(BENCHMARK_DIR, "03_radiometric_sota_segnext")

    pipelines = {
        "Baseline (MobileNetV3)": (p2_dir, "#4A90E2", "--"),
        "Paper SOTA (SegFormer MiT-B0)": (p1_dir, "#F5A623", "-."),
        "Our Proposal (NR-SegNeXt + FSFOC)": (p3_dir, "#2E7D32", "-"),
    }

    metrics_data = {}
    history_data = {}

    for name, (dirpath, color, ls) in pipelines.items():
        m_file = os.path.join(dirpath, "test_metrics.json")
        h_file = os.path.join(dirpath, "train_history.json")
        if os.path.exists(m_file) and os.path.exists(h_file):
            with open(m_file, "r") as f:
                metrics_data[name] = (json.load(f), color, ls)
            with open(h_file, "r") as f:
                history_data[name] = (json.load(f), color, ls)

    if not metrics_data:
        print("[!] No completed benchmark runs found to visualize.")
        return

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # 1. Training & Validation Loss Curves
    if history_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        for name, (hist, color, ls) in history_data.items():
            epochs = range(1, len(hist["train_loss"]) + 1)
            ax1.plot(epochs, hist["train_loss"], label=name, color=color, linestyle=ls, linewidth=2.2, marker="o", markersize=4)
            ax2.plot(epochs, hist["val_loss"], label=name, color=color, linestyle=ls, linewidth=2.2, marker="s", markersize=4)

        ax1.set_title("Training Loss Convergence", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Cross-Entropy / Radiometric Loss", fontsize=11)
        ax1.legend(frameon=True)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.set_title("Validation Loss Trajectory", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Validation Loss", fontsize=11)
        ax2.legend(frameon=True)
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        loss_fig = os.path.join(FIGURES_DIR, "training_loss_curves.png")
        plt.savefig(loss_fig, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved: {loss_fig}")

    # 2. Validation Macro-F1 Progression
    if history_data:
        plt.figure(figsize=(8, 5), dpi=300)
        for name, (hist, color, ls) in history_data.items():
            epochs = range(1, len(hist["val_f1"]) + 1)
            val_f1_pct = [v * 100 for v in hist["val_f1"]]
            plt.plot(epochs, val_f1_pct, label=name, color=color, linestyle=ls, linewidth=2.4, marker="o", markersize=5)

        plt.title("Validation Macro-F1 Progression Across Epochs", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Macro-F1 Score (%)", fontsize=11)
        plt.legend(frameon=True, loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        f1_fig = os.path.join(FIGURES_DIR, "validation_f1_curves.png")
        plt.savefig(f1_fig, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved: {f1_fig}")

    # 3. 12-Class F1 Comparative Breakdown Bar Chart
    first_pipeline = list(metrics_data.keys())[0]
    classes = list(metrics_data[first_pipeline][0].get("per_class", {}).keys())

    if classes:
        plt.figure(figsize=(15, 6), dpi=300)
        x = np.arange(len(classes))
        n_pipes = len(metrics_data)
        width = 0.8 / n_pipes

        for idx, (name, (m, color, _)) in enumerate(metrics_data.items()):
            f1_scores = [m.get("per_class", {}).get(c, {}).get("f1_score", m.get("per_class", {}).get(c, {}).get("f1", 0)) * 100 for c in classes]
            offset = (idx - n_pipes / 2 + 0.5) * width
            plt.bar(x + offset, f1_scores, width, label=name, color=color, alpha=0.88, edgecolor="black", linewidth=0.5)

        plt.title("Per-Class Defect Diagnosis F1-Score Comparison (12 Defect Taxonomies)", fontsize=13, fontweight="bold")
        plt.xlabel("Photovoltaic Defect Anomaly Class", fontsize=11)
        plt.ylabel("F1-Score (%)", fontsize=11)
        plt.xticks(x, classes, rotation=35, ha="right", fontsize=9.5)
        plt.ylim(0, 105)
        plt.legend(frameon=True, loc="upper right")
        plt.grid(True, axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        class_fig = os.path.join(FIGURES_DIR, "class_f1_comparison_barchart.png")
        plt.savefig(class_fig, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved: {class_fig}")

    # 4. Latency (FPS) vs Macro-F1 Pareto Frontier
    plt.figure(figsize=(8, 5.5), dpi=300)
    for name, (m, color, _) in metrics_data.items():
        fps = m.get("inference_fps", 0)
        f1 = m.get("macro_f1", 0) * 100
        params = m.get("parameters", 1e6) / 1e6
        bubble_size = max(50, min(800, params * 80))

        plt.scatter(fps, f1, s=bubble_size, color=color, alpha=0.7, edgecolors="black", linewidth=1.5, label=f"{name} ({params:.2f}M params)")
        plt.annotate(
            f"{name}\n{f1:.1f}% F1, {fps:.0f} FPS",
            (fps, f1),
            textcoords="offset points",
            xytext=(10, -5),
            ha="left",
            fontsize=9.5,
            fontweight="bold"
        )

    plt.title("Edge Efficiency: Inference Latency (FPS) vs Macro-F1 Score", fontsize=13, fontweight="bold")
    plt.xlabel("Inference Throughput (Frames Per Second - FPS)", fontsize=11)
    plt.ylabel("Macro-F1 Score (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="lower left", fontsize=8.5)
    plt.tight_layout()
    pareto_fig = os.path.join(FIGURES_DIR, "tradeoff_fps_vs_f1.png")
    plt.savefig(pareto_fig, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved: {pareto_fig}")


if __name__ == "__main__":
    plot_all_benchmarks()
