"""
Evaluation Metrics Suite for Solar PV Thermal Defect Detection.
Computes Overall Accuracy, Macro-F1, Class-wise Precision & Recall, and Cohen's Kappa.
"""

import numpy as np


def compute_metrics(y_true, y_pred, class_names=None):
    """
    Computes comprehensive multi-class classification metrics.
    y_true: 1D array or list of ground truth labels
    y_pred: 1D array or list of predicted class labels
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num_classes = len(class_names) if class_names is not None else int(max(np.max(y_true), np.max(y_pred)) + 1)

    # Accuracy
    accuracy = float(np.mean(y_true == y_pred))

    # Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Per-class Precision, Recall, F1
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precision_per_class.append(float(prec))
        recall_per_class.append(float(rec))
        f1_per_class.append(float(f1))

    # Macro averages
    macro_precision = float(np.mean(precision_per_class))
    macro_recall = float(np.mean(recall_per_class))
    macro_f1 = float(np.mean(f1_per_class))

    # Cohen's Kappa
    total_samples = len(y_true)
    po = accuracy
    pe = sum((np.sum(cm[c, :]) * np.sum(cm[:, c])) for c in range(num_classes)) / (total_samples ** 2) if total_samples > 0 else 0
    kappa = float((po - pe) / (1 - pe)) if (1 - pe) != 0 else 0.0

    report = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "cohens_kappa": kappa,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            (class_names[c] if class_names else str(c)): {
                "precision": precision_per_class[c],
                "recall": recall_per_class[c],
                "f1_score": f1_per_class[c],
                "support": int(np.sum(cm[c, :])),
            }
            for c in range(num_classes)
        }
    }
    return report


def print_evaluation_summary(metrics_dict):
    """
    Prints a formatted research-grade summary table of metrics.
    """
    print("\n" + "=" * 65)
    print("      SOLAR PV THERMAL DEFECT DETECTION - EVALUATION SUMMARY      ")
    print("=" * 65)
    print(f"  Overall Accuracy : {metrics_dict['accuracy'] * 100:.2f}%")
    print(f"  Macro Precision  : {metrics_dict['macro_precision'] * 100:.2f}%")
    print(f"  Macro Recall     : {metrics_dict['macro_recall'] * 100:.2f}%")
    print(f"  Macro F1-Score   : {metrics_dict['macro_f1'] * 100:.2f}%")
    print(f"  Cohen's Kappa    : {metrics_dict['cohens_kappa']:.4f}")
    print("-" * 65)
    print(f"{'Class Name':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support'}")
    print("-" * 65)
    for cls_name, vals in metrics_dict["per_class"].items():
        print(f"{cls_name:<20} | {vals['precision']*100:>8.2f}% | {vals['recall']*100:>8.2f}% | {vals['f1_score']*100:>8.2f}% | {vals['support']:<6}")
    print("=" * 65 + "\n")
