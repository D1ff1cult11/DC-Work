# Track 1: Multi-Class Solar PV Defect Screening & Diagnosis
## Autonomous UAV Thermal Infrared Inspection (IEC TS 62446-3)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8%20(sm__120)-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industrial-grade deep learning benchmark evaluating **13 neural architectures and physics-guided front-ends** on 20,000 radiometrically calibrated aerial Long-Wave Infrared (LWIR 8–14 µm) solar module images under severe **57:1 class imbalance**.

---

## 1. Benchmark Overview

Real-time drone surveillance of utility-scale solar farms requires high-throughput screening to identify anomalous modules across millions of panels. This track formulates the challenge as **12-class physical fault diagnosis** compliant with international standard **IEC TS 62446-3**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  12-CLASS DEFECT TAXONOMY & SPLIT BREAKDOWN                                            │
├──────────────────────┬────────┬────────┬───────┬───────┬───────────────────────────────┤
│ Class Name           │ Total  │ Train  │ Val   │ Test  │ Physical Failure Mechanism    │
├──────────────────────┼────────┼────────┼───────┼───────┼───────────────────────────────┤
│ 0. No-Anomaly        │ 10,000 │  7,000 │ 1,500 │ 1,500 │ Healthy baseline (ΔT < 1°C)   │
│ 1. Cell              │  1,877 │  1,314 │   281 │   282 │ Single-cell micro-shunt       │
│ 2. Vegetation        │  1,639 │  1,147 │   246 │   246 │ Overgrown weeds / tree shade  │
│ 3. Diode             │  1,499 │  1,049 │   225 │   225 │ Sub-string bypass diode open  │
│ 4. Cell-Multi        │  1,288 │    902 │   193 │   193 │ Multiple scattered hot cells  │
│ 5. Shadowing         │  1,056 │    739 │   158 │   159 │ Racking / pole shadows        │
│ 6. Cracking          │    940 │    658 │   141 │   141 │ Silicon wafer fracture        │
│ 7. Offline-Module    │    827 │    579 │   124 │   124 │ Open-circuit panel (cold)     │
│ 8. Hot-Spot          │    249 │    174 │    37 │    38 │ Localized junction burn (fire)│
│ 9. Hot-Spot-Multi    │    246 │    172 │    37 │    37 │ Clustered severe hotspots     │
│ 10. Soiling          │    204 │    143 │    31 │    30 │ Dust / bird droppings         │
│ 11. Diode-Multi      │    175 │    123 │    26 │    26 │ Multi-string bypass failure   │
├──────────────────────┼────────┼────────┼───────┼───────┼───────────────────────────────┤
│ Total                │ 20,000 │ 14,000 │ 3,000 │ 3,000 │ 57.1:1 Maximum Imbalance Ratio│
└──────────────────────┴────────┴────────┴───────┴───────┴───────────────────────────────┘
```

---

## 2. Empirical Benchmark Matrix (Held-Out Test Set: 3,000 Images)

All 13 configurations trained with AdamW ($3\times10^{-4}$ LR, Cosine Annealing, 15 epochs) on an NVIDIA GeForce RTX 5060 Laptop GPU:

| Config ID | Architecture | Front-End Descriptor | Loss Function | Accuracy | Macro-F1 | Precision | Recall | Kappa | Params | FPS (GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A2** | **ResNet-18** | Standard | Weighted CE | **68.45%** | **55.15%** | 56.40% | 55.70% | **0.5891** | 11.18M | **5,146.7** |
| **B1** | **SegNeXt (MSCA)** | Standard | Weighted CE | **61.67%** | **51.11%** | 54.02% | 51.52% | **0.5193** | 2.63M | **3,608.9** |
| **B3** | SegNeXt | SFOC | Weighted CE | 59.10% | 47.51% | 50.84% | 47.66% | 0.4859 | 2.63M | 3,557.6 |
| **A1** | MobileNetV3-Small| Standard | Weighted CE | 51.55% | 40.30% | 46.10% | 41.77% | 0.4132 | 1.53M | 3,941.7 |
| **B5** | SegNeXt | FSFOC | Focal Loss | 45.68% | 38.90% | 44.59% | 38.86% | 0.3541 | 2.63M | 1,046.5 |
| **B6** | SegNeXt | FSFOC-8 | Weighted CE | 45.88% | 38.61% | 44.20% | 39.46% | 0.3592 | 2.64M | 977.6 |
| **B4** | SegNeXt | CFOG | Weighted CE | 43.74% | 38.29% | 42.17% | 39.06% | 0.3374 | 2.63M | 3,407.5 |
| **B2** | SegNeXt | FSFOC-4 | Weighted CE | 45.54% | 38.23% | 44.26% | 38.50% | 0.3537 | 2.63M | 3,560.0 |
| **A3** | **SegFormer (Paper)**| Standard | Weighted CE | **48.08%** | **38.18%** | 45.71% | 38.87% | **0.3742** | 1.94M | **1,162.3** |
| **03** | SegNeXt (Pilot) | Radiometric| Weighted CE | 61.90% | 30.41% | 54.12% | 32.22% | 0.4431 | 2.63M | 445.7 |
| **B7** | SegFormer | FSFOC | Weighted CE | 31.09% | 28.50% | 38.65% | 30.95% | 0.2223 | 1.95M | 1,096.5 |
| **01** | **SegFormer (Paper)**| Standard | **Unweighted CE**| **62.10%** | **24.23%** | 33.68% | 27.24% | **0.4286** | 1.94M | **3,192.2** |
| **02** | MobileNetV3 (Pilot)| Standard | Unweighted CE | 53.42% | 11.41% | 22.84% | 13.91% | 0.1741 | 1.53M | 463.5 |

---

## 3. Key Findings

1. **Failure of Unweighted Optimization on Imbalance:**  
   As-published unweighted Cross-Entropy causes transformers to predict majority healthy panels, collapsing to **24.23% Macro-F1** and 0.0% sensitivity on rare defects.
2. **Normalized Inverse-Frequency Class Weighting:**  
   Setting $w_c = \frac{N}{C \cdot N_c}$ upweights rare classes ($w_{\text{Diode-Multi}} = 9.52$, $w_{\text{No-Anomaly}} = 0.167$), boosting `Diode-Multi` F1 from **0.0% to 75.9%** and recovering SegFormer to **38.18%**.
3. **SegNeXt MSCA Superiority:**  
   SegNeXt with Multi-Scale Convolutional Attention strip kernels ($1\times7, 1\times11, 1\times21$) outperforms SegFormer by **+12.93% Macro-F1 (51.11% vs. 38.18%)** while running **3.1× faster (3,608.9 FPS vs. 1,162.3 FPS)**.

---

## 4. Quick Start

```bash
# Run the complete 13-model ablation benchmark
python run_ablation_study.py --epochs 15 --batch_size 64

# Audit and verify all test results dynamically
python audit_results.py
```
