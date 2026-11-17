# Comprehensive Ablation Study & Failure Analysis Report
## Solar PV Thermal Defect Detection (LWIR 8-14 um)

> Total configurations evaluated: **13**

### Ablation Results Matrix

| Config | Architecture | Front-End | Loss | Acc | Macro-F1 | Precision | Recall | Kappa | Params | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01_paper_segformer** | SegFormer MiT-B0 | None | CE | 62.10% | **24.23%** | 27.89% | 25.65% | 0.3975 | 1.94M | 3192.2 |
| **02_baseline_mobilenet_v3_small** | MobileNetV3 | None | CE | 53.42% | **11.41%** | 26.50% | 11.97% | 0.1465 | 1.53M | 463.5 |
| **03_radiometric_sota_segnext** | ? | ? | ? | 61.90% | **30.41%** | 40.10% | 28.84% | 0.3846 | 2.63M | 445.7 |
| **A1_MobileNetV3_WCE** | MobileNetV3 | None | WCE | 51.55% | **40.30%** | 38.68% | 50.27% | 0.3958 | 1.53M | 3941.7 |
| **A2_ResNet18_WCE** | ResNet18 | None | WCE | 68.45% | **55.15%** | 52.16% | 64.00% | 0.5888 | 11.18M | 5146.7 |
| **A3_SegFormer_WCE** | SegFormer MiT-B0 | None | WCE | 48.08% | **38.18%** | 37.30% | 48.29% | 0.3603 | 1.94M | 1162.3 |
| **B1_SegNeXt_Only** | SegNeXt MSCA | None | WCE | 61.67% | **51.11%** | 49.00% | 61.57% | 0.5123 | 2.63M | 3608.9 |
| **B2_FSFOC4_SegNeXt_WCE** | SegNeXt MSCA | FSFOC-4 | WCE | 45.54% | **38.23%** | 35.56% | 49.89% | 0.3442 | 2.63M | 3560.0 |
| **B3_SFOC_SegNeXt** | SegNeXt MSCA | SFOC-2D | WCE | 59.10% | **47.51%** | 45.08% | 59.15% | 0.4849 | 2.63M | 3557.6 |
| **B4_CFOG_SegNeXt** | SegNeXt MSCA | CFOG-8 | WCE | 43.74% | **38.29%** | 35.74% | 50.01% | 0.3356 | 2.63M | 3407.5 |
| **B5_FSFOC_SegNeXt_Focal** | SegNeXt MSCA | FSFOC-4 | Focal | 45.68% | **38.90%** | 36.19% | 50.12% | 0.3454 | 2.63M | 1046.5 |
| **B6_FSFOC8_SegNeXt** | SegNeXt MSCA | FSFOC-8 | WCE | 45.88% | **38.61%** | 36.23% | 49.39% | 0.3463 | 2.63M | 977.6 |
| **B7_FSFOC_SegFormer** | SegFormer MiT-B0 | None | CE | 31.09% | **28.50%** | 28.62% | 40.66% | 0.2175 | 1.94M | 1096.5 |

---

### Per-Class F1 Breakdown (Top 3 Models)

| Class | Support | **A2_ResNet18_WCE** | **B1_SegNeXt_Only** | **B3_SFOC_SegNeXt** |
| :--- | :---: | :---: | :---: | :---: |
| `No-Anomaly` | 1500 | 83.3% | 75.9% | 73.8% |
| `Cell` | 281 | 57.9% | 52.7% | 46.8% |
| `Cell-Multi` | 193 | 41.8% | 34.2% | 32.4% |
| `Cracking` | 141 | 66.9% | 65.0% | 62.0% |
| `Diode` | 224 | 88.6% | 85.7% | 87.4% |
| `Diode-Multi` | 26 | 75.4% | 75.9% | 59.5% |
| `Hot-Spot` | 37 | 40.7% | 29.4% | 22.0% |
| `Hot-Spot-Multi` | 36 | 32.2% | 38.4% | 34.2% |
| `Offline-Module` | 124 | 47.5% | 37.6% | 32.4% |
| `Shadowing` | 158 | 50.1% | 43.5% | 47.6% |
| `Soiling` | 30 | 20.8% | 24.1% | 20.0% |
| `Vegetation` | 245 | 56.5% | 51.0% | 52.0% |

---

### Key Ablation Insights

1. **FSFOC Front-End Impact (B1 vs B2):** Comparing SegNeXt alone vs FSFOC+SegNeXt quantifies the exact contribution of physics-informed steerable gradient decomposition.
2. **SFOC vs FSFOC (B2 vs B3):** Full 2D steerable kernels vs separable 1D fast approximation — trades accuracy for edge deployability.
3. **CFOG Alternative (B4):** Channel Features of Oriented Gradients as an alternative orientation descriptor.
4. **Focal Loss Impact (B2 vs B5):** Proves whether down-weighting easy majority-class samples helps rare defect detection beyond class-weighted CE.
5. **Orientation Granularity (B2 vs B6):** 4 vs 8 steerable orientations — tests whether finer angular resolution improves detection of angled cracks and diode strings.