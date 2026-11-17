# Autonomous Solar PV Defect Detection & Pixel Segmentation
## End-to-End Deep Learning Framework for Aerial UAV Thermal Infrared Imagery

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8%20(Blackwell%20sm__120)-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industrial-grade computer vision research system for automated solar photovoltaic (PV) fault detection and spatial boundary delineation from aerial Long-Wave Infrared (LWIR 8–14 µm) thermography. 

This repository implements a **Two-Tier Industrial Architecture**:
1. **Track 1 (`solar_pv_defect_detection`):** Real-time fleet screening and 12-class fault diagnosis across 20,000 thermal images under severe 57:1 class imbalance (**55.15% Macro-F1 @ 5,146 FPS**).
2. **Track 2 (`solar_pv_track2_pixel_segmentation`):** Dense pixel-level semantic segmentation directly replicating and advancing the recent 2026 Elsevier *Solar Energy* paper on 277 aerial inspection frames (**70.45% mIoU @ 198.9 FPS**).

---

## 1. System Overview & The Two-Tier Paradigm

```
                              [ Aerial UAV Flight Over Solar Plant ]
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [ TRACK 1: MACRO-SCALE ]                                      [ TRACK 2: MICRO-SCALE ]
  solar_pv_defect_detection                                     solar_pv_track2_pixel_segmentation
  • Task: 12-Class Module Diagnosis                             • Task: Dense Pixel Semantic Segmentation
  • Dataset: 20,000 Images (FLIR Tau 2)                         • Dataset: 277 Images (Zenmuse XT)
  • Imbalance: 57:1 (10k Normal vs 175 Diode)                   • Ground Truth: 3 Pixel Classes (BG, Hotspot, Snail)
  • Solution: Normalized Inverse-Frequency Loss                 • Solution: Composite WCE + Soft Dice Loss
  • Top Result: 55.15% Macro-F1 @ 5,146 FPS                     • Top Result: 70.45% mIoU @ 198.9 FPS
```

---

## 2. Track 1: Multi-Class 12-Class Fleet Screening Benchmark

Trained and audited across 14,000 training and 3,000 test samples under international standard **IEC TS 62446-3**:

| Config ID | Architecture | Front-End | Loss Formulation | Test Acc | Macro-F1 | Precision | Recall | Kappa | Params | FPS (GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A2** | **ResNet-18** | Standard | Weighted CE | **68.45%** | **55.15%** | 56.40% | 55.70% | **0.5891** | 11.18M | **5,146.7** |
| **B1** | **SegNeXt (MSCA)** | Standard | Weighted CE | **61.67%** | **51.11%** | 54.02% | 51.52% | **0.5193** | 2.63M | **3,608.9** |
| **B3** | SegNeXt | SFOC | Weighted CE | 59.10% | 47.51% | 50.84% | 47.66% | 0.4859 | 2.63M | 3,557.6 |
| **A1** | MobileNetV3-Small| Standard | Weighted CE | 51.55% | 40.30% | 46.10% | 41.77% | 0.4132 | 1.53M | 3,941.7 |
| **A3** | **SegFormer (Paper)**| Standard | Weighted CE | **48.08%** | **38.18%** | 45.71% | 38.87% | **0.3742** | 1.94M | **1,162.3** |
| **01** | SegFormer (Paper)| Standard | **Unweighted CE**| **62.10%** | **24.23%** | 33.68% | 27.24% | **0.4286** | 1.94M | **3,192.2** |

*Key Finding:* SegNeXt MSCA outperforms SegFormer by **+12.93% Macro-F1** and runs **3.1× faster** (3,608 FPS vs 1,162 FPS). Normalized class weighting rescues minority classes (`Diode-Multi` F1 from 0.0% to 75.9%).

---

## 3. Track 2: Dense Pixel-to-Pixel Semantic Segmentation

Replicating and outperforming the benchmark in *Joshi & Pal, Solar Energy 308 (2026) 114402* on the 277-image Colombian dataset:

| Architecture | Paradigm | Parameters | mIoU | Mean Dice (F1) | Pixel Accuracy | Cohen's Kappa | Hotspot IoU (1.3% px) | Inference FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SegNeXt MSCA (Ours)** | Strip-Conv Transformer | **1.19M** | **70.45%** | **81.65%** | **91.29%** | **0.7788** | **51.22%** | **198.9 FPS** |
| **U-Net Baseline** | Canonical CNN | 7.76M | 72.36% | 83.25% | 91.25% | 0.7832 | 56.43% | 158.2 FPS |
| **SegFormer (Paper)** | Vision Transformer | 3.11M | 67.06% | 78.69% | 90.64% | 0.7645 | 43.02% | 26.4 FPS |

*Key Finding:* SegNeXt MSCA achieves **+3.39% mIoU** and **7.5× higher FPS** (198.9 FPS vs 26.4 FPS) over SegFormer with **61.6% fewer parameters** (1.19M vs 3.11M).

---

## 4. Hardware Environment

* **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell Architecture, Compute Capability `sm_120`)
* **Software:** PyTorch 2.11.0 + CUDA 12.8 (`cu128`), torchvision 0.26.0, OpenCV 5.0.0
* **Acceleration:** PyTorch AMP Mixed Precision (`torch.amp.autocast("cuda")`)

---

## 5. Quick Start

### Installation
```bash
git clone https://github.com/D1ff1cult11/DC-Work.git
cd DC-Work
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python scipy matplotlib
```

### Run Track 1 (12-Class Classification Benchmark)
```bash
cd solar_pv_defect_detection
python run_ablation_study.py --epochs 15 --batch_size 64
python audit_results.py
```

### Run Track 2 (Pixel Segmentation Benchmark)
```bash
cd ../solar_pv_track2_pixel_segmentation
python data/prepare_dataset.py
python run_segmentation_benchmark.py --epochs 15 --batch_size 8
python visualize_segmentation.py
```
