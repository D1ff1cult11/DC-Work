# Lightweight Transformer-Driven Solar PV Defect Detection

> **Directed Course (DC) Research Project**  
> **Target Journal / Venue:** IEEE Transactions on Industrial Informatics / Elsevier Solar Energy  
> **Key References:**  
> * **Paper:** *Lightweight transformer driven segmentation of defects in solar PV thermal imagery* (Elsevier *Solar Energy*, Vol 308, 2026, Article 114402; arXiv:2507.20680)  
> * **Dataset:** *Infrared Solar Modules* (Raptor Maps / Kaggle: `marcosgabriel/infrared-solar-modules`)

---

## 1. Project Background & Executive Answer

### Does the Kaggle Dataset work for this research?
**YES, it is an outstanding match in domain, modality, and physical defect mechanics, with one key architectural distinction:**

```
┌─────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│          ScienceDirect Paper (Joshi & Pal 2026)             │                   Kaggle Dataset (Raptor Maps)              │
├─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ • Thermal Infrared (TIR) drone imagery                     │ • Thermal Infrared (TIR) drone imagery                      │
│ • Detects: Hotspots, Snail Trails, Degradation              │ • Detects: Hotspots, Diodes, Cracks, Shading, Soiling       │
│ • Task: Semantic Segmentation (pixel-level defect mask)     │ • Task: Multi-Class Module Classification (12 classes)      │
│ • Dataset Size: 277 full thermal images                     │ • Dataset Size: 20,000 cropped thermal module images        │
└─────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### Why this is a major advantage for your Directed Course:
1. **Statistical Reliability:** Training a transformer on the paper's 277 images carries a high risk of overfitting. Benchmarking lightweight transformers on the **20,000 images** of the Kaggle dataset provides genuine statistical power and industrial credibility.
2. **Comprehensive Anomaly Taxonomy:** The Kaggle dataset covers **11 distinct anomaly types** (bypass diodes, micro-cracks, module-level offline states, soiling, vegetation shading), whereas the paper only evaluates two (hotspots and snail trails).
3. **Modality Adaptation Novelty:** It provides a direct path to formulate the academic thesis: **"Adapting Vision Foundation Models from RGB to Thermal Imagery using Lightweight Transformers and Few-Shot Prompting."**

> [!NOTE]
> If your professor strictly insists on pixel-level segmentation on full drone frames, reference **[Reference 180]** in the paper's bibliography: the open-access **Roboflow Universe "solar-panel-infrared-images"** dataset. Both tracks are fully documented in `docs/02_Dataset_Evaluation_and_Comparison.md`.

---

## 2. Repository Structure

```
solar_pv_defect_detection/
├── README.md                                  # Master project overview & executive briefing
├── docs/
│   ├── 01_Paper_Deep_Dive_and_Methodology.md  # Architectural breakdown of Joshi & Pal (Solar Energy 2026)
│   ├── 02_Dataset_Evaluation_and_Comparison.md# Kaggle vs. Paper dataset analysis & alignment
│   ├── 03_Professor_Briefing_and_Pitch.md    # Ready-to-send email and meeting talking points
│   └── 04_Technical_Implementation_Plan.md    # 8-week engineering timeline & experimental design
├── src/
│   ├── dataset.py                             # PyTorch Dataset with CLAHE thermal contrast enhancement
│   ├── metrics.py                             # Precision, Recall, Macro-F1, Confusion Matrix, Cohen's Kappa
│   ├── train_eval.py                          # Training, validation, testing loop with early stopping
│   └── models/
│       ├── lightweight_transformer.py         # SegFormer/MiT-inspired spatial-reduction transformer (~3.8M params)
│       └── baselines.py                       # MobileNetV3-Small and ResNet18 CNN baselines
└── data/
    └── README.md                              # Download instructions for Kaggle Infrared Solar Modules
```

---

## 3. Preprocessing & Model Architecture

### Thermal Preprocessing
Thermal infrared images have low contrast and diffuse defect boundaries. The preprocessing pipeline applies:
1. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances local temperature contrast without blowing out sensor noise.
2. **Radiometric Min-Max Normalization:** Converts 8-bit or 16-bit raw values to standard floating-point tensors $[0, 1]$.
3. **Spatial Normalization:** Resizes to $224 \times 224$ while preserving thermal aspect ratios.

### Lightweight Transformer (MiT / SegFormer-Style)
```
Input (3 x 224 x 224)
        │
┌───────▼────────────────────────┐
│ Stage 1: Overlap Patch Embed   │  (Embed Dim: 32, Stride: 4)
│          SRA (SR Ratio: 8)     │  MixFFN (Depthwise Conv 3x3)
└───────┬────────────────────────┘
        │
┌───────▼────────────────────────┐
│ Stage 2: Overlap Patch Embed   │  (Embed Dim: 64, Stride: 2)
│          SRA (SR Ratio: 4)     │  MixFFN (Depthwise Conv 3x3)
└───────┬────────────────────────┘
        │
┌───────▼────────────────────────┐
│ Stage 3: Overlap Patch Embed   │  (Embed Dim: 160, Stride: 2)
│          SRA (SR Ratio: 2)     │  MixFFN (Depthwise Conv 3x3)
└───────┬────────────────────────┘
        │
┌───────▼────────────────────────┐
│ Stage 4: Overlap Patch Embed   │  (Embed Dim: 256, Stride: 2)
│          SRA (SR Ratio: 1)     │  MixFFN (Depthwise Conv 3x3)
└───────┬────────────────────────┘
        │
┌───────▼────────────────────────┐
│ Global Pooling & MLP Head      │  --> Output: 12 Class Logits
└────────────────────────────────┘
```
*Total Trainable Parameters:* **~3.8 Million** (Lightweight, real-time edge/UAV deployable).

---

## 4. Quickstart Guide

### 1. Download Dataset
Follow the guide in `data/README.md` or run:
```bash
kaggle datasets download -d marcosgabriel/infrared-solar-modules
Expand-Archive -Path infrared-solar-modules.zip -DestinationPath ./data/infrared_solar_modules
```

### 2. Train the Lightweight Transformer
```bash
python src/train_eval.py --model_type transformer --data_dir ./data/infrared_solar_modules --epochs 25 --batch_size 32
```

### 3. Benchmark against Edge Baseline (MobileNetV3)
```bash
python src/train_eval.py --model_type mobilenet_v3_small --data_dir ./data/infrared_solar_modules --epochs 25
```

---

## 5. Talking Points for Your Meeting with Dr. Mazumder
1. *"The ScienceDirect paper by Joshi & Pal uses SegFormer's lightweight spatial-reduction attention on UAV thermal imagery to segment hotspots and snail trails."*
2. *"The Kaggle dataset contains 20,000 thermal images spanning 11 defect types. This provides 70x more data than the paper's 277 images, allowing us to evaluate lightweight transformers with high statistical rigor."*
3. *"I have scaffolded the complete PyTorch repository with CLAHE contrast enhancement, the SegFormer-inspired transformer model (~3.8M parameters), and benchmark harnesses for MobileNetV3 and ResNet18."*
