# Pipeline 1: Paper Reproduction (SegFormer on Thermal PV Imagery)

> **Reference Paper:** *Lightweight transformer driven segmentation of defects in solar PV thermal imagery*  
> **Authors:** Deepak Joshi, Mayukha Pal  
> **Journal:** *Solar Energy* (Elsevier, Vol 308, 2026, Article 114402; arXiv:2507.20680)

---

## Overview & Architecture
This pipeline reproduces the exact model architecture and training setup described in the ScienceDirect paper:
1. **Backbone:** Hierarchical Mix Transformer (MiT-B0 style).
2. **Key Mechanism:** Spatial-Reduction Attention (SRA) with downsampling ratios $R \in [8, 4, 2, 1]$ to reduce self-attention from $\mathcal{O}(N^2)$ to $\mathcal{O}(N^2/R)$.
3. **Mix-FFN:** Replaces traditional positional encodings with $3 \times 3$ depthwise convolutions inside feed-forward blocks, preserving translational invariance.
4. **Decoder:** Lightweight all-MLP head concatenating multi-scale features from 4 encoder stages.
5. **Preprocessing:** Contrast-Limited Adaptive Histogram Equalization (CLAHE) + min-max normalization.

## How to Run
```bash
# Train the paper's model on the dataset
python pipelines/01_paper_reproduction_segformer/train.py --epochs 25 --batch_size 32

# Evaluate best checkpoint on test set
python pipelines/01_paper_reproduction_segformer/evaluate.py
```
