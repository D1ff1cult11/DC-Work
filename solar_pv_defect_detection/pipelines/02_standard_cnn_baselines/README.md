# Pipeline 2: Standard CNN Baselines (Pre-SFOC Benchmark)

> **Purpose:** Establishes the comparative baseline against classical convolutional neural networks commonly evaluated in the 2020–2024 literature.

---

## Benchmarked Architectures
1. **MobileNetV3-Small:** Lightweight industrial standard for UAV edge deployment (~2.5M parameters).
2. **ResNet-18:** Canonical deep residual baseline (~11.2M parameters).

## How to Run
```bash
# Train MobileNetV3-Small baseline
python pipelines/02_standard_cnn_baselines/train.py --arch mobilenet_v3_small --epochs 20

# Train ResNet-18 baseline
python pipelines/02_standard_cnn_baselines/train.py --arch resnet18 --epochs 20
```
