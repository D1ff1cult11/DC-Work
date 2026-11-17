import torch
from sfoc import SFOCExtractor, FSFOCExtractor
from cfog import CFOGExtractor
from models.segnext_msca import SegNeXtThermalSegmentor
from models.fsoc_counter import FewShotDefectCounter

print("=" * 60)
print("  TESTING ADVANCED A* NOVELTY MODULES")
print("=" * 60)

device = torch.device("cpu")

# 1. Test SFOC & FSFOC
print("\n[1] Testing SFOC Extractor (Steerable Filters 1st & 2nd Order)...")
sfoc = SFOCExtractor(kernel_size=9, sigma=1.5).to(device)
dummy_img = torch.randn(2, 1, 256, 256)
sfoc_out = sfoc(dummy_img)
print(f" -> Input Shape: {dummy_img.shape}")
print(f" -> SFOC Output Tensor Shape: {sfoc_out.shape} (Channels: Raw, Gradient, Curvature, Orientation)")
assert sfoc_out.shape == (2, 4, 256, 256)

print("\n[1b] Testing Fast-SFOC (FSFOC with Separable Kernels)...")
fsfoc = FSFOCExtractor(kernel_size=7, num_orientations=4).to(device)
fsfoc_out = fsfoc(dummy_img)
print(f" -> FSFOC Output Tensor Shape: {fsfoc_out.shape}")
assert fsfoc_out.shape == (2, 4, 256, 256)
print(" -> FSFOC verified successfully!")

# 1c. Test CFOG
print("\n[1c] Testing CFOG (Channel Features of Oriented Gradients)...")
cfog = CFOGExtractor(num_bins=8).to(device)
cfog_out = cfog(dummy_img)
print(f" -> CFOG Output Tensor Shape: {cfog_out.shape} (8 Orientation Channels)")
assert cfog_out.shape == (2, 8, 256, 256)
print(" -> CFOG verified successfully!")

# 2. Test SegNeXt MSCA
print("\n[2] Testing SegNeXt with Multi-Scale Convolutional Attention (MSCA)...")
segnext = SegNeXtThermalSegmentor(in_channels=4, num_classes=3).to(device)
seg_out = segnext(sfoc_out)
print(f" -> SegNeXt Trainable Parameters: {segnext.count_parameters():,}")
print(f" -> SegNeXt Output Shape: {seg_out.shape} (Classes: Background, Normal Panel, Defect)")
assert seg_out.shape == (2, 3, 256, 256)
print(" -> SegNeXt MSCA verified successfully!")

# 3. Test FSOC Counter
print("\n[3] Testing Few-Shot Object Counting (FSOC) Module...")
fsoc = FewShotDefectCounter(feature_dim=256).to(device)
query_feat = torch.randn(2, 256, 32, 32)
exemplar_feats = torch.randn(2, 3, 256) # 3 support exemplars
density_map, count = fsoc(query_feat, exemplar_feats)
print(f" -> Predicted Density Map Shape: {density_map.shape}")
print(f" -> Predicted Defect Counts: {count.tolist()}")
assert density_map.shape == (2, 1, 32, 32)
print(" -> FSOC module verified successfully!")

print("\n" + "=" * 60)
print("  ALL ADVANCED MODULES FUNCTIONAL AND VALIDATED!")
print("=" * 60)
