# RSNA Knee MRI Model

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition%20Deployment-20BEFF.svg?logo=kaggle)](https://www.kaggle.com)

## Purpose
End-to-end deep learning inference model for **12-label multi-label knee MRI abnormality prediction** in the RSNA Knee Abnormality Detection challenge.

### 12 Target Abnormalities
1. `ACL` (Anterior Cruciate Ligament Tear)
2. `MCL` (Medial Collateral Ligament Tear)
3. `Medial Meniscus` (Medial Meniscus Tear)
4. `Lateral Meniscus` (Lateral Meniscus Tear)
5. `Medial OA` (Medial Compartment Osteoarthritis)
6. `Lateral OA` (Lateral Compartment Osteoarthritis)
7. `PF OA` (Patellofemoral Osteoarthritis)
8. `Effusion` (Joint Effusion)
9. `Synovitis` (Synovial Thickening / Synovitis)
10. `Baker's` (Baker's Cyst)
11. `Contusion` (Bone Contusion / Edema)
12. `Fracture` (Occult / Manifest Fracture)

---

## Model Architecture & Specifications

The model implements a hierarchical multi-sequence 2D-slice Multiple Instance Learning (MIL) pipeline designed specifically for clinical MRI volumetric protocols:

```
MRI Slices (DCM) ──> 3D Normal Slice Ordering ──> Percentile Norm [1.0%, 99.0%]
                                                               │
                                                               ▼
                                                  2D ResNet-18 Slice Encoder
                                                               │
                                                               ▼
                                                 Slice-Level Feature Vectors
                                                               │
                                                               ▼
                                                  Intra-Series Mean-Max Pool
                                                               │
Series Metadata (Fluid, Fat, Plane) ──> Embedder (10 -> 64) ──>│
                                                               │
                                                               ▼
                                                  Inter-Series Mean-Max Pool
                                                               │
                                                               ▼
                                                  Study Head (12-Label Logits)
```

- **Total Parameters**: `12,656,716`
- **Trainable Parameters**: `12,656,716`
- **Backbone**: ResNet-18 feature extractor modified for single-channel grayscale MRI (`1 x 224 x 224`)
- **Metadata Fusion**: 10-dimensional series metadata embedded into a 64-dimensional dense representation and fused via LayerNorm + GELU
- **Sequence Aggregation**: Mean-Max dual pooling over slice and series dimensions with zero-padding mask awareness
- **Model Checkpoint**: `best_model.pth` (~152 MB, verified floating-point trained weights)

---

## Preprocessing & Volumetric Handling

1. **3D Spatial Geometry Slice Ordering**:
   - Computes acquisition plane normal vector: $\vec{n} = \vec{u}_{\text{IOP}[0:3]} \times \vec{v}_{\text{IOP}[3:6]}$
   - Projects patient coordinates onto the normal: $\text{Position} = \vec{p}_{\text{IPP}} \cdot \vec{n}$
   - Fallback hierarchy: `SliceLocation` $\to$ `InstanceNumber` $\to$ Natural alphanumeric filename sorting.
2. **Slice Sampling**:
   - Uniform linear spacing sampling of **16 slices per series**.
3. **Intensity Normalization**:
   - Percentile-based clipping between the 1.0th and 99.0th percentiles mapped to $[0.0, 1.0]$.
4. **Series Handling**:
   - Dynamically loads up to **6 series per study** (Axial, Coronal, Sagittal $\times$ Fluid-Sensitive/Non-Fluid).
   - Missing series are zero-padded and masked with `series_mask = 0.0`.

---

## Calibrated Decision Thresholds

Optimal class-specific decision thresholds selected via post-training validation optimization:

| Abnormality Target | Optimal Threshold | Default Baseline |
| :--- | :---: | :---: |
| **ACL** | `0.05` | 0.50 |
| **MCL** | `0.50` | 0.50 |
| **Medial Meniscus** | `0.50` | 0.50 |
| **Lateral Meniscus** | `0.50` | 0.50 |
| **Medial OA** | `0.50` | 0.50 |
| **Lateral OA** | `0.50` | 0.50 |
| **PF OA** | `0.50` | 0.50 |
| **Effusion** | `0.50` | 0.50 |
| **Synovitis** | `0.50` | 0.50 |
| **Baker's** | `0.50` | 0.50 |
| **Contusion** | `0.50` | 0.50 |
| **Fracture** | `0.50` | 0.50 |

---

## Offline & Zero-Internet Compliance

- **No Online Downloads**: Model architectures initialize with `weights=None` and load directly from local `best_model.pth`.
- **Hardware Acceleration**: Automatic CUDA acceleration on Kaggle Tesla T4 / P100 / RTX GPUs with fallback to CPU.
- **Self-Contained**: Zero external API dependencies or hardcoded paths.

---

## Kaggle Notebook Usage

### Single Study Inference

```python
import sys
sys.path.append("/kaggle/input/rsna-knee-mri-model")

from rsna_knee_kaggle_model.inference import KneeMRIInferenceEngine

# Initialize engine (automatically selects CUDA GPU if available)
engine = KneeMRIInferenceEngine(model_dir="/kaggle/input/rsna-knee-mri-model")

# Predict on a test study directory
study_dir = "/kaggle/input/rsna-knee-abnormality-detection/test/test_study_0000"
result = engine.predict_study(study_dir)

print(f"Study UID: {result['StudyInstanceUID']}")
for label, prob in result["probabilities"].items():
    pred_bin = result["binary_predictions"][label]
    print(f"  {label:<20}: Prob={prob:.4f} | Binary={pred_bin}")
```

### Full Competition Batch Submission

```python
import glob
import pandas as pd
from rsna_knee_kaggle_model.inference import KneeMRIInferenceEngine

engine = KneeMRIInferenceEngine(model_dir="/kaggle/input/rsna-knee-mri-model")

# Discover all test studies and load metadata
test_study_dirs = sorted(glob.glob("/kaggle/input/rsna-knee-abnormality-detection/test/*"))
test_series_df = pd.read_csv("/kaggle/input/rsna-knee-abnormality-detection/test_series.csv")

# Generate submission dataframe
submission_df = engine.predict_batch_studies(test_study_dirs, series_metadata_df=test_series_df)
submission_df.to_csv("submission.csv", index=False)
print("Submission saved successfully:", submission_df.shape)
```

---

## Repository Structure

```
rsna-knee-mri-model/
├── model.py              # Hierarchical multi-series 2D CNN architecture
├── inference.py          # Production KneeMRIInferenceEngine
├── dicom_loader.py       # 3D spatial geometry slice ordering & loading
├── preprocessing.py      # Percentile normalization & volume collation
├── config.json           # Inference & architecture configuration
├── thresholds.json       # Calibrated class-specific thresholds
├── best_model.pth        # Verified trained weights (12,656,716 params)
├── requirements.txt      # Minimal dependencies
├── LICENSE               # MIT License
└── README.md             # Model documentation
```

---

## License

This repository is licensed under the [MIT License](LICENSE).
