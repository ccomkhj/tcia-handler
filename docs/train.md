# Training Data Preparation Guide

This document describes how the multi-modal MRI dataset is structured for deep learning model training, including the metadata format and recommended usage patterns.

## Overview

The dataset is designed for **2.5D multi-modal prostate segmentation** using:
- **T2-weighted MRI** (5 slices for spatial context)
- **ADC** (Apparent Diffusion Coefficient) map (1 slice)
- **Calc** (Calculated DWI) image (1 slice)

**Target labels:**
- `mask_prostate`: Binary segmentation of prostate region
- `mask_target1`: Binary segmentation of target lesion

## Dataset Location

```
data/aligned_v2/
├── class1/
│   ├── case_0144/
│   │   ├── t2/              # T2-weighted PNG images
│   │   ├── adc/             # ADC map PNG images (resampled to T2 grid)
│   │   ├── calc/            # Calc PNG images (resampled to T2 grid)
│   │   ├── mask_prostate/   # Prostate segmentation masks
│   │   ├── mask_target1/    # Target lesion masks
│   │   ├── t2_aligned/      # DICOM files (for archival)
│   │   ├── adc_aligned/     # DICOM files (for archival)
│   │   └── calc_aligned/    # DICOM files (for archival)
│   └── case_XXXX/
├── class2/
├── class3/
├── class4/
└── metadata.json            # Training metadata
```

## Metadata File

The metadata file (`data/aligned_v2/metadata.json`) contains all information needed to construct a PyTorch DataLoader.

### Generate Metadata

```bash
# Generate with default settings (5 T2 context slices)
python tools/generate_training_metadata.py

# Custom T2 context window (must be odd)
python tools/generate_training_metadata.py --t2-context 7

# Specify different data directory
python tools/generate_training_metadata.py --data-dir /path/to/aligned_v2
```

### Metadata Structure

```json
{
  "version": "1.0",
  "created": "2026-01-23T21:18:54.576123",
  "config": {
    "input_size": [256, 256],
    "t2_context_window": 5,
    "boundary_padding": "edge_replicate",
    "modalities": ["t2", "adc", "calc"],
    "masks": ["mask_prostate", "mask_target1"],
    "mask_positive_value": 255
  },
  "global_stats": {
    "t2": {"mean": 55.8, "std": 57.2, "min": 0, "max": 255, "p1": 0.0, "p99": 243.0},
    "adc": {"mean": 57.9, "std": 57.4, "min": 0, "max": 255, "p1": 0.0, "p99": 243.0},
    "calc": {"mean": 52.9, "std": 58.8, "min": 0, "max": 255, "p1": 0.0, "p99": 221.0}
  },
  "cases": { ... },
  "samples": [ ... ],
  "summary": { ... }
}
```

### Key Fields

#### Config
| Field | Description |
|-------|-------------|
| `input_size` | Image dimensions [width, height] |
| `t2_context_window` | Number of T2 slices in context (default: 5) |
| `boundary_padding` | How boundary slices are handled (`edge_replicate`) |
| `mask_positive_value` | Pixel value indicating positive label (255) |

#### Global Stats
Per-modality intensity statistics computed across all samples:
- `mean`, `std`: For normalization
- `p1`, `p99`: 1st and 99th percentiles for robust clipping

#### Cases
Per-case metadata including:
```json
{
  "class1/case_0228": {
    "class": 1,
    "num_slices": 60,
    "has_adc": true,
    "has_calc": true,
    "stats": { "t2": {...}, "adc": {...}, "calc": {...} },
    "slices_with_prostate": [10, 11, 12, ...],
    "slices_with_target": [12, 13, 14, 15, 16]
  }
}
```

#### Samples
Slice-level training samples:
```json
{
  "sample_id": "class1/case_0228/slice_0030",
  "case_id": "class1/case_0228",
  "class": 1,
  "slice_idx": 30,
  "slice_num": 30,
  "files": {
    "t2": "0030.png",
    "adc": "0030.png",
    "calc": "0030.png",
    "mask_prostate": "0030.png",
    "mask_target1": "0030.png"
  },
  "t2_context_indices": [28, 29, 30, 31, 32],
  "has_adc": true,
  "has_calc": true,
  "has_prostate": true,
  "has_target": false
}
```

| Field | Description |
|-------|-------------|
| `sample_id` | Unique identifier for this sample |
| `case_id` | Parent case identifier |
| `class` | PI-RADS class (1-4) |
| `slice_idx` | Zero-based index in the volume |
| `slice_num` | Original slice number from filename |
| `files` | Filename for each modality/mask |
| `t2_context_indices` | Indices for 5 T2 context slices (with edge replication) |
| `has_adc` | Whether ADC data is available for this case |
| `has_calc` | Whether Calc data is available for this case |
| `has_prostate` | Whether this slice contains prostate mask |
| `has_target` | Whether this slice contains target lesion mask |

#### Summary
```json
{
  "total_cases": 194,
  "total_samples": 11362,
  "cases_with_adc": 174,
  "cases_with_calc": 165,
  "cases_complete": 165,
  "samples_with_adc": 10491,
  "samples_with_calc": 9951,
  "samples_complete": 9951,
  "samples_with_prostate": 5678,
  "samples_with_target": 1407,
  "class_distribution": {"1": 17, "2": 58, "3": 60, "4": 59}
}
```

## 2.5D Input Construction

For each training sample, the input tensor is constructed as:

```
Input shape: (7, 256, 256)
├── Channel 0-4: T2 context slices [i-2, i-1, i, i+1, i+2]
├── Channel 5:   ADC slice at index i
└── Channel 6:   Calc slice at index i

Output shape: (2, 256, 256)
├── Channel 0: Prostate mask (binary)
└── Channel 1: Target lesion mask (binary)
```

### Boundary Handling

For slices at volume boundaries, **edge replication** is used:
- Slice 0: `t2_context_indices = [0, 0, 0, 1, 2]`
- Slice 1: `t2_context_indices = [0, 0, 1, 2, 3]`
- Slice N-1: `t2_context_indices = [N-3, N-2, N-1, N-1, N-1]`

## PyTorch Dataset Example

```python
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MultiModalSegmentationDataset(Dataset):
    """
    PyTorch Dataset for 2.5D multi-modal prostate segmentation.
    
    Args:
        metadata_path: Path to metadata.json
        transform: Optional augmentation transform
        require_complete: If True, only include samples with ADC+Calc
        require_positive: If True, only include samples with prostate mask
    """
    
    def __init__(
        self,
        metadata_path: str | Path,
        transform=None,
        require_complete: bool = True,
        require_positive: bool = False,
    ):
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        self.base_dir = Path(metadata_path).parent
        self.transform = transform
        self.global_stats = self.metadata["global_stats"]
        
        # Filter samples based on requirements
        self.samples = []
        for sample in self.metadata["samples"]:
            if require_complete and not (sample["has_adc"] and sample["has_calc"]):
                continue
            if require_positive and not sample["has_prostate"]:
                continue
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        case_dir = self.base_dir / sample["case_id"]
        
        # Load T2 context slices (5 slices)
        t2_slices = []
        for slice_idx in sample["t2_context_indices"]:
            t2_file = f"{slice_idx:04d}.png"
            t2_path = case_dir / "t2" / t2_file
            t2_img = np.array(Image.open(t2_path), dtype=np.float32)
            t2_slices.append(t2_img)
        
        # Load ADC and Calc (handle missing data)
        if sample["has_adc"]:
            adc_path = case_dir / "adc" / sample["files"]["adc"]
            adc = np.array(Image.open(adc_path), dtype=np.float32)
        else:
            adc = np.zeros((256, 256), dtype=np.float32)
        
        if sample["has_calc"]:
            calc_path = case_dir / "calc" / sample["files"]["calc"]
            calc = np.array(Image.open(calc_path), dtype=np.float32)
        else:
            calc = np.zeros((256, 256), dtype=np.float32)
        
        # Stack into input tensor: (7, H, W)
        image = np.stack(t2_slices + [adc, calc], axis=0)
        
        # Load masks: (2, H, W)
        mask_prostate = np.array(
            Image.open(case_dir / "mask_prostate" / sample["files"]["mask_prostate"]),
            dtype=np.float32
        ) / 255.0
        mask_target = np.array(
            Image.open(case_dir / "mask_target1" / sample["files"]["mask_target1"]),
            dtype=np.float32
        ) / 255.0
        mask = np.stack([mask_prostate, mask_target], axis=0)
        
        # Apply transforms (if any)
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Normalize using global stats
        image = self._normalize(image)
        
        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "sample_id": sample["sample_id"],
            "has_adc": sample["has_adc"],
            "has_calc": sample["has_calc"],
        }
    
    def _normalize(self, image):
        """Normalize each channel using global statistics."""
        # T2 channels (0-4)
        t2_mean = self.global_stats["t2"]["mean"]
        t2_std = self.global_stats["t2"]["std"]
        image[:5] = (image[:5] - t2_mean) / (t2_std + 1e-8)
        
        # ADC channel (5)
        adc_mean = self.global_stats["adc"]["mean"]
        adc_std = self.global_stats["adc"]["std"]
        image[5] = (image[5] - adc_mean) / (adc_std + 1e-8)
        
        # Calc channel (6)
        calc_mean = self.global_stats["calc"]["mean"]
        calc_std = self.global_stats["calc"]["std"]
        image[6] = (image[6] - calc_mean) / (calc_std + 1e-8)
        
        return image


# Usage example
if __name__ == "__main__":
    dataset = MultiModalSegmentationDataset(
        metadata_path="data/aligned_v2/metadata.json",
        require_complete=True,
        require_positive=False,
    )
    
    print(f"Total samples: {len(dataset)}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Iterate
    for batch in dataloader:
        images = batch["image"]      # (B, 7, 256, 256)
        masks = batch["mask"]        # (B, 2, 256, 256)
        print(f"Batch - images: {images.shape}, masks: {masks.shape}")
        break
```

## Data Augmentation Recommendations

For medical image segmentation, consider these augmentations:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.ElasticTransform(
            alpha=50,
            sigma=10,
            p=0.3
        ),
        A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
    ])
```

**Note:** Apply spatial transforms to all channels simultaneously to maintain alignment.

## Filtering Strategies

### 1. Complete Data Only (Recommended for initial training)
```python
dataset = MultiModalSegmentationDataset(
    metadata_path="data/aligned_v2/metadata.json",
    require_complete=True,  # Only samples with ADC + Calc
)
# Results in ~9,951 samples
```

### 2. All Samples (Handle missing data in forward pass)
```python
dataset = MultiModalSegmentationDataset(
    metadata_path="data/aligned_v2/metadata.json",
    require_complete=False,  # Include all samples
)
# Results in ~11,362 samples
# Missing ADC/Calc filled with zeros
```

### 3. Positive Samples Only (For class imbalance)
```python
dataset = MultiModalSegmentationDataset(
    metadata_path="data/aligned_v2/metadata.json",
    require_positive=True,  # Only slices with prostate mask
)
# Results in ~5,678 samples
```

## Train/Val/Test Split

The metadata does not include split assignments. Recommended approach:

```python
from sklearn.model_selection import train_test_split

# Load metadata
with open("data/aligned_v2/metadata.json") as f:
    metadata = json.load(f)

# Get unique case IDs
case_ids = list(metadata["cases"].keys())

# Split at case level (not slice level) to prevent data leakage
train_cases, temp_cases = train_test_split(case_ids, test_size=0.3, random_state=42)
val_cases, test_cases = train_test_split(temp_cases, test_size=0.5, random_state=42)

# Filter samples by case
train_samples = [s for s in metadata["samples"] if s["case_id"] in train_cases]
val_samples = [s for s in metadata["samples"] if s["case_id"] in val_cases]
test_samples = [s for s in metadata["samples"] if s["case_id"] in test_cases]
```

## Model Architecture Considerations

### Input Channels
- **7 channels**: 5 T2 (context) + 1 ADC + 1 Calc

### Output Channels
- **2 channels**: Prostate mask + Target lesion mask
- Use **Sigmoid activation** for multi-label output
- Use **BCEWithLogitsLoss** or **Dice Loss**

### Recommended Architectures
- **U-Net** with ResNet/EfficientNet encoder
- **nnU-Net** (automatic configuration)
- **Attention U-Net** for improved boundary detection

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total cases | 194 |
| Total samples (slices) | 11,362 |
| Cases with complete data | 165 (85%) |
| Samples with complete data | 9,951 (88%) |
| Samples with prostate | 5,678 (50%) |
| Samples with target | 1,407 (12%) |
| Image size | 256 × 256 |
| Pixel format | uint8 (0-255) |

### Class Distribution
| Class | Cases | Description |
|-------|-------|-------------|
| 1 | 17 | PI-RADS 1 |
| 2 | 58 | PI-RADS 2 |
| 3 | 60 | PI-RADS 3 |
| 4 | 59 | PI-RADS 4 |

## Troubleshooting

### Missing ADC/Calc Data
Some cases are missing ADC and/or Calc data. Check `has_adc` and `has_calc` flags:
```python
# Count missing data
missing_adc = sum(1 for s in metadata["samples"] if not s["has_adc"])
missing_calc = sum(1 for s in metadata["samples"] if not s["has_calc"])
print(f"Missing ADC: {missing_adc}, Missing Calc: {missing_calc}")
```

### Regenerating Metadata
If the dataset changes, regenerate metadata:
```bash
python tools/generate_training_metadata.py --data-dir data/aligned_v2
```

### Memory Issues
For large datasets, use:
- `num_workers > 0` in DataLoader
- `pin_memory=True` for GPU training
- Consider pre-loading to HDF5 for faster I/O

## Related Documentation

- [DICOM Mapping](dicom_mapping.md) - How spatial alignment works
- [README](../README.md) - Project overview and data directories
