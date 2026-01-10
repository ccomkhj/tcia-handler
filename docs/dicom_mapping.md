# DICOM Multi-Modal Mapping Guide

This document describes how to map and align different MRI sequences (T2, ADC, Calc) with segmentation masks for deep learning training.

## Data Structure Overview

### Processed Directories

| Directory | Content | Description |
|-----------|---------|-------------|
| `data/processed/` | T2-weighted images | High-resolution anatomical reference |
| `data/processed_ep2d_adc/` | ADC maps | Apparent Diffusion Coefficient from DWI |
| `data/processed_ep2d_calc/` | Calculated DWI | Derived diffusion images |
| `data/processed_seg/` | Segmentation masks | Prostate & target ROI masks |

### Directory Structure

```
data/processed/class{N}/case_{XXXX}/{SeriesInstanceUID}/
├── images/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
└── meta.json

data/processed_seg/class{N}/case_{XXXX}/{SeriesInstanceUID}/
├── prostate/
│   ├── 0010.png
│   └── ...
├── target1/
│   ├── 0014.png
│   └── ...
└── biopsies.json
```

## DICOM Spatial Properties

### T2-Weighted Series
- **Pixel Spacing**: 0.664mm × 0.664mm
- **Slice Thickness**: 1.5mm
- **Image Size**: 256 × 256
- **Typical Slice Count**: 50-60

### ADC / Calc Series
- **Pixel Spacing**: 1.625mm × 1.625mm
- **Slice Thickness**: 3.6mm
- **Image Size**: 132 × 160
- **Typical Slice Count**: 20

### Key Observations
1. **Resolution Ratio**: T2 is ~2.5× higher resolution than ADC/Calc
2. **Z-Spacing Ratio**: T2 has ~2.4× more slices (1.5mm vs 3.6mm spacing)
3. **Different FOV**: T2 and ADC/Calc may start at different z-positions

## Linking Keys

### StudyInstanceUID (Primary Link)
All sequences from the same imaging session share the same `StudyInstanceUID`:

```python
# Example for case_0001
T2:   StudyInstanceUID = "1.3.6.1.4.1.14519.5.2.1.85548304921965658367726869399297351743"
ADC:  StudyInstanceUID = "1.3.6.1.4.1.14519.5.2.1.85548304921965658367726869399297351743"  # Same!
Calc: StudyInstanceUID = "1.3.6.1.4.1.14519.5.2.1.85548304921965658367726869399297351743"  # Same!
```

### SeriesInstanceUID
Each sequence has a unique `SeriesInstanceUID`. Masks are aligned to T2 and share its SeriesInstanceUID.

### Spatial Coordinates
DICOM headers provide exact spatial positioning:
- `ImagePositionPatient`: (x, y, z) coordinates of first voxel in LPS
- `SliceLocation`: z-coordinate for each slice
- `ImageOrientationPatient`: Direction cosines for row/column (6 values)
- `PixelSpacing`: In-plane resolution [row_spacing, col_spacing]

### Coordinate System (LPS)
DICOM uses the **LPS** (Left-Posterior-Superior) coordinate system:
- **L**: Patient's left (positive X direction)
- **P**: Patient's posterior (positive Y direction)  
- **S**: Patient's superior/head (positive Z direction)

This is the same convention used by ITK/SimpleITK. 3D Slicer uses RAS internally but converts when reading/writing DICOM.

### IJK to World Transformation
The transformation from image indices (i, j, k) to world coordinates (x, y, z) is:

```
[x]   [Xx*Δi  Yx*Δj  Zx*Δk  Sx]   [i]
[y] = [Xy*Δi  Yy*Δj  Zy*Δk  Sy] × [j]
[z]   [Xz*Δi  Yz*Δj  Zz*Δk  Sz]   [k]
[1]   [0      0      0      1 ]   [1]
```

Where:
- `(Xx, Xy, Xz)` = row direction cosines (first 3 values of ImageOrientationPatient)
- `(Yx, Yy, Yz)` = column direction cosines (last 3 values of ImageOrientationPatient)
- `(Zx, Zy, Zz)` = slice direction (cross product of row × column)
- `(Δi, Δj, Δk)` = spacing (PixelSpacing[1], PixelSpacing[0], SliceThickness)
- `(Sx, Sy, Sz)` = origin (ImagePositionPatient)

## Slice Mapping Strategy

### Step 1: Extract Spatial Metadata
During DICOM conversion, extract per-slice z-positions:

```python
# From DICOM header
ds = pydicom.dcmread(dcm_file)
z_position = float(ds.ImagePositionPatient[2])  # or ds.SliceLocation
```

### Step 2: Build Z-Position Index
Create a lookup table for each series:

```python
t2_z_positions = [-74.88, -73.38, -71.88, ...]  # 60 values
adc_z_positions = [-64.83, -61.23, -57.63, ...]  # 20 values
```

### Step 3: Find Corresponding Slices
For each ADC/Calc slice, find the nearest T2 slice(s):

```python
def find_nearest_slice(target_z, reference_z_positions):
    """Find index of nearest slice by z-position."""
    distances = [abs(z - target_z) for z in reference_z_positions]
    return np.argmin(distances)

# Example: ADC slice at z=-61.23mm → T2 slice at z=-61.38mm (slice 9)
```

### Step 4: Resample to Common Grid
Use SimpleITK to resample ADC/Calc to T2 spatial grid. The key insight from 3D Slicer is that **proper resampling requires the full spatial metadata** (origin, spacing, AND direction):

```python
import SimpleITK as sitk
import numpy as np

def build_direction_matrix(orientation):
    """Build 3x3 direction matrix from ImageOrientationPatient (6 values)."""
    row_dir = np.array(orientation[0:3])  # Direction along columns (I axis)
    col_dir = np.array(orientation[3:6])  # Direction along rows (J axis)
    slice_dir = np.cross(row_dir, col_dir)  # K axis (normal to image plane)
    slice_dir = slice_dir / np.linalg.norm(slice_dir)
    
    # Return as flat 9-element array (row-major)
    return [
        row_dir[0], col_dir[0], slice_dir[0],
        row_dir[1], col_dir[1], slice_dir[1],
        row_dir[2], col_dir[2], slice_dir[2],
    ]

def resample_to_reference(moving_img, reference_img, interpolator=sitk.sitkLinear):
    """Resample moving image to reference geometry.
    
    Both images must have proper spatial metadata set:
    - Origin (from ImagePositionPatient)
    - Spacing (from PixelSpacing + SliceThickness)
    - Direction (from ImageOrientationPatient)
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())  # Identity - spatial metadata handles mapping
    return resampler.Execute(moving_img)
```

**Critical**: For mask resampling, use `sitk.sitkNearestNeighbor` to preserve binary values.

## Multi-Channel Training Setup

### Target Output Structure
After mapping and resampling:

```
data/aligned/class{N}/case_{XXXX}/{StudyInstanceUID}/
├── t2/
│   ├── 0000.png ... 0059.png
├── adc/
│   ├── 0000.png ... 0059.png  (resampled to T2 grid)
├── calc/
│   ├── 0000.png ... 0059.png  (resampled to T2 grid)
├── mask_prostate/
│   ├── 0000.png ... 0059.png
├── mask_target/
│   ├── 0000.png ... 0059.png
└── mapping.json
```

### Mapping JSON Format
```json
{
  "study_uid": "1.3.6.1.4.1.14519...",
  "case_id": "0001",
  "class": 2,
  "t2_series_uid": "1.3.6.1.4.1.14519...",
  "adc_series_uid": "1.3.6.1.4.1.14519...",
  "calc_series_uid": "1.3.6.1.4.1.14519...",
  "num_slices": 60,
  "slice_mapping": {
    "0": {"t2_z": -74.88, "adc_z": -64.83, "adc_idx": 0},
    "1": {"t2_z": -73.38, "adc_z": -64.83, "adc_idx": 0},
    ...
  },
  "resampled": true,
  "reference_spacing": [0.664, 0.664, 1.5]
}
```

## Training Data Loader Example

```python
import numpy as np
from PIL import Image

def load_multichannel_sample(aligned_dir, slice_idx):
    """Load T2 + ADC + Calc as 3-channel input."""
    t2 = np.array(Image.open(aligned_dir / "t2" / f"{slice_idx:04d}.png"))
    adc = np.array(Image.open(aligned_dir / "adc" / f"{slice_idx:04d}.png"))
    calc = np.array(Image.open(aligned_dir / "calc" / f"{slice_idx:04d}.png"))
    
    # Stack as 3-channel input
    image = np.stack([t2, adc, calc], axis=0)  # (3, H, W)
    
    # Load mask
    mask = np.array(Image.open(aligned_dir / "mask_prostate" / f"{slice_idx:04d}.png"))
    
    return image, mask
```

## Pipeline Commands

```bash
# Step 1: Run DICOM conversion (extracts spatial metadata)
python service/preprocess.py --step dicom_to_png

# Step 2: Generate multi-modal mapping
python service/mapping.py --all

# Step 3: Validate alignment
python service/mapping.py --validate

# Step 4: Visualize alignment
python tools/preprocessing/visualize_overlay_masks.py
```

## Troubleshooting

### Common Issues

1. **Missing ADC/Calc for some cases**
   - Not all cases have diffusion sequences
   - Check `data/nbia_ep2d_adc/` exists for the case

2. **Slice count mismatch after resampling**
   - ADC/Calc may not cover entire T2 volume
   - Missing slices filled with zeros or interpolated

3. **Misaligned masks on ADC/Calc**
   - Masks are created from T2-referenced STL overlays
   - **Root cause**: Different FOV and origin between T2 and ADC/Calc
   - **Solution**: Use full spatial transformation (origin + spacing + direction)
   - Simple resize is NOT sufficient - must use coordinate-based resampling

### Validation Checklist
- [ ] StudyInstanceUID matches across all sequences
- [ ] Resampled ADC/Calc have same dimensions as T2
- [ ] Mask indices align with T2 slice indices
- [ ] No empty channels in multi-modal stack
- [ ] Masks visually align with anatomy on ADC/Calc (not just T2)

## Lessons from 3D Slicer

The 3D Slicer approach to multi-modal alignment uses these key principles:

1. **Full IJK-to-World Matrix**: Every image has a complete 4×4 transformation matrix that maps voxel indices to physical (LPS) coordinates.

2. **Coordinate System Consistency**: All operations are performed in a common coordinate system (LPS for DICOM/ITK, RAS for Slicer internal).

3. **Reference-Based Resampling**: When overlaying or combining images, one image serves as the reference geometry, and others are resampled to match.

4. **Direction Cosines Matter**: The `ImageOrientationPatient` tag is essential for correct spatial alignment, especially when images have different orientations or oblique acquisitions.

5. **Nearest Neighbor for Masks**: Binary masks should always use nearest-neighbor interpolation during resampling to preserve discrete values.

See `Slicer/Libs/vtkSegmentationCore/vtkOrientedImageDataResample.cxx` and `Slicer/Base/Python/sitkUtils.py` for implementation details.

