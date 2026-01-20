# DICOM Multi-Modal Mapping Guide

## TL;DR

**Problem**: T2, ADC, and Calc MRI sequences have different resolutions and spatial positions.

**Solution**: Use SimpleITK to resample ADC/Calc to T2's grid using DICOM spatial metadata.

```
T2:  256×256×60  @ 0.664mm spacing  ← Reference
ADC: 132×160×20  @ 1.625mm spacing  → Resampled to 256×256×60
```

**Key Insight**: Both images share the same world coordinate system (LPS mm). The spatial metadata (origin, spacing, direction from DICOM) enables coordinate-based resampling—NOT simple pixel resizing.

---

## Quick Start

### Commands

```bash
# Align T2 + ADC + Calc + Masks
uv run dicom-mapper process --input-dir data --output-dir data/aligned_v2

# Process single case
uv run dicom-mapper process --input-dir data --output-dir data/aligned_v2 --class-num 1 --case-id 0144

# Visualize alignment
uv run dicom-mapper visualize --aligned-dir data/aligned_v2 --output-dir data/visualizations_v2

# Multi-modal side-by-side visualization (T2 + ADC + masks)
python tools/preprocessing/visualize_overlay_masks.py --multimodal
```

### Output Structure

```
data/aligned_v2/class{N}/case_{XXXX}/
├── t2/           # 60 aligned PNG slices
├── adc/          # 60 aligned PNG slices (resampled to T2)
├── calc/         # 60 aligned PNG slices (resampled to T2)
├── mask_prostate/ # 60 mask PNG slices (padded)
└── mask_target1/  # 60 mask PNG slices (padded)
```

---

## Core Concepts

### 1. Sequence Linking

T2 and ADC are linked by **`case_id`** AND **`StudyInstanceUID`**:
- Same `case_id` folder: `data/processed/class1/case_0144/` ↔ `data/processed_ep2d_adc/class1/case_0144/`
- **CRITICAL**: Must also match `StudyInstanceUID` from `meta.json`

#### Multi-Series Per Case

Some cases have **multiple imaging studies** (different `StudyInstanceUID`), each with its own T2/ADC/Calc:

```
case_0044/
├── 1.3.6.1.4.1...229... (StudyUID: 236806...)  ← Study A
└── 1.3.6.1.4.1...572... (StudyUID: 260314...)  ← Study B
```

**⚠️ Pitfall**: Alphabetical ordering may pick ADC from Study A while T2 is from Study B. Their z-coordinates won't align!

**Solution**: Match by `StudyInstanceUID`:

```python
def select_matching_series(series_dirs, t2_study_uid):
    """Select series with matching StudyInstanceUID."""
    for series_dir in series_dirs:
        meta = json.loads((series_dir / "meta.json").read_text())
        if meta.get("StudyInstanceUID") == t2_study_uid:
            return series_dir
    return series_dirs[0]  # Fallback
```

📁 **Implementation**: `tools/preprocessing/visualize_overlay_masks.py` line ~1032

### 2. Spatial Alignment

**Why simple resize doesn't work**: T2 and ADC have different:
- Origins (where the image starts in space)
- Spacings (pixel/voxel size in mm)
- Fields of view (z-range coverage)

**Solution**: Use DICOM spatial metadata to transform through world coordinates:

```
Voxel (i,j,k) → World (x,y,z) mm → Voxel (i',j',k')
```

#### Z-Position Mapping

T2 and ADC/Calc may have **different z-coverage**:

```
T2:   z-range [-27.79, 60.71]  (60 slices)
ADC:  z-range [-17.74, 50.66]  (20 slices)
                ↑
         Overlap: [-17.74, 50.66]
```

**Mapping algorithm**:
1. For each T2 slice, find nearest ADC slice by z-position
2. Only create mapping if distance < `slice_thickness` (~3.6mm)
3. T2 slices outside ADC z-range show "Not Available"

```python
# Z-position mapping
for t2_idx, t2_z in enumerate(t2_z_positions):
    distances = np.abs(adc_z_arr - t2_z)
    nearest_idx = int(np.argmin(distances))
    if distances[nearest_idx] < adc_thickness:
        mapping[t2_idx] = nearest_idx  # Valid mapping
    # else: no ADC for this T2 slice
```

### 3. The Resampling Process

```python
# Core code in dicom_mapper/processing/resampling.py
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(t2_volume)    # T2 defines output grid
resampler.SetTransform(sitk.Transform())  # Identity - metadata handles mapping
result = resampler.Execute(adc_volume)    # ADC now matches T2 dimensions
```

### 4. Key DICOM Tags

| Tag | Purpose |
|-----|---------|
| \`PixelSpacing\` | In-plane resolution (row, col) in mm |
| \`SliceThickness\` | Z-axis spacing in mm |
| \`ImagePositionPatient\` | Origin (x, y, z) of first voxel |
| \`ImageOrientationPatient\` | Direction cosines (6 values → 3×3 matrix) |

---

## Debugging Breakpoints

| Priority | File | Line | What to Check |
|----------|------|------|---------------|
| 🔴 | \`pipeline.py\` | **248** | \`adc_resampled.GetSize()\` should match T2 |
| 🔴 | \`resampling.py\` | **63** | Core resampling - compare sizes |
| 🟡 | \`pipeline.py\` | **154** | DICOM spacing/origin extraction |

```python
# Quick debug commands
img.GetSize()      # (X, Y, Z) dimensions
img.GetSpacing()   # (Δx, Δy, Δz) in mm
img.GetOrigin()    # First voxel position
```

---

# Detailed Documentation

Everything below provides in-depth explanations, diagrams, and implementation details.

---

## Data Structure Overview

### Processed Directories

| Directory | Content | Description |
|-----------|---------|-------------|
| \`data/processed/\` | T2-weighted images | High-resolution anatomical reference |
| \`data/processed_ep2d_adc/\` | ADC maps | Apparent Diffusion Coefficient from DWI |
| \`data/processed_ep2d_calc/\` | Calculated DWI | Derived diffusion images |
| \`data/processed_seg/\` | Segmentation masks | Prostate & target ROI masks |

### Directory Structure

```
data/processed/class{N}/case_{XXXX}/{SeriesInstanceUID}/
├── images/
│   ├── 0000.png
│   └── ...
└── meta.json

data/processed_seg/class{N}/case_{XXXX}/{SeriesInstanceUID}/
├── prostate/
│   └── 0010.png ...
├── target1/
│   └── 0014.png ...
└── biopsies.json
```

---

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

---

## Linking Keys

### StudyInstanceUID (Primary Link)
All sequences from the same imaging session share the same \`StudyInstanceUID\`.

### Coordinate System (LPS)
DICOM uses the **LPS** (Left-Posterior-Superior) coordinate system:
- **L**: Patient's left (positive X direction)
- **P**: Patient's posterior (positive Y direction)  
- **S**: Patient's superior/head (positive Z direction)

### IJK to World Transformation

```
[x]   [Xx*Δi  Yx*Δj  Zx*Δk  Sx]   [i]
[y] = [Xy*Δi  Yy*Δj  Zy*Δk  Sy] × [j]
[z]   [Xz*Δi  Yz*Δj  Zz*Δk  Sz]   [k]
[1]   [0      0      0      1 ]   [1]
```

---

## Implementation Details

### Why Per-Slice DICOMs?
\`highdicom.sc.SCImage\` only supports 2D grayscale arrays. 3D arrays are interpreted as RGB.

**Solution**: Split into 2D slices, one SCImage per slice, sharing same SeriesInstanceUID.

### Mask Handling
Masks in \`processed_seg/\` only cover slices with segmentation (e.g., slices 21-49).
Pipeline pads to full T2 dimensions with zeros.

### Key Dependencies
- **highdicom**: DICOM Secondary Capture images
- **SimpleITK**: Spatial resampling
- **pydicom**: DICOM metadata

---

## Troubleshooting

### Common Issues

1. **Missing ADC/Calc**: Not all cases have diffusion sequences
2. **Misaligned masks**: Use full spatial transformation, not simple resize
3. **SCImage error**: Split 3D into 2D slices
4. **Masks not visible**: Check padding and filename alignment
5. **ADC matches but Calc doesn't** (or vice versa):
   - **Cause**: Multiple series per case with different `StudyInstanceUID`
   - Alphabetical selection may pick series from wrong study
   - **Fix**: Match by `StudyInstanceUID` (see Core Concepts §1)
6. **ADC/Calc shows "Not Available" for some T2 slices**:
   - **Cause**: T2 z-range extends beyond ADC/Calc coverage
   - This is **correct behavior** - no ADC data exists for those slices
   - Check z-ranges: T2 may have 60 slices, ADC only covers middle 40

### Validation Checklist
- [ ] `StudyInstanceUID` matches across T2, ADC, Calc
- [ ] Resampled ADC/Calc match T2 dimensions
- [ ] Mask indices align with T2 slice indices
- [ ] Masks visually align with anatomy
- [ ] Z-overlap region has valid mappings

---

## Code Walkthrough with Debugging Breakpoints

### Part 1: Sequence Mapping

📁 **File:** \`dicom_mapper/cli/pipeline.py\`

```python
# Line 225-227: Load T2 as REFERENCE
t2_volume, t2_datasets = load_modality_volume(
    case_id, class_num, root_dir, "processed", "nbia", resampler
)

# Line 242-244: Load ADC for SAME case_id
adc_volume, adc_datasets = load_modality_volume(
    case_id, class_num, root_dir, "processed_ep2d_adc", "nbia_ep2d_adc", resampler
)
```

### Part 2: The Core Resampling

📁 **File:** \`dicom_mapper/processing/resampling.py\`

```python
# Line 63-68: THE KEY FUNCTION
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(reference)      # T2 defines output grid
resampler.SetInterpolator(interpolator)
resampler.SetDefaultPixelValue(0)
resampler.SetTransform(sitk.Transform())    # Identity - metadata handles mapping!

return resampler.Execute(moving)  # Returns ADC on T2's grid
```

### Part 3: What Happens Inside

```
For each output voxel (i, j, k) in T2 grid:
    1. Compute world coords: (x, y, z) = T2_matrix × (i, j, k, 1)
    2. Find ADC voxel: (i', j', k') = ADC_matrix⁻¹ × (x, y, z, 1)
    3. Interpolate ADC value at (i', j', k')
    4. Store in output[i, j, k]
```

### Debug Session Checklist

| Step | File | Line | What to Check |
|------|------|------|---------------|
| 1 | \`pipeline.py\` | 225 | T2 volume loaded |
| 2 | \`pipeline.py\` | 242 | ADC volume loaded |
| 3 | \`pipeline.py\` | 154 | Spacing/origin from DICOM |
| 4 | \`resampling.py\` | 63 | Core resampling |
| 5 | \`pipeline.py\` | 248 | ADC matches T2 dimensions |

### Quick Debug Commands

```python
img.GetSize()      # (X, Y, Z) dimensions
img.GetSpacing()   # (Δx, Δy, Δz) in mm
img.GetOrigin()    # First voxel position
img.GetDirection() # 9-element direction matrix

arr = sitk.GetArrayFromImage(img)
arr.shape  # (Z, Y, X) - note axis order!
```

---

## Lessons from 3D Slicer

1. **Full IJK-to-World Matrix**: 4×4 transformation for voxel → physical coordinates
2. **Coordinate System Consistency**: LPS for DICOM/ITK, RAS for Slicer
3. **Reference-Based Resampling**: One image as reference, others resampled to match
4. **Direction Cosines Matter**: ImageOrientationPatient essential for alignment
5. **Nearest Neighbor for Masks**: Preserve binary values during resampling
