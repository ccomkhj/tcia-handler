# TCIA Handler & DICOM Mapper

Tools for TCIA manifest generation, data preprocessing, and **Standardized Multi-Modal MRI Mapping** (T2, ADC, Calc, Seg).

This repo is part of [MRI_2.5D](https://github.com/ccomkhj/MRI_2.5D_Segmentation).

## 🚀 New: `dicom_mapper` Package
This project has been refactored to use `uv` and `highdicom` for robust, standard-compliant DICOM processing.

### Setup
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 🛠️ Usage: Mapping & Alignment
Use the new CLI to align sequences, generate Secondary Capture DICOM series for aligned image volumes, and export training-ready PNGs plus mask PNGs.

```bash
# Process all cases (Align -> DICOM -> PNG)
uv run dicom-mapper process --input-dir data --output-dir data/aligned_v2

# Process specific class
uv run dicom-mapper process --input-dir data --output-dir data/aligned_v2 --class-num 2

# Process single case
uv run dicom-mapper process --input-dir data --output-dir data/aligned_v2 --case-id 0001
```

### 👁️ Usage: Visualization
Verify alignment by overlaying segmentation masks on T2, ADC, and Calc images.

```bash
# Visualize aligned output
uv run dicom-mapper visualize --aligned-dir data/aligned_v2 --output-dir data/visualizations_v2
```

---

## Legacy Scripts (Data Prep)
The following scripts handle data preparation around the external MRI dataset. DICOM download itself remains a manual NBIA Data Retriever step.

```bash
# Run the automated preprocessing pipeline
python service/preprocess.py --all

# Individual preprocessing steps
python service/preprocess.py --step excel_to_parquet
python service/preprocess.py --step generate_ignore_list
python service/preprocess.py --step merge_datasets
python service/preprocess.py --step generate_tcia
python service/preprocess.py --step dicom_to_png
python service/preprocess.py --step process_overlays
python service/preprocess.py --step validate_2d5

# Point preprocessing at an external MRI workspace if needed
python service/preprocess.py --all --mri-root /path/to/mri
```

`python service/preprocess.py --all` skips the TCIA manifest step by default because it requires manual NBIA follow-up. Run `--step generate_tcia` explicitly when you need fresh `.tcia` files.

**Note:** `service/mapping.py` is deprecated. Prefer `uv run dicom-mapper`.

## Project Structure

```
tcia-handler/
├── dicom_mapper/          # NEW: Main Python package
│   ├── cli/               # Command-line interface
│   ├── core/              # Geometry & Highdicom logic
│   ├── processing/        # Resampling logic
│   └── io/                # DICOM I/O & PNG Export
├── service/
│   ├── preprocess.py      # Legacy orchestration script
│   └── mapping.py         # (Deprecated) Old mapping script
├── tools/                 # Helper scripts for data prep
└── data/                  # Data directory
```

## Data Directories

### Source Data (Raw/Preprocessed)
These directories contain **original, unaligned** data in their native coordinate spaces:

| Directory | Content | Resolution | Aligned? |
|-----------|---------|------------|----------|
| `data/processed/` | T2 images (reference) | 256×256 | N/A (reference) |
| `data/processed_ep2d_adc/` | ADC maps | ~132×160 | ❌ Native ADC space |
| `data/processed_ep2d_calc/` | Calc maps | ~132×160 | ❌ Native Calc space |
| `data/processed_seg/` | Segmentation masks | 256×256 | ✅ T2 space |
| `data/nbia*/` | Original DICOM files | Various | Source metadata |

### Output Data (For AI Training) ✅
After running the pipeline, use **`data/aligned_v2/`** for training:

```
data/aligned_v2/class{N}/case_{XXXX}/
├── t2/              # ✅ PNG files for AI training
├── adc/             # ✅ PNG files for AI training (resampled to T2)
├── calc/            # ✅ PNG files for AI training (resampled to T2)
├── mask_prostate/   # ✅ Mask PNG files
├── mask_target1/    # ✅ Mask PNG files
├── t2_aligned/      # Secondary Capture DICOM files (*.dcm)
├── adc_aligned/     # Secondary Capture DICOM files (*.dcm)
└── calc_aligned/    # Secondary Capture DICOM files (*.dcm)
```

**For AI training**: Use `t2/`, `adc/`, `calc/`, `mask_*/` (PNG files).

**Slice correspondence**: `t2/0025.png`, `adc/0025.png`, `calc/0025.png`, and `mask_prostate/0025.png` all correspond to the **same physical location** in T2's coordinate system.

### Visualization Only
The `visualize_overlay_masks.py` script resamples on-the-fly for verification—it does NOT modify source directories:

```bash
# Verify alignment visually (reads DICOMs, resamples in memory)
python tools/preprocessing/visualize_overlay_masks.py --multimodal
```

## Why Spatial Resampling?
T2, ADC, and Calc have different resolutions, origins, and field-of-view. Simple pixel resizing causes misalignment. The pipeline uses **SimpleITK's native DICOM reader** to correctly transform ADC/Calc to T2's coordinate system using DICOM spatial metadata (origin, spacing, direction).

See [docs/dicom_mapping.md](docs/dicom_mapping.md) for technical details.
