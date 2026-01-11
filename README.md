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
Use the new CLI to align sequences, generate Standard DICOMs (Secondary Capture/Segmentation), and export training-ready PNGs.

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
The following scripts handle initial data download and preparation.

```bash
# Full preprocessing pipeline (Steps 1-6)
python service/preprocess.py --all

# Individual preprocessing steps
python service/preprocess.py --step excel_to_parquet
python service/preprocess.py --step merge_datasets
python service/preprocess.py --step generate_tcia
python service/preprocess.py --step dicom_to_png
python service/preprocess.py --step process_overlays
```

**Note:** `service/mapping.py` is **deprecated**. Please use `uv run dicom-mapper` instead.

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

## Outputs (New)
- `data/aligned_v2/`
    - `classX/case_Y/`
        - `t2_aligned.dcm` (Standard Multi-frame DICOM)
        - `t2/` (Exported PNGs for training)
