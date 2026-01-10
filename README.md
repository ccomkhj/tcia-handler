# TCIA Handler

Scripts for generating TCIA manifest files and multi-modal MRI alignment for the Prostate MRI pipeline.

This repo is part of [MRI_2.5D](https://github.com/ccomkhj/MRI_2.5D_Segmentation).

```
git clone https://github.com/ccomkhj/MRI_2.5D_Segmentation mri
```

## Contents

```
tcia-handler/
├── service/
│   ├── preprocess.py      # Full preprocessing pipeline
│   └── mapping.py         # Multi-modal alignment (T2 + ADC + Calc)
├── tools/
│   ├── preprocessing/
│   ├── validation/
│   ├── generate_tcia_by_class.py
│   └── generate_tcia_by_study.py
└── docs/
    └── dicom_mapping.md   # Multi-modal mapping guide
```

## Usage

Run from the repo root. Set `MRI_ROOT` if repos are not siblings.

```bash
# Full preprocessing pipeline (Steps 1-6)
python service/preprocess.py --all

# Individual preprocessing steps
python service/preprocess.py --step excel_to_parquet
python service/preprocess.py --step merge_datasets
python service/preprocess.py --step generate_tcia
python service/preprocess.py --step dicom_to_png
python service/preprocess.py --step process_overlays
python service/preprocess.py --step validate_2d5

# Multi-modal alignment (T2 + ADC + Calc → aligned output)
python service/mapping.py --all              # Align all cases
python service/mapping.py --class 2          # Specific class
python service/mapping.py --dry-run          # Preview only
python service/mapping.py --validate         # Validate output

# Visualize masks with spatial alignment
python tools/preprocessing/visualize_overlay_masks.py
```

## Outputs

- `data/splitted_images/`, `data/splitted_info/`
- `data/tcia/{t2,ep2d_adc,ep2d_calc,study}/class{1-4}.tcia`
- `data/processed/`, `data/processed_ep2d_*/`, `data/processed_seg/`
- `data/aligned/` — Multi-channel aligned output (T2 + ADC + Calc + masks)
- `data/visualizations/`

## Notes

- See `docs/dicom_mapping.md` for multi-modal alignment details.
- See `tools/README_TCIA_GENERATOR.md` for TCIA manifest details.
