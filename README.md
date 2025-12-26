# TCIA Handler

Scripts for generating TCIA manifest files for the Prostate MRI pipeline.

This repo is part of [MRI_2.5D](https://github.com/ccomkhj/MRI_2.5D_Segmentation).

```
git clone https://github.com/ccomkhj/MRI_2.5D_Segmentation mri
```

## Contents

```
tcia-handler/
├── service/
│   └── preprocess.py
└── tools/
    ├── preprocessing/
    ├── validation/
    ├── generate_tcia_by_class.py
    ├── generate_tcia_by_study.py
    └── README_TCIA_GENERATOR.md
```

## Usage

Run these commands from the `mri/` repo root so output files land in
`mri/data/`.

If your repos are not siblings, set:
- `MRI_ROOT` to the `mri/` repo root

```bash
# Full preprocessing pipeline (Steps 1-6)
python service/preprocess.py --all

# Individual steps
python service/preprocess.py --step excel_to_parquet
python service/preprocess.py --step merge_datasets
python service/preprocess.py --step generate_tcia
python service/preprocess.py --step dicom_to_png
python service/preprocess.py --step process_overlays
python service/preprocess.py --step validate_2d5

# Or run TCIA-only scripts
python tools/generate_tcia_by_class.py
python tools/generate_tcia_by_study.py
```

## Outputs

- `data/splitted_images/`, `data/splitted_info/`
- `data/tcia/{t2,ep2d_adc,ep2d_calc}/class{1-4}.tcia`
- `data/tcia/study/class{1-4}.tcia`
- `data/processed*`, `data/processed_seg/`, `data/visualizations/`

## Notes

- Data download workflow: https://github.com/ccomkhj/tcia-handler
- See `tcia-handler/tools/README_TCIA_GENERATOR.md` for TCIA details.
