# TCIA Handler

Scripts for generating TCIA manifest files for the Prostate MRI pipeline.

This repo is split out from `mri/` so TCIA-specific logic can evolve
independently.

## Contents

```
tcia-handler/
└── tools/
    ├── tcia_generator.py
    ├── generate_tcia_by_class.py
    ├── generate_tcia_by_study.py
    └── README_TCIA_GENERATOR.md

```

## Usage

```bash
# By series (T2, ADC, CALC_BVAL)
python ../tcia-handler/tools/generate_tcia_by_class.py

# By study (full download)
python ../tcia-handler/tools/generate_tcia_by_study.py
```

## Outputs

- `data/tcia/{t2,ep2d_adc,ep2d_calc}/class{1-4}.tcia`
- `data/tcia/study/class{1-4}.tcia`

## Notes

- Input files and parquet mappings are expected under `mri/data/`.
- See `tcia-handler/tools/README_TCIA_GENERATOR.md` for details.
