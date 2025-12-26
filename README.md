# TCIA Handler

Scripts for generating TCIA manifest files for the Prostate MRI pipeline.

This repo is split out from `mri/` so TCIA-specific logic can evolve
independently.

Data download workflow and manifests are maintained in:
https://github.com/ccomkhj/tcia-handler

## Contents

```
tcia-handler/
└── tools/
    └── tcia/
        ├── tcia_generator.py
        ├── generate_tcia_by_class.py
        ├── generate_tcia_by_study.py
        └── README_TCIA_GENERATOR.md
```

## Usage

Run these commands from the `mri/` repo root so output files land in
`mri/data/tcia/`.

If your repos are not siblings, set one of:
- `TCIA_TOOLS_DIR` to the `tools/tcia` directory
- `TCIA_HANDLER_ROOT` to the `tcia-handler` repo root

```bash
# By series (T2, ADC, CALC_BVAL)
python ../tcia-handler/tools/tcia/generate_tcia_by_class.py

# By study (full download)
python ../tcia-handler/tools/tcia/generate_tcia_by_study.py
```

## Outputs

- `data/tcia/{t2,ep2d_adc,ep2d_calc}/class{1-4}.tcia`
- `data/tcia/study/class{1-4}.tcia`

## Notes

- Input files and parquet mappings are expected under `mri/data/`.
- See `tcia-handler/tools/tcia/README_TCIA_GENERATOR.md` for details.
