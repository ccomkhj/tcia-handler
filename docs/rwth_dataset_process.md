# RWTH Dataset Processing

Process vendor-bundled MRI ZIP files (Philips/Siemens scanner exports) into inference-ready aligned format.

## Prerequisites

- ZIP files placed in `test_sample/` (or any directory)
- Project dependencies installed (`uv sync`)

## Supported Scanners

| Scanner | T2 | ADC | Calc |
|---------|:--:|:---:|:----:|
| Philips Ambition 1.5T | yes | yes | no |
| Philips Achieva 3T | yes | yes | no |
| Philips Elition X 3T | yes | yes | no |
| Siemens Magnetom Prisma 3T | yes | yes | yes |

Calc/Trace is only available from Siemens scanners. The inference pipeline handles missing modalities by substituting zeros.

## Usage

```bash
uv run python tools/process_vendor_samples.py \
  --input-dir test_sample \
  --output-dir data/aligned_v2_sample
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `test_sample` | Directory containing vendor ZIP files |
| `--output-dir` | `data/aligned_v2_sample` | Output directory for aligned data |

## What It Does

1. Extracts each ZIP to a temp directory
2. Parses DICOMDIR to discover all series
3. Classifies series by DICOM tags:
   - **T2**: `ScanningSequence` contains `SE`, `EchoTime > 80`, `RepetitionTime > 2500`. When multiple candidates exist, selects the one closest to 512x512.
   - **ADC**: `ImageType` contains `ADC`
   - **Calc**: `ImageType` contains `TRACEW` or `CALC`
4. Loads T2 as reference volume (SimpleITK)
5. Resamples ADC/Calc to T2 spatial grid
6. Exports aligned PNGs (uint8, normalized 0-255)
7. Creates all-zero mask PNGs (no segmentation labels)
8. Generates `metadata.json` for the inference pipeline

## Output Structure

```
data/aligned_v2_sample/
├── metadata.json
└── sample/
    ├── case_philips_ambition_1.5t/
    │   ├── t2/
    │   │   ├── 0000.png
    │   │   ├── 0001.png
    │   │   └── ...
    │   ├── adc/
    │   │   └── ...
    │   ├── mask_prostate/
    │   │   └── ...           (all-zero)
    │   └── mask_target1/
    │       └── ...           (all-zero)
    ├── case_philips_achieva_3t/
    ├── case_philips_elition_x_3t/
    └── case_siemens_prisma_3t/
        ├── t2/
        ├── adc/
        ├── calc/              (only Siemens has this)
        ├── mask_prostate/
        └── mask_target1/
```

## Running Inference

Point the MRI inference pipeline to the generated `metadata.json`:

```python
from mri.data.metadata import load_metadata

metadata_path = "path/to/data/aligned_v2_sample/metadata.json"
meta = load_metadata(metadata_path)
```

The metadata is compatible with both `run_segmentation_inference()` and `run_classification_inference()`.

## Adding New ZIPs

To process additional vendor ZIPs, either:

1. Add the ZIP filename to the `ZIP_TO_CASE` mapping in `tools/process_vendor_samples.py`
2. Or let the script auto-generate a case name from the filename

The series classification is tag-based and should work for any Philips or Siemens MRI export that includes a DICOMDIR.
