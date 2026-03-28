# DICOM Multi-Modal Mapping Guide

This guide describes the current `dicom_mapper` pipeline in this repository. It focuses on what `uv run dicom-mapper process` does today, which inputs it expects, and where the aligned outputs come from.

## Quick Start

```bash
# Align all discovered cases
uv run dicom-mapper process --input-dir data --output-dir data/aligned_v2

# Restrict processing to one class
uv run dicom-mapper process --input-dir data --output-dir data/aligned_v2 --class-num 1

# Process one case
uv run dicom-mapper process --input-dir data --output-dir data/aligned_v2 --class-num 1 --case-id 0144

# Render visual QA output from an aligned directory
uv run dicom-mapper visualize --aligned-dir data/aligned_v2 --output-dir data/visualizations_v2
```

For additional manual QA, `python tools/preprocessing/visualize_overlay_masks.py --multimodal` still exists, but it is a separate inspection script and not the implementation behind `dicom-mapper process`.

## Required Inputs

The CLI expects a root directory with these subtrees:

| Path | Purpose |
|------|---------|
| `data/processed/` | T2 PNG stacks grouped by `class*/case_*/{SeriesInstanceUID}` |
| `data/processed_ep2d_adc/` | ADC PNG stacks in native ADC space |
| `data/processed_ep2d_calc/` | Calc PNG stacks in native Calc space |
| `data/processed_seg/` | Mask PNG slices grouped by case and series UID |
| `data/nbia/` | Original T2 DICOM directories |
| `data/nbia_ep2d_adc/` | Original ADC DICOM directories |
| `data/nbia_ep2d_calc/` | Original Calc DICOM directories |

The processed PNG folders are used to discover cases and series UIDs. The original DICOM folders provide the geometry used for spatially correct resampling.

## Output Structure

```text
data/aligned_v2/class{N}/case_{XXXX}/
├── t2/              # PNG slices for training
├── adc/             # ADC resampled to the T2 grid
├── calc/            # Calc resampled to the T2 grid
├── mask_prostate/   # Full-volume mask PNGs
├── mask_target1/    # Full-volume mask PNGs
├── t2_aligned/      # Secondary Capture DICOM slices
├── adc_aligned/     # Secondary Capture DICOM slices
└── calc_aligned/    # Secondary Capture DICOM slices
```

Training code should use the PNG folders and `metadata.json`. The `*_aligned/` directories are inspection and archival outputs, not the training input format.

## How the Current Pipeline Matches Data

The mapper currently works as follows:

1. Discover cases from `data/processed/class*/case_*`.
2. For each modality, look inside that case directory and take the first discovered series folder.
3. Treat that folder name as the target `SeriesInstanceUID`.
4. Search the corresponding DICOM tree for files from the same case and prefer the directory whose DICOM `SeriesInstanceUID` matches the target folder name.
5. If no exact DICOM match is found, fall back to the first discovered DICOM directory for that case.

For masks, the pipeline looks in `data/processed_seg/class{N}/case_{XXXX}/`. If a segmentation series directory matches the T2 `SeriesInstanceUID`, it uses that. Otherwise it falls back to the first segmentation series directory.

This is important: the current CLI assumes one relevant processed series per modality and case. If a case contains multiple candidate series, the first discovered directory wins and the result should be reviewed.

## Why Resampling Works

Simple pixel resizing is not enough because T2, ADC, and Calc usually differ in:

- pixel spacing
- slice spacing
- image origin
- image direction

The pipeline reads the original DICOM series with `SimpleITK.ImageSeriesReader`, which preserves spatial metadata, then resamples ADC and Calc into T2 space with `SimpleITK.ResampleImageFilter`.

```python
reader = sitk.ImageSeriesReader()
reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(dicom_dir)))
volume = reader.Execute()

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(t2_volume)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0)
resampler.SetTransform(sitk.Transform())
aligned_adc = resampler.Execute(adc_volume)
```

The default fill value is `0`, so regions outside the ADC or Calc field of view become zero-valued voxels in the aligned output. The current mapper does not emit a `"Not Available"` sentinel image.

## Mask Handling

`data/processed_seg/` usually contains only slices where a structure is present. The mapper expands those sparse slices into a full T2-length volume:

- start with an all-zero mask volume matching T2 dimensions
- load each saved mask PNG by slice number
- resize with nearest-neighbor only if dimensions differ
- write the full mask volume back out as `mask_<name>/0000.png`, `0001.png`, ...

The resulting mask directories always follow T2 slice numbering.

## DICOM Output Notes

The mapper writes aligned image DICOMs as per-slice Secondary Capture instances. `highdicom.sc.SCImage` is used one frame at a time, so each aligned series becomes a directory of `.dcm` files that share a series UID.

There is a helper for DICOM SEG creation in `dicom_mapper/core/highdicom_creation.py`, but the CLI does not currently call it. Mask outputs are PNG only.

## Known Limitations

- Cross-modality matching is not currently based on `StudyInstanceUID` inside the main CLI.
- Multi-series cases may require manual validation because the first processed series directory is selected.
- If exact DICOM series matching fails, the CLI falls back to the first DICOM directory found for the case.
- ADC and Calc are resampled with linear interpolation. Masks are kept discrete via PNG export and nearest-neighbor resize when needed.

## Validation Checklist

- Run `uv run dicom-mapper visualize --aligned-dir data/aligned_v2 --output-dir data/visualizations_v2`.
- Confirm slice counts match across `t2/`, `adc/`, `calc/`, and each `mask_*` directory for a case.
- Spot-check cases with multiple source series directories.
- Inspect geometry in Python when debugging:

```python
img.GetSize()
img.GetSpacing()
img.GetOrigin()
img.GetDirection()
```

## Related Files

- `dicom_mapper/cli/pipeline.py`: end-to-end orchestration
- `dicom_mapper/processing/resampling.py`: DICOM loading and resampling
- `dicom_mapper/core/highdicom_creation.py`: Secondary Capture helpers
- `docs/train.md`: training dataset and metadata format
