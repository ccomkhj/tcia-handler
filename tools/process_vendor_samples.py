#!/usr/bin/env python3
"""
Process vendor-bundled MRI ZIP files into inference-ready aligned format.

Extracts ZIPs, classifies DICOM series by tags (T2/ADC/Calc),
resamples to T2 reference grid, exports aligned PNGs + zero masks,
and generates metadata.json for the inference pipeline.

Usage:
    uv run python tools/process_vendor_samples.py
    uv run python tools/process_vendor_samples.py --input-dir test_sample --output-dir data/aligned_v2_sample
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ZIP filename -> case name mapping
ZIP_TO_CASE = {
    "PHilips AMbition 1,5T GL7B.zip": "case_philips_ambition_1.5t",
    "Philips Achieva 3T Prostata GL9.zip": "case_philips_achieva_3t",
    "Philips Elition X 3T Prostata GL9.zip": "case_philips_elition_x_3t",
    "SIEMENS MAGNETOM PRISMA 3T GL8.zip": "case_siemens_prisma_3t",
}


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Extract ZIP and return the extraction directory."""
    logger.info(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def find_dicomdir(extract_dir: Path) -> Optional[Path]:
    """Find DICOMDIR file in extracted directory."""
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.upper() == "DICOMDIR":
                return Path(root) / f
    return None


def get_series_info(dicomdir_path: Path, extract_dir: Path) -> list[dict]:
    """Parse DICOMDIR and read first DICOM of each series to classify it."""
    ds = pydicom.dcmread(dicomdir_path)

    # Collect series UIDs and their first referenced file
    series_list = []
    current_uid = None
    current_info = None

    for record in ds.DirectoryRecordSequence:
        if record.DirectoryRecordType == "SERIES":
            if current_info is not None:
                series_list.append(current_info)
            current_uid = str(getattr(record, "SeriesInstanceUID", ""))
            current_info = {"uid": current_uid, "count": 0, "first_file": None}
        elif record.DirectoryRecordType == "IMAGE" and current_info is not None:
            current_info["count"] += 1
            if current_info["first_file"] is None:
                ref = getattr(record, "ReferencedFileID", None)
                if ref:
                    if isinstance(ref, pydicom.multival.MultiValue):
                        ref = os.sep.join(ref)
                    current_info["first_file"] = str(ref)

    if current_info is not None:
        series_list.append(current_info)

    # Read first DICOM of each series to get classification tags
    dicomdir_parent = dicomdir_path.parent
    result = []

    for series in series_list:
        if not series["first_file"]:
            continue

        dcm_path = dicomdir_parent / series["first_file"]
        if not dcm_path.exists():
            # Try finding the file by walking
            rel_parts = series["first_file"].replace(os.sep, "/").split("/")
            candidates = list(dicomdir_parent.rglob(rel_parts[-1]))
            if candidates:
                dcm_path = candidates[0]
            else:
                logger.warning(f"Could not find DICOM file: {series['first_file']}")
                continue

        try:
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        except Exception as e:
            logger.warning(f"Could not read {dcm_path}: {e}")
            continue

        image_type = list(getattr(dcm, "ImageType", []))
        image_type_upper = [str(t).upper() for t in image_type]

        info = {
            "uid": series["uid"],
            "num_images": series["count"],
            "series_number": getattr(dcm, "SeriesNumber", 0),
            "rows": getattr(dcm, "Rows", 0),
            "columns": getattr(dcm, "Columns", 0),
            "image_type": image_type_upper,
            "scanning_sequence": str(getattr(dcm, "ScanningSequence", "")),
            "echo_time": float(getattr(dcm, "EchoTime", 0)),
            "repetition_time": float(getattr(dcm, "RepetitionTime", 0)),
            "dicom_dir": str(dcm_path.parent),
        }
        result.append(info)

        logger.debug(
            f"  Series#{info['series_number']}: {info['rows']}x{info['columns']} "
            f"TR={info['repetition_time']:.0f} TE={info['echo_time']:.0f} "
            f"Type={info['image_type']} Imgs={info['num_images']}"
        )

    return result


def classify_series(series_list: list[dict]) -> dict[str, Optional[dict]]:
    """
    Classify series into T2, ADC, and Calc based on DICOM tags.

    Returns dict with keys 't2', 'adc', 'calc' mapping to series info or None.
    """
    adc = None
    calc = None
    t2_candidates = []

    for s in series_list:
        img_type = s["image_type"]

        # ADC: ImageType contains "ADC"
        if "ADC" in img_type:
            adc = s
            logger.info(
                f"  ADC: Series#{s['series_number']} "
                f"({s['rows']}x{s['columns']}, {s['num_images']} imgs)"
            )
            continue

        # Calc/Trace: ImageType contains "TRACEW" or "CALC"
        if "TRACEW" in img_type or "CALC" in img_type:
            calc = s
            logger.info(
                f"  Calc: Series#{s['series_number']} "
                f"({s['rows']}x{s['columns']}, {s['num_images']} imgs)"
            )
            continue

        # T2 candidate: SE sequence, TE > 80, TR > 2500
        scan_seq = s["scanning_sequence"].upper()
        if "SE" in scan_seq and s["echo_time"] > 80 and s["repetition_time"] > 2500:
            t2_candidates.append(s)

    # Select T2: closest matrix size to 512x512
    if t2_candidates:
        t2 = min(t2_candidates, key=lambda s: abs(s["rows"] - 512) + abs(s["columns"] - 512))
        logger.info(
            f"  T2: Series#{t2['series_number']} "
            f"({t2['rows']}x{t2['columns']}, {t2['num_images']} imgs) "
            f"[selected from {len(t2_candidates)} candidates]"
        )
    else:
        t2 = None
        logger.warning("  No T2 series found!")

    if adc is None:
        logger.warning("  No ADC series found!")
    if calc is None:
        logger.info("  No Calc/Trace series found (expected for Philips scanners)")

    return {"t2": t2, "adc": adc, "calc": calc}


def load_volume(dicom_dir: str) -> sitk.Image:
    """Load DICOM series as SimpleITK volume."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_names:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def resample_to_reference(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample moving volume to reference grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving)


def export_volume_to_png(volume: sitk.Image, output_dir: Path) -> int:
    """Export SimpleITK volume slices to PNG sequence. Returns number of slices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arr = sitk.GetArrayFromImage(volume)  # (Z, Y, X)
    num_slices = arr.shape[0]

    for i in range(num_slices):
        frame = arr[i]
        # Normalize to uint8
        if frame.dtype != np.uint8:
            min_val = float(frame.min())
            max_val = float(frame.max())
            if max_val > min_val:
                frame = ((frame - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            else:
                frame = np.zeros_like(frame, dtype=np.uint8)
        Image.fromarray(frame).save(output_dir / f"{i:04d}.png")

    return num_slices


def create_zero_masks(num_slices: int, height: int, width: int, output_dir: Path):
    """Create all-zero mask PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zero_img = Image.fromarray(np.zeros((height, width), dtype=np.uint8))
    for i in range(num_slices):
        zero_img.save(output_dir / f"{i:04d}.png")


def process_single_zip(
    zip_path: Path,
    case_name: str,
    output_root: Path,
    group_name: str = "sample",
):
    """Process a single vendor ZIP into the aligned output structure."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {zip_path.name} -> {case_name}")
    logger.info(f"{'='*60}")

    case_dir = output_root / group_name / case_name

    with tempfile.TemporaryDirectory() as tmpdir:
        extract_dir = Path(tmpdir) / "extracted"
        extract_zip(zip_path, extract_dir)

        # Find and parse DICOMDIR
        dicomdir_path = find_dicomdir(extract_dir)
        if dicomdir_path is None:
            logger.error(f"No DICOMDIR found in {zip_path.name}")
            return False

        series_list = get_series_info(dicomdir_path, extract_dir)
        logger.info(f"Found {len(series_list)} series")

        # Classify series
        classified = classify_series(series_list)
        if classified["t2"] is None:
            logger.error(f"No T2 series found for {case_name}, skipping")
            return False

        # Load T2 reference volume
        logger.info("Loading T2 reference volume...")
        t2_volume = load_volume(classified["t2"]["dicom_dir"])
        t2_size = t2_volume.GetSize()  # (X, Y, Z)
        num_slices = t2_size[2]
        height, width = t2_size[1], t2_size[0]
        logger.info(f"  T2 volume: {t2_size}, spacing={t2_volume.GetSpacing()}")

        # Export T2
        logger.info("Exporting T2 PNGs...")
        export_volume_to_png(t2_volume, case_dir / "t2")

        # Process ADC
        if classified["adc"] is not None:
            logger.info("Loading and resampling ADC...")
            adc_volume = load_volume(classified["adc"]["dicom_dir"])
            logger.info(f"  ADC volume: {adc_volume.GetSize()}, spacing={adc_volume.GetSpacing()}")
            adc_resampled = resample_to_reference(adc_volume, t2_volume)
            export_volume_to_png(adc_resampled, case_dir / "adc")
        else:
            logger.info("No ADC series found, skipping")

        # Process Calc
        if classified["calc"] is not None:
            logger.info("Loading and resampling Calc...")
            calc_volume = load_volume(classified["calc"]["dicom_dir"])
            logger.info(f"  Calc volume: {calc_volume.GetSize()}, spacing={calc_volume.GetSpacing()}")
            calc_resampled = resample_to_reference(calc_volume, t2_volume)
            export_volume_to_png(calc_resampled, case_dir / "calc")
        else:
            logger.info("No Calc/Trace series found, skipping")

        # Create zero masks
        logger.info("Creating zero masks...")
        create_zero_masks(num_slices, height, width, case_dir / "mask_prostate")
        create_zero_masks(num_slices, height, width, case_dir / "mask_target1")

    logger.info(f"Done: {case_name} ({num_slices} slices, {width}x{height})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process vendor MRI ZIPs into inference-ready aligned format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("test_sample"),
        help="Directory containing vendor ZIP files (default: test_sample)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/aligned_v2_sample"),
        help="Output directory (default: data/aligned_v2_sample)",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return

    # Find ZIP files
    zip_files = sorted(args.input_dir.glob("*.zip"))
    if not zip_files:
        logger.error(f"No ZIP files found in {args.input_dir}")
        return

    logger.info(f"Found {len(zip_files)} ZIP files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each ZIP
    success_count = 0
    for zip_path in zip_files:
        case_name = ZIP_TO_CASE.get(zip_path.name)
        if case_name is None:
            # Generate case name from filename
            stem = zip_path.stem.lower().replace(" ", "_").replace(",", ".")
            case_name = f"case_{stem}"
            logger.warning(f"Unknown ZIP {zip_path.name}, using generated name: {case_name}")

        if process_single_zip(zip_path, case_name, args.output_dir):
            success_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Processed {success_count}/{len(zip_files)} ZIPs successfully")
    logger.info(f"Output: {args.output_dir}")

    # Generate metadata
    if success_count > 0:
        logger.info("\nGenerating metadata.json...")
        # Add tools directory to path so we can import generate_training_metadata
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_training_metadata import generate_metadata

        generate_metadata(
            data_dir=args.output_dir,
            output_path=args.output_dir / "metadata.json",
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
