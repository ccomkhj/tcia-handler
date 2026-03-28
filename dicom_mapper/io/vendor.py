"""Vendor-bundled MRI ZIP handling: extraction, DICOMDIR parsing, series classification."""

import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import pydicom

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


def get_series_info(dicomdir_path: Path) -> list[dict]:
    """Parse DICOMDIR and read first DICOM of each series to get classification tags."""
    ds = pydicom.dcmread(dicomdir_path)

    series_list = []
    current_info = None

    for record in ds.DirectoryRecordSequence:
        if record.DirectoryRecordType == "SERIES":
            if current_info is not None:
                series_list.append(current_info)
            uid = str(getattr(record, "SeriesInstanceUID", ""))
            current_info = {"uid": uid, "count": 0, "first_file": None}
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

    dicomdir_parent = dicomdir_path.parent
    result = []

    for series in series_list:
        if not series["first_file"]:
            continue

        dcm_path = dicomdir_parent / series["first_file"]
        if not dcm_path.exists():
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

        image_type = [str(t).upper() for t in getattr(dcm, "ImageType", [])]

        info = {
            "uid": series["uid"],
            "num_images": series["count"],
            "series_number": getattr(dcm, "SeriesNumber", 0),
            "rows": getattr(dcm, "Rows", 0),
            "columns": getattr(dcm, "Columns", 0),
            "image_type": image_type,
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

    Classification rules:
    - ADC: ImageType contains "ADC"
    - Calc/Trace: ImageType contains "TRACEW" or "CALC"
    - T2: ScanningSequence contains "SE", EchoTime > 80, RepetitionTime > 2500.
      When multiple candidates exist, selects closest to 512x512.

    Returns dict with keys 't2', 'adc', 'calc' mapping to series info or None.
    """
    adc = None
    calc = None
    t2_candidates = []

    for s in series_list:
        img_type = s["image_type"]

        if "ADC" in img_type:
            adc = s
            logger.info(
                f"  ADC: Series#{s['series_number']} "
                f"({s['rows']}x{s['columns']}, {s['num_images']} imgs)"
            )
            continue

        if "TRACEW" in img_type or "CALC" in img_type:
            calc = s
            logger.info(
                f"  Calc: Series#{s['series_number']} "
                f"({s['rows']}x{s['columns']}, {s['num_images']} imgs)"
            )
            continue

        scan_seq = s["scanning_sequence"].upper()
        if "SE" in scan_seq and s["echo_time"] > 80 and s["repetition_time"] > 2500:
            t2_candidates.append(s)

    if t2_candidates:
        t2 = min(
            t2_candidates,
            key=lambda s: abs(s["rows"] - 512) + abs(s["columns"] - 512),
        )
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


def case_name_from_zip(zip_path: Path) -> str:
    """Get case name from ZIP filename, using known mapping or generating one."""
    name = ZIP_TO_CASE.get(zip_path.name)
    if name is None:
        stem = zip_path.stem.lower().replace(" ", "_").replace(",", ".")
        name = f"case_{stem}"
        logger.warning(f"Unknown ZIP {zip_path.name}, using generated name: {name}")
    return name
