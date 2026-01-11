import pydicom
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def extract_slice_locations_from_dicom(dicom_dir: Path) -> Tuple[List[float], Dict]:
    """
    Extract per-slice z-positions and rescaling metadata from DICOM files.

    Args:
        dicom_dir: Directory containing DICOM series.

    Returns:
        Tuple of (sorted z-positions list, metadata dict).
    """
    slice_locations = []
    meta = {}

    dcm_files = list(dicom_dir.rglob("*.dcm"))
    if not dcm_files:
        return [], {}

    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)

            # Get z-position (prefer ImagePositionPatient, fallback to SliceLocation)
            if hasattr(ds, 'ImagePositionPatient'):
                z = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'SliceLocation'):
                z = float(ds.SliceLocation)
            else:
                continue

            slice_locations.append(z)

            # Extract metadata from first file
            if not meta:
                meta = {
                    "StudyInstanceUID": getattr(ds, "StudyInstanceUID", ""),
                    "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", ""),
                    "PatientID": getattr(ds, "PatientID", ""),
                    "PixelSpacing": list(ds.PixelSpacing) if hasattr(ds, "PixelSpacing") else [1.0, 1.0],
                    "SliceThickness": float(getattr(ds, "SliceThickness", 1.0)),
                    "Rows": int(getattr(ds, "Rows", 256)),
                    "Columns": int(getattr(ds, "Columns", 256)),
                    "RescaleSlope": float(getattr(ds, "RescaleSlope", 1.0)),
                    "RescaleIntercept": float(getattr(ds, "RescaleIntercept", 0.0)),
                }
                if hasattr(ds, 'ImagePositionPatient'):
                    meta["Origin"] = [float(x) for x in ds.ImagePositionPatient]
                    meta["ImageOrientationPatient"] = [float(x) for x in getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]

        except Exception as e:
            logger.warning(f"Error reading DICOM file {dcm_file}: {e}")
            continue

    # Sort and deduplicate
    slice_locations = sorted(set(slice_locations))

    return slice_locations, meta

def load_dicom_series(dicom_dir: Path) -> List[pydicom.dataset.FileDataset]:
    """
    Load a full DICOM series as a list of pydicom datasets, sorted by slice location.
    
    Args:
        dicom_dir: Directory containing DICOM files for a single series.
        
    Returns:
        List of pydicom datasets sorted by z-position.
    """
    datasets = []
    dcm_files = list(dicom_dir.rglob("*.dcm"))
    
    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(str(dcm_file))
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"Error reading DICOM {dcm_file}: {e}")
            continue
            
    # Sort by ImagePositionPatient[2] (Z) or SliceLocation
    def get_z(ds):
        if hasattr(ds, 'ImagePositionPatient'):
            return float(ds.ImagePositionPatient[2])
        elif hasattr(ds, 'SliceLocation'):
            return float(ds.SliceLocation)
        return 0.0
        
    datasets.sort(key=get_z)
    return datasets
