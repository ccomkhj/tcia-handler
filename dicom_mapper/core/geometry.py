from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from pathlib import Path

@dataclass
class SeriesInfo:
    """Information about a single DICOM series."""
    series_uid: str
    study_uid: str
    patient_id: str
    case_id: str
    class_num: int
    num_slices: int
    spacing: Tuple[float, float, float]  # (x, y, z)
    origin: Tuple[float, float, float]
    size: Tuple[int, int, int]  # (width, height, depth)
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    slice_locations: List[float] = field(default_factory=list)
    processed_dir: Optional[Path] = None
    dicom_dir: Optional[Path] = None

def compute_slice_mapping(t2_info: SeriesInfo, other_info: SeriesInfo) -> Dict[int, Dict[str, Any]]:
    """
    Compute slice-by-slice mapping between T2 and another modality (e.g. ADC, Calc).

    Args:
        t2_info: T2 series info with slice locations.
        other_info: Other modality series info with slice locations.

    Returns:
        Dict mapping T2 slice index to {z_position, other_slice_idx, distance, method}.
    """
    mapping = {}

    if not t2_info.slice_locations or not other_info.slice_locations:
        # Fallback to proportional mapping if spatial info is missing
        for i in range(t2_info.num_slices):
            ratio = i / max(1, t2_info.num_slices - 1)
            other_idx = int(ratio * (other_info.num_slices - 1))
            other_idx = max(0, min(other_idx, other_info.num_slices - 1))
            mapping[i] = {
                "t2_idx": i,
                "other_idx": other_idx,
                "method": "proportional"
            }
        return mapping

    # Spatial mapping using z-positions
    t2_z = np.array(t2_info.slice_locations)
    other_z = np.array(other_info.slice_locations)

    for i, z in enumerate(t2_z):
        # Find nearest slice in other modality
        distances = np.abs(other_z - z)
        nearest_idx = int(np.argmin(distances))
        min_distance = float(distances[nearest_idx])

        mapping[i] = {
            "t2_idx": i,
            "t2_z": float(z),
            "other_idx": nearest_idx,
            "other_z": float(other_z[nearest_idx]),
            "distance_mm": min_distance,
            "method": "spatial"
        }

    return mapping
