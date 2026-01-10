#!/usr/bin/env python3
"""
Visualize segmentation masks overlaid on MRI images.

This script creates visualization images showing:
- Original MRI image
- Segmentation mask overlay (colored)
- Side-by-side comparison
- Multi-mask overlay (prostate + targets)

Supports spatial alignment for multi-modal visualization:
- T2 images with directly matched masks
- ADC/Calc images with spatially aligned masks (using z-coordinate matching)

Requirements:
    pip install pandas pyarrow numpy pillow matplotlib tqdm pydicom SimpleITK
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import pydicom
    import SimpleITK as sitk

    HAS_SPATIAL_LIBS = True
except ImportError:
    HAS_SPATIAL_LIBS = False


# Color scheme for different structures
STRUCTURE_COLORS = {
    "prostate": (255, 255, 0, 100),  # Yellow, semi-transparent
    "target1": (255, 0, 0, 150),  # Red, more opaque
    "target2": (255, 128, 0, 150),  # Orange
    "target3": (255, 0, 255, 150),  # Magenta
    "default": (0, 255, 0, 100),  # Green for unknown structures
}

SEQUENCE_PROCESSED_DIRS = {
    "t2": Path("data/processed"),
    "ep2d_adc": Path("data/processed_ep2d_adc"),
    "ep2d_calc": Path("data/processed_ep2d_calc"),
}

# DICOM directories for extracting spatial metadata
DICOM_DIRS = {
    "t2": Path("data/nbia"),
    "ep2d_adc": Path("data/nbia_ep2d_adc"),
    "ep2d_calc": Path("data/nbia_ep2d_calc"),
}


# =============================================================================
# Spatial Alignment Utilities
# =============================================================================


def extract_slice_locations_from_dicom(dicom_dir: Path) -> List[float]:
    """
    Extract per-slice z-positions from DICOM files.

    Args:
        dicom_dir: Directory containing DICOM series

    Returns:
        Sorted list of z-positions
    """
    if not HAS_SPATIAL_LIBS:
        return []

    slice_locations = []
    dcm_files = list(dicom_dir.rglob("*.dcm"))

    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            if hasattr(ds, "ImagePositionPatient"):
                z = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, "SliceLocation"):
                z = float(ds.SliceLocation)
            else:
                continue
            slice_locations.append(z)
        except Exception:
            continue

    return sorted(set(slice_locations))


def find_dicom_series_dir(
    dicom_base: Path, class_num: int, case_id: str, series_uid: str
) -> Optional[Path]:
    """Find DICOM directory for a specific series."""
    class_dir = dicom_base / f"class{class_num}"
    if not class_dir.exists():
        return None

    for manifest_dir in class_dir.glob("manifest-*"):
        for patient_dir in manifest_dir.rglob(f"*{case_id}*"):
            if patient_dir.is_dir():
                for study_dir in patient_dir.iterdir():
                    if study_dir.is_dir():
                        for series_dir in study_dir.iterdir():
                            if series_dir.is_dir():
                                dcm_files = list(series_dir.glob("*.dcm"))
                                if dcm_files:
                                    try:
                                        ds = pydicom.dcmread(
                                            str(dcm_files[0]), stop_before_pixels=True
                                        )
                                        if (
                                            getattr(ds, "SeriesInstanceUID", "")
                                            == series_uid
                                        ):
                                            return series_dir
                                    except Exception:
                                        pass
    return None


def compute_z_to_slice_mapping(
    source_z: List[float], target_z: List[float]
) -> Dict[int, int]:
    """
    Compute mapping from source slice indices to nearest target slice indices.

    Args:
        source_z: Z-positions of source slices (e.g., ADC)
        target_z: Z-positions of target slices (e.g., T2 masks)

    Returns:
        Dict mapping source slice index to target slice index
    """
    if not source_z or not target_z:
        return {}

    mapping = {}
    target_z_arr = np.array(target_z)

    for i, z in enumerate(source_z):
        distances = np.abs(target_z_arr - z)
        nearest_idx = int(np.argmin(distances))
        mapping[i] = nearest_idx

    return mapping


def get_spatial_slice_mapping(
    source_meta: Dict,
    target_meta: Dict,
    source_dicom_dir: Optional[Path],
    target_dicom_dir: Optional[Path],
    source_num_slices: int,
    target_num_slices: int,
) -> Dict[int, int]:
    """
    Get slice mapping between source and target series using spatial coordinates.

    Falls back to proportional mapping if spatial data is unavailable.

    Args:
        source_meta: Source series metadata (from meta.json)
        target_meta: Target series metadata (e.g., T2 for masks)
        source_dicom_dir: Path to source DICOM directory
        target_dicom_dir: Path to target DICOM directory
        source_num_slices: Number of slices in source
        target_num_slices: Number of slices in target

    Returns:
        Dict mapping source slice index to target slice index
    """
    # Try spatial mapping first
    if HAS_SPATIAL_LIBS and source_dicom_dir and target_dicom_dir:
        source_z = extract_slice_locations_from_dicom(source_dicom_dir)
        target_z = extract_slice_locations_from_dicom(target_dicom_dir)

        if source_z and target_z:
            return compute_z_to_slice_mapping(source_z, target_z)

    # Fallback to proportional mapping
    mapping = {}
    for i in range(source_num_slices):
        ratio = i / max(1, source_num_slices - 1)
        target_idx = int(ratio * (target_num_slices - 1))
        target_idx = max(0, min(target_idx, target_num_slices - 1))
        mapping[i] = target_idx

    return mapping


def resample_mask_to_size(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resample a mask to target size using nearest neighbor interpolation.
    NOTE: This is a simple resize, use resample_mask_spatially for proper alignment.

    Args:
        mask: Binary mask array
        target_size: (width, height) tuple

    Returns:
        Resampled mask
    """
    if mask.shape[1] == target_size[0] and mask.shape[0] == target_size[1]:
        return mask

    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize(target_size, Image.NEAREST)
    return np.array(mask_img)


def extract_dicom_spatial_info(dicom_dir: Path) -> Optional[Dict]:
    """
    Extract spatial information from DICOM files for coordinate transformation.

    This function extracts the full spatial information needed for proper
    coordinate transformation, including the 3x3 direction matrix (computed
    from ImageOrientationPatient using the cross product for slice direction).

    DICOM uses LPS (Left-Posterior-Superior) coordinate system.
    SimpleITK also uses LPS by default.

    Args:
        dicom_dir: Directory containing DICOM series

    Returns:
        Dict with:
        - origin: [x, y, z] in LPS coordinates
        - spacing: [col_spacing, row_spacing, slice_spacing]
        - size: [cols, rows]
        - direction: 9-element flat array for 3x3 direction matrix (row-major)
    """
    if not HAS_SPATIAL_LIBS or not dicom_dir or not dicom_dir.exists():
        return None

    dcm_files = list(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        return None

    try:
        # Read first DICOM file for in-plane info
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)

        pixel_spacing = (
            [float(x) for x in ds.PixelSpacing]
            if hasattr(ds, "PixelSpacing")
            else [1.0, 1.0]
        )
        rows = int(ds.Rows) if hasattr(ds, "Rows") else 256
        cols = int(ds.Columns) if hasattr(ds, "Columns") else 256

        # Get origin from ImagePositionPatient
        if hasattr(ds, "ImagePositionPatient"):
            origin = [float(x) for x in ds.ImagePositionPatient]
        else:
            origin = [0.0, 0.0, 0.0]

        # Get image orientation (direction cosines for row and column)
        # ImageOrientationPatient contains 6 values:
        #   [row_x, row_y, row_z, col_x, col_y, col_z]
        # Row direction: direction of increasing column index
        # Col direction: direction of increasing row index
        if hasattr(ds, "ImageOrientationPatient"):
            orientation = [float(x) for x in ds.ImageOrientationPatient]
        else:
            orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Default axial

        # Extract row and column direction vectors
        row_dir = np.array(orientation[0:3])  # Direction along columns (I axis)
        col_dir = np.array(orientation[3:6])  # Direction along rows (J axis)

        # Compute slice direction as cross product (K axis)
        # This gives the normal to the image plane
        slice_dir = np.cross(row_dir, col_dir)
        slice_dir = slice_dir / np.linalg.norm(slice_dir)  # Normalize

        # Build 3x3 direction matrix (column vectors are the axis directions)
        # For SimpleITK, direction is stored as row-major flat array
        # direction[i*3+j] = matrix[row=i][col=j]
        # Column 0: I axis (row direction)
        # Column 1: J axis (col direction)
        # Column 2: K axis (slice direction)
        direction = [
            row_dir[0], col_dir[0], slice_dir[0],  # Row 0
            row_dir[1], col_dir[1], slice_dir[1],  # Row 1
            row_dir[2], col_dir[2], slice_dir[2],  # Row 2
        ]

        # Get slice thickness
        slice_thickness = (
            float(ds.SliceThickness) if hasattr(ds, "SliceThickness") else 1.0
        )

        return {
            "origin": origin,
            # Spacing: [column_spacing, row_spacing, slice_spacing]
            # PixelSpacing is [row_spacing, column_spacing] in DICOM
            "spacing": [pixel_spacing[1], pixel_spacing[0], slice_thickness],
            "size": [cols, rows],
            "direction": direction,
        }
    except Exception as e:
        return None


def resample_mask_spatially(
    mask: np.ndarray,
    source_spatial: Dict,
    target_spatial: Dict,
    target_image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resample a mask from source coordinate space to target coordinate space.

    This implements the same approach as 3D Slicer for resampling between
    different image geometries. It properly handles different:
    - Field of View (FOV)
    - Origins (ImagePositionPatient)
    - Pixel spacings
    - Direction cosines (ImageOrientationPatient)

    The transformation follows Slicer's method:
    1. Build IJK-to-World transform for source (T2 mask space)
    2. Build IJK-to-World transform for target (ADC/Calc space)
    3. Resample source to target geometry

    Args:
        mask: Binary mask array (H, W) from source (T2) space
        source_spatial: Spatial info dict from T2 DICOM:
            - origin: [x, y, z] in LPS
            - spacing: [col, row, slice]
            - direction: 9-element direction matrix
        target_spatial: Spatial info dict from ADC/Calc DICOM
        target_image_size: Target image size (width, height)

    Returns:
        Resampled mask in target coordinate space
    """
    if not HAS_SPATIAL_LIBS:
        # Fallback to simple resize
        return resample_mask_to_size(mask, target_image_size)

    try:
        # Create 2D SimpleITK image from mask with source spatial info
        # Note: SimpleITK arrays are in [z,y,x] order, 2D is [y,x] = [row, col]
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))

        # Set source spatial information (T2 space) - 2D
        # Extract 2D spacing and origin from 3D info
        source_spacing_2d = [source_spatial["spacing"][0], source_spatial["spacing"][1]]
        source_origin_2d = [source_spatial["origin"][0], source_spatial["origin"][1]]

        # Extract 2D direction from 3D direction matrix
        # direction is 9 elements: [d00,d01,d02, d10,d11,d12, d20,d21,d22]
        # For 2D, we need the top-left 2x2 submatrix
        src_dir = source_spatial.get("direction", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        source_direction_2d = [
            src_dir[0], src_dir[1],  # First row: d00, d01
            src_dir[3], src_dir[4],  # Second row: d10, d11
        ]

        mask_sitk.SetSpacing(source_spacing_2d)
        mask_sitk.SetOrigin(source_origin_2d)
        mask_sitk.SetDirection(source_direction_2d)

        # Create reference image in target space
        reference = sitk.Image(
            target_image_size[0], target_image_size[1], sitk.sitkUInt8
        )

        target_spacing_2d = [target_spatial["spacing"][0], target_spatial["spacing"][1]]
        target_origin_2d = [target_spatial["origin"][0], target_spatial["origin"][1]]

        tgt_dir = target_spatial.get("direction", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        target_direction_2d = [
            tgt_dir[0], tgt_dir[1],
            tgt_dir[3], tgt_dir[4],
        ]

        reference.SetSpacing(target_spacing_2d)
        reference.SetOrigin(target_origin_2d)
        reference.SetDirection(target_direction_2d)

        # Resample mask to target space using identity transform
        # The spatial metadata (origin, spacing, direction) handles the coordinate mapping
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Keep binary mask
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform(2, sitk.sitkIdentity))  # 2D identity

        resampled = resampler.Execute(mask_sitk)
        result = sitk.GetArrayFromImage(resampled)

        return result.astype(np.uint8)

    except Exception as e:
        # Fallback to simple resize on error
        print(f"Warning: Spatial resampling failed ({e}), using simple resize")
        return resample_mask_to_size(mask, target_image_size)


def find_matching_cases(
    processed_dir: Path,
    processed_seg_dir: Path,
    sequence_name: Optional[str] = None,
) -> List[Tuple[Path, Path]]:
    """
    Find matching case directories between processed and processed_seg.

    Args:
        processed_dir: Path to processed/ directory
        processed_seg_dir: Path to processed_seg/ directory
        sequence_name: Optional sequence label for logging

    Returns:
        List of tuples (processed_case_path, processed_seg_case_path)
    """
    matches = []

    print(f"\n{'='*80}")
    if sequence_name:
        print(f"Finding matching cases ({sequence_name})...")
    else:
        print("Finding matching cases...")
    print(f"{'='*80}")

    # Iterate through all class directories
    for class_dir in sorted(processed_dir.glob("class*")):
        class_name = class_dir.name
        seg_class_dir = processed_seg_dir / class_name

        if not seg_class_dir.exists():
            continue

        # Iterate through all cases in this class
        for case_dir in sorted(class_dir.glob("case_*")):
            case_name = case_dir.name
            seg_case_dir = seg_class_dir / case_name

            if seg_case_dir.exists():
                matches.append((case_dir, seg_case_dir))

    print(f"✓ Found {len(matches)} matching cases\n")
    return matches


def build_seg_series_index(case_seg: Path) -> Dict[str, Dict]:
    """Index segmentation series dirs with structure and mask counts."""
    seg_series_info: Dict[str, Dict] = {}
    for seg_series_dir in sorted(case_seg.iterdir()):
        if not seg_series_dir.is_dir():
            continue
        structure_dirs = [d for d in seg_series_dir.iterdir() if d.is_dir()]
        if not structure_dirs:
            continue
        mask_counts = [len(list(sd.glob("*.png"))) for sd in structure_dirs]
        max_mask_count = max(mask_counts) if mask_counts else 0
        if max_mask_count == 0:
            continue
        seg_series_info[seg_series_dir.name] = {
            "dir": seg_series_dir,
            "structure_dirs": structure_dirs,
            "mask_count": max_mask_count,
        }
    return seg_series_info


def read_meta(series_dir: Path) -> Dict:
    meta_path = series_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def build_seg_series_meta_index(
    seg_series_info: Dict[str, Dict], t2_case_dir: Optional[Path]
) -> Dict[str, Dict]:
    """Load T2 meta.json for segmentation series UIDs."""
    if t2_case_dir is None or not t2_case_dir.exists():
        return {}

    meta_index = {}
    for series_uid in seg_series_info.keys():
        meta = read_meta(t2_case_dir / series_uid)
        if meta:
            meta_index[series_uid] = meta
    return meta_index


def select_seg_series_entry(
    series_uid: str,
    seg_series_info: Dict[str, Dict],
    image_count: int,
    series_meta: Optional[Dict] = None,
    seg_series_meta: Optional[Dict[str, Dict]] = None,
) -> Tuple[Optional[Dict], str]:
    if series_uid in seg_series_info:
        return seg_series_info[series_uid], "exact"
    if not seg_series_info:
        return None, "missing"

    study_uid = series_meta.get("StudyInstanceUID") if series_meta else None
    if study_uid and seg_series_meta:
        candidates = [
            uid
            for uid, meta in seg_series_meta.items()
            if meta.get("StudyInstanceUID") == study_uid
        ]
        if candidates:
            target_slices = series_meta.get("num_slices") or image_count

            def score(uid: str) -> Tuple[int, int, str]:
                meta = seg_series_meta.get(uid, {})
                seg_slices = meta.get("num_slices")
                if seg_slices is None:
                    seg_slices = seg_series_info[uid]["mask_count"]
                diff = abs(int(seg_slices) - int(target_slices))
                return (diff, -int(seg_slices), uid)

            best_uid = min(candidates, key=score)
            return seg_series_info[best_uid], "study_uid"

    def score(name: str) -> Tuple[int, int, str]:
        mask_count = seg_series_info[name]["mask_count"]
        diff = abs(mask_count - image_count)
        return (diff, -mask_count, name)

    best_name = min(seg_series_info.keys(), key=score)
    return seg_series_info[best_name], "fallback"


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load an image as numpy array.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (H, W) or None if failed
    """
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        return None


def create_overlay(
    image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Create an overlay of mask on image.

    Args:
        image: Original image (H, W) grayscale
        mask: Binary mask (H, W)
        color: RGBA color tuple

    Returns:
        RGB image with overlay (H, W, 3)
    """
    # Convert grayscale to RGB
    if len(image.shape) == 2:
        img_rgb = np.stack([image, image, image], axis=-1)
    else:
        img_rgb = image.copy()

    # Normalize if needed
    if img_rgb.max() > 1:
        img_rgb = img_rgb.astype(np.float32) / 255.0

    # Create colored mask
    r, g, b, a = color
    alpha = a / 255.0

    # Apply mask color with alpha blending
    mask_bool = mask > 0
    if mask_bool.any():
        img_rgb[mask_bool, 0] = (1 - alpha) * img_rgb[mask_bool, 0] + alpha * (
            r / 255.0
        )
        img_rgb[mask_bool, 1] = (1 - alpha) * img_rgb[mask_bool, 1] + alpha * (
            g / 255.0
        )
        img_rgb[mask_bool, 2] = (1 - alpha) * img_rgb[mask_bool, 2] + alpha * (
            b / 255.0
        )

    return (img_rgb * 255).astype(np.uint8)


def visualize_case(
    case_processed: Path,
    case_seg: Path,
    output_dir: Path,
    max_slices: int = 10,
    sequence_name: Optional[str] = None,
    t2_processed_dir: Optional[Path] = None,
    dicom_base_dir: Optional[Path] = None,
    t2_dicom_base_dir: Optional[Path] = None,
) -> Dict:
    """
    Create visualizations for a single case.

    Supports spatial alignment when visualizing non-T2 sequences by:
    1. Finding the matching T2 series via StudyInstanceUID
    2. Computing slice mapping using DICOM z-coordinates
    3. Selecting the appropriate mask slice for each image slice

    Args:
        case_processed: Path to case in processed/
        case_seg: Path to case in processed_seg/
        output_dir: Output directory for visualizations
        max_slices: Maximum number of slices to visualize per series
        sequence_name: Name of the sequence (t2, ep2d_adc, ep2d_calc)
        t2_processed_dir: Path to T2 processed directory (for cross-sequence matching)
        dicom_base_dir: Path to DICOM base directory for this sequence
        t2_dicom_base_dir: Path to T2 DICOM base directory

    Returns:
        Statistics dict
    """
    stats = {
        "visualizations_created": 0,
        "series_processed": 0,
        "structures_found": 0,
        "series_fallback_used": 0,
        "series_missing_seg": 0,
        "series_fallback_mismatch": 0,
        "series_matched_study": 0,
        "spatial_alignment_used": 0,
    }

    case_name = case_processed.name
    class_name = case_processed.parent.name
    class_num = int(class_name.replace("class", ""))
    case_id = case_name.replace("case_", "")

    # Determine if we need spatial alignment (non-T2 sequences)
    needs_spatial_alignment = sequence_name and sequence_name != "t2"

    # Create output directory
    case_output_dir = output_dir / class_name / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # Build segmentation series index once per case
    seg_series_info = build_seg_series_index(case_seg)
    t2_case_dir = None
    if t2_processed_dir is not None:
        t2_case_dir = t2_processed_dir / class_name / case_name
    seg_series_meta = build_seg_series_meta_index(seg_series_info, t2_case_dir)

    # Find matching series directories
    for series_dir in sorted(case_processed.glob("*")):
        if not series_dir.is_dir():
            continue

        series_uid = series_dir.name

        # Get image directory
        images_dir = series_dir / "images"
        if not images_dir.exists():
            continue

        # Get list of image files
        image_files = sorted(images_dir.glob("*.png"))
        if not image_files:
            continue

        series_meta = read_meta(series_dir)
        seg_entry, match_type = select_seg_series_entry(
            series_uid,
            seg_series_info,
            len(image_files),
            series_meta=series_meta,
            seg_series_meta=seg_series_meta,
        )
        if seg_entry is None:
            stats["series_missing_seg"] += 1
            continue

        if match_type == "study_uid":
            stats["series_matched_study"] += 1
        elif match_type == "fallback":
            stats["series_fallback_used"] += 1
        diff = abs(seg_entry["mask_count"] - len(image_files))
        if diff > max(5, int(len(image_files) * 0.2)):
            stats["series_fallback_mismatch"] += 1
            seq_label = sequence_name or "sequence"
            print(
                f"⚠️  {seq_label} {class_name}/{case_name} {series_uid}: "
                f"mask count {seg_entry['mask_count']} vs images {len(image_files)}"
            )

        stats["series_processed"] += 1

        # Find all structure directories (prostate, target1, etc.)
        structure_dirs = seg_entry["structure_dirs"]
        stats["structures_found"] += len(structure_dirs)

        if not structure_dirs:
            continue

        # Compute spatial slice mapping for non-T2 sequences
        slice_mapping: Dict[int, int] = {}
        source_spatial_info: Optional[Dict] = None
        target_spatial_info: Optional[Dict] = None

        if needs_spatial_alignment and HAS_SPATIAL_LIBS:
            # Get the T2 series UID that the masks are aligned to
            t2_series_uid = seg_entry["dir"].name

            # Find DICOM directories
            source_dicom_dir = None
            target_dicom_dir = None

            if dicom_base_dir:
                source_dicom_dir = find_dicom_series_dir(
                    dicom_base_dir, class_num, case_id, series_uid
                )
                # Extract spatial info for source (ADC/Calc)
                if source_dicom_dir:
                    source_spatial_info = extract_dicom_spatial_info(source_dicom_dir)

            if t2_dicom_base_dir and t2_case_dir:
                # Find T2 series with matching study UID
                for t2_series_dir in t2_case_dir.glob("*"):
                    if t2_series_dir.is_dir():
                        t2_meta = read_meta(t2_series_dir)
                        if t2_meta.get("StudyInstanceUID") == series_meta.get(
                            "StudyInstanceUID"
                        ):
                            target_dicom_dir = find_dicom_series_dir(
                                t2_dicom_base_dir,
                                class_num,
                                case_id,
                                t2_series_dir.name,
                            )
                            # Extract spatial info for target (T2)
                            if target_dicom_dir:
                                target_spatial_info = extract_dicom_spatial_info(
                                    target_dicom_dir
                                )
                            break

            # Get T2 slice count from metadata
            t2_num_slices = seg_entry["mask_count"]
            if target_dicom_dir:
                t2_z = extract_slice_locations_from_dicom(target_dicom_dir)
                if t2_z:
                    t2_num_slices = len(t2_z)

            # Compute mapping
            slice_mapping = get_spatial_slice_mapping(
                source_meta=series_meta,
                target_meta={},
                source_dicom_dir=source_dicom_dir,
                target_dicom_dir=target_dicom_dir,
                source_num_slices=len(image_files),
                target_num_slices=t2_num_slices,
            )

            if slice_mapping:
                stats["spatial_alignment_used"] += 1

        # Sample slices evenly across the volume
        if len(image_files) > max_slices:
            indices = np.linspace(0, len(image_files) - 1, max_slices, dtype=int)
            image_files = [image_files[i] for i in indices]

        # Process each slice
        for img_file in image_files:
            slice_num = img_file.stem
            slice_idx = int(slice_num)

            # Determine which mask slice to use
            if slice_mapping and slice_idx in slice_mapping:
                mask_slice_idx = slice_mapping[slice_idx]
                mask_slice_num = f"{mask_slice_idx:04d}"
            else:
                mask_slice_num = slice_num

            # Load original image
            image = load_image(img_file)
            if image is None:
                continue

            # Create overlay with all available masks
            overlay_img = None
            structure_names = []

            for struct_dir in structure_dirs:
                struct_name = struct_dir.name
                # Use spatially mapped mask slice if available
                mask_file = struct_dir / f"{mask_slice_num}.png"

                if not mask_file.exists():
                    # Try original slice number as fallback
                    mask_file = struct_dir / f"{slice_num}.png"
                    if not mask_file.exists():
                        continue

                # Load mask
                mask = load_image(mask_file)
                if mask is None:
                    continue

                # Resample mask to match image coordinate space
                # Use spatial resampling if we have DICOM spatial info, otherwise fallback to simple resize
                if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                    if source_spatial_info and target_spatial_info:
                        # Proper spatial resampling using DICOM coordinates
                        mask = resample_mask_spatially(
                            mask,
                            source_spatial=target_spatial_info,  # Mask is in T2 space
                            target_spatial=source_spatial_info,  # Resample to ADC/Calc space
                            target_image_size=(image.shape[1], image.shape[0]),
                        )
                    else:
                        # Fallback to simple resize (may cause alignment issues)
                        mask = resample_mask_to_size(
                            mask, (image.shape[1], image.shape[0])
                        )

                # Get color for this structure
                color = STRUCTURE_COLORS.get(struct_name, STRUCTURE_COLORS["default"])

                # Create overlay
                if overlay_img is None:
                    overlay_img = create_overlay(image, mask, color)
                else:
                    overlay_img = create_overlay(overlay_img, mask, color)

                structure_names.append(struct_name)

            if overlay_img is None:
                continue

            # Create visualization figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Original MRI")
            axes[0].axis("off")

            # Overlay
            axes[1].imshow(overlay_img)
            axes[1].set_title(f'Overlay ({", ".join(structure_names)})')
            axes[1].axis("off")

            # Masks only
            mask_composite = np.zeros((*image.shape, 3), dtype=np.uint8)
            for struct_dir in structure_dirs:
                struct_name = struct_dir.name
                # Use spatially mapped mask slice if available
                mask_file = struct_dir / f"{mask_slice_num}.png"
                if not mask_file.exists():
                    mask_file = struct_dir / f"{slice_num}.png"

                if mask_file.exists():
                    mask = load_image(mask_file)
                    if mask is not None:
                        # Resample mask to match image coordinate space
                        if (
                            mask.shape[0] != image.shape[0]
                            or mask.shape[1] != image.shape[1]
                        ):
                            if source_spatial_info and target_spatial_info:
                                mask = resample_mask_spatially(
                                    mask,
                                    source_spatial=target_spatial_info,
                                    target_spatial=source_spatial_info,
                                    target_image_size=(image.shape[1], image.shape[0]),
                                )
                            else:
                                mask = resample_mask_to_size(
                                    mask, (image.shape[1], image.shape[0])
                                )

                        color = STRUCTURE_COLORS.get(
                            struct_name, STRUCTURE_COLORS["default"]
                        )
                        mask_bool = mask > 0
                        mask_composite[mask_bool] = color[:3]

            axes[2].imshow(mask_composite)
            axes[2].set_title("Masks Only")
            axes[2].axis("off")

            # Add legend
            legend_elements = []
            for struct_name in set(structure_names):
                color = STRUCTURE_COLORS.get(struct_name, STRUCTURE_COLORS["default"])
                legend_elements.append(
                    patches.Patch(
                        facecolor=np.array(color[:3]) / 255.0,
                        label=struct_name.capitalize(),
                    )
                )

            if legend_elements:
                fig.legend(
                    handles=legend_elements,
                    loc="lower center",
                    ncol=len(legend_elements),
                )

            plt.suptitle(f"{class_name}/{case_name} - Slice {slice_num}", fontsize=14)
            plt.tight_layout()

            # Save figure
            output_file = case_output_dir / f"slice_{slice_num}.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

            stats["visualizations_created"] += 1

    return stats


def main():
    """Main execution function."""

    print("=" * 80)
    print("Segmentation Mask Visualization")
    print("=" * 80)

    # Configuration
    processed_seg_dir = Path("data/processed_seg")
    output_base_dir = Path("data/visualizations")
    max_slices_per_series = 10  # Visualize up to 10 slices per series

    total_stats = {
        "visualizations_created": 0,
        "series_processed": 0,
        "structures_found": 0,
        "series_fallback_used": 0,
        "series_missing_seg": 0,
        "series_fallback_mismatch": 0,
        "series_matched_study": 0,
        "spatial_alignment_used": 0,
    }
    total_cases = 0
    sequences_processed = 0

    # Process each sequence (t2, ep2d_adc, ep2d_calc)
    for sequence_name, processed_dir in SEQUENCE_PROCESSED_DIRS.items():
        if not processed_dir.exists():
            print(f"⚠️  Skipping {sequence_name}: {processed_dir} not found")
            continue

        sequence_seg_dir = (
            processed_seg_dir.parent / f"{processed_seg_dir.name}_{sequence_name}"
        )
        if not sequence_seg_dir.exists():
            sequence_seg_dir = processed_seg_dir

        if not sequence_seg_dir.exists():
            print(f"⚠️  Skipping {sequence_name}: {sequence_seg_dir} not found")
            continue

        output_dir = (
            output_base_dir
            if sequence_name == "t2"
            else output_base_dir / sequence_name
        )

        # Find matching cases
        matching_cases = find_matching_cases(
            processed_dir, sequence_seg_dir, sequence_name
        )

        if not matching_cases:
            print(f"⚠️  No matching cases found for {sequence_name}")
            continue

        sequences_processed += 1
        total_cases += len(matching_cases)

        # Process each case with progress bar
        print(f"{'='*80}")
        print(f"Creating visualizations for {sequence_name}...")
        print(f"{'='*80}\n")

        pbar = tqdm(matching_cases, desc="Processing cases", unit="case")

        for case_processed, case_seg in pbar:
            case_name = case_processed.name
            class_name = case_processed.parent.name

            pbar.set_description(f"{class_name}/{case_name}")

            # Get DICOM directories for spatial alignment
            dicom_base = DICOM_DIRS.get(sequence_name)
            t2_dicom_base = DICOM_DIRS.get("t2")

            stats = visualize_case(
                case_processed,
                case_seg,
                output_dir,
                max_slices=max_slices_per_series,
                sequence_name=sequence_name,
                t2_processed_dir=SEQUENCE_PROCESSED_DIRS["t2"],
                dicom_base_dir=dicom_base,
                t2_dicom_base_dir=t2_dicom_base,
            )

            # Accumulate stats
            for key, value in stats.items():
                total_stats[key] += value

            pbar.set_postfix(
                {
                    "Viz": total_stats["visualizations_created"],
                    "Series": total_stats["series_processed"],
                }
            )

        pbar.close()

        print(f"\nOutput directory ({sequence_name}): {output_dir}/")

    if sequences_processed == 0:
        print("❌ Error: No sequences processed (missing inputs or no matching cases).")
        return

    # Final summary
    print(f"\n{'='*80}")
    print(f"Visualization Complete!")
    print(f"{'='*80}")
    print(f"\nSummary Statistics:")
    print(f"  Sequences processed: {sequences_processed}")
    print(f"  Cases processed: {total_cases}")
    print(f"  Series processed: {total_stats['series_processed']}")
    print(f"  Structures found: {total_stats['structures_found']}")
    print(f"  Visualizations created: {total_stats['visualizations_created']}")
    print(f"  Series matched by study UID: {total_stats['series_matched_study']}")
    print(f"  Series fallback used: {total_stats['series_fallback_used']}")
    print(f"  Series missing masks: {total_stats['series_missing_seg']}")
    print(
        f"  Fallback mismatch (slice count): {total_stats['series_fallback_mismatch']}"
    )
    print(
        f"  Spatial alignment used (ADC/Calc): {total_stats['spatial_alignment_used']}"
    )
    print(f"\nOutput base directory: {output_base_dir}/")
    print(
        f"  Structure (t2): {output_base_dir}/class{{N}}/case_{{XXXX}}/slice_{{NNNN}}.png"
    )
    print(
        f"  Structure (others): {output_base_dir}/{{sequence}}/class{{N}}/case_{{XXXX}}/slice_{{NNNN}}.png"
    )

    if HAS_SPATIAL_LIBS:
        print(f"\n✓ Spatial alignment libraries available (pydicom, SimpleITK)")
    else:
        print(f"\n⚠️  Spatial alignment libraries not available. Install with:")
        print(f"    pip install pydicom SimpleITK")


if __name__ == "__main__":
    main()
