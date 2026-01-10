#!/usr/bin/env python3
"""
Multi-Modal MRI Mapping Service

Maps and aligns different MRI sequences (T2, ADC, Calc) with segmentation masks
for multi-channel deep learning training.

Features:
1. Links sequences by StudyInstanceUID
2. Extracts per-slice spatial coordinates from DICOM
3. Resamples ADC/Calc to T2 resolution
4. Creates aligned multi-channel output

Usage:
    # Generate mapping for all cases
    python service/mapping.py --all

    # Process specific class
    python service/mapping.py --class 2

    # Validate existing mapping
    python service/mapping.py --validate

    # Dry run (show what would be mapped)
    python service/mapping.py --dry-run
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
from PIL import Image
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MappingConfig:
    """Configuration for multi-modal mapping."""
    # Input directories
    t2_processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    adc_processed_dir: Path = field(default_factory=lambda: Path("data/processed_ep2d_adc"))
    calc_processed_dir: Path = field(default_factory=lambda: Path("data/processed_ep2d_calc"))
    seg_processed_dir: Path = field(default_factory=lambda: Path("data/processed_seg"))
    
    # DICOM source directories (for spatial metadata extraction)
    t2_dicom_dir: Path = field(default_factory=lambda: Path("data/nbia"))
    adc_dicom_dir: Path = field(default_factory=lambda: Path("data/nbia_ep2d_adc"))
    calc_dicom_dir: Path = field(default_factory=lambda: Path("data/nbia_ep2d_calc"))
    
    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("data/aligned"))
    
    # Processing options
    resample_to_t2: bool = True
    interpolation: str = "linear"  # "linear" or "nearest"
    fill_missing_slices: bool = True
    save_nifti: bool = False


# =============================================================================
# Data Classes
# =============================================================================

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


@dataclass
class CaseMapping:
    """Mapping information for a single case."""
    case_id: str
    class_num: int
    study_uid: str
    t2: Optional[SeriesInfo] = None
    adc: Optional[SeriesInfo] = None
    calc: Optional[SeriesInfo] = None
    seg_series_uid: Optional[str] = None
    slice_mapping: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    is_complete: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "case_id": self.case_id,
            "class_num": self.class_num,
            "study_uid": self.study_uid,
            "t2_series_uid": self.t2.series_uid if self.t2 else None,
            "adc_series_uid": self.adc.series_uid if self.adc else None,
            "calc_series_uid": self.calc.series_uid if self.calc else None,
            "seg_series_uid": self.seg_series_uid,
            "t2_num_slices": self.t2.num_slices if self.t2 else 0,
            "adc_num_slices": self.adc.num_slices if self.adc else 0,
            "calc_num_slices": self.calc.num_slices if self.calc else 0,
            "t2_spacing": list(self.t2.spacing) if self.t2 else None,
            "adc_spacing": list(self.adc.spacing) if self.adc else None,
            "adc_rescale": {"slope": self.adc.rescale_slope, "intercept": self.adc.rescale_intercept} if self.adc else None,
            "calc_rescale": {"slope": self.calc.rescale_slope, "intercept": self.calc.rescale_intercept} if self.calc else None,
            "slice_mapping": self.slice_mapping,
            "is_complete": self.is_complete,
        }


# =============================================================================
# Spatial Metadata Extraction
# =============================================================================

def extract_slice_locations_from_dicom(dicom_dir: Path) -> Tuple[List[float], Dict]:
    """
    Extract per-slice z-positions and rescaling metadata from DICOM files.
    
    Args:
        dicom_dir: Directory containing DICOM series
    
    Returns:
        Tuple of (sorted z-positions list, metadata dict)
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
                
        except Exception as e:
            continue
    
    # Sort and deduplicate
    slice_locations = sorted(set(slice_locations))
    
    return slice_locations, meta


def load_series_info_from_meta(meta_path: Path, class_num: int) -> Optional[SeriesInfo]:
    """Load series info from existing meta.json file."""
    if not meta_path.exists():
        return None
    
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        
        # Count slices from images directory
        images_dir = meta_path.parent / "images"
        num_slices = len(list(images_dir.glob("*.png"))) if images_dir.exists() else meta.get("num_slices", 0)
        
        # Extract case_id from path
        case_dir = meta_path.parent.parent
        case_id = case_dir.name.replace("case_", "")
        
        return SeriesInfo(
            series_uid=meta.get("SeriesInstanceUID", ""),
            study_uid=meta.get("StudyInstanceUID", ""),
            patient_id=meta.get("PatientID", ""),
            case_id=case_id,
            class_num=class_num,
            num_slices=num_slices,
            spacing=(1.0, 1.0, 1.0),  # Will be updated from DICOM
            origin=(0.0, 0.0, 0.0),
            size=(256, 256, num_slices),
            processed_dir=meta_path.parent,
        )
    except Exception as e:
        logger.warning(f"Failed to load meta from {meta_path}: {e}")
        return None


# =============================================================================
# Mapping Builder
# =============================================================================

class MultiModalMapper:
    """Builds and manages multi-modal case mappings."""
    
    def __init__(self, config: MappingConfig):
        self.config = config
        self.mappings: Dict[str, CaseMapping] = {}  # key: "class{N}_case_{XXXX}"
    
    def discover_cases(self) -> Dict[str, Dict[str, Path]]:
        """
        Discover all cases across all modalities.
        
        Returns:
            Dict mapping case_key to {modality: processed_dir}
        """
        cases = defaultdict(dict)
        
        def add_case(key: str, class_num: int, case_id: str, modality: str, path: Path):
            """Helper to ensure case_id and class_num are always set."""
            cases[key][modality] = path
            # Always set case_id and class_num (first discovery wins, but they should be the same)
            if "class_num" not in cases[key]:
                cases[key]["class_num"] = class_num
            if "case_id" not in cases[key]:
                cases[key]["case_id"] = case_id
        
        # T2 cases
        for class_dir in self.config.t2_processed_dir.glob("class*"):
            class_num = int(class_dir.name.replace("class", ""))
            for case_dir in class_dir.glob("case_*"):
                case_id = case_dir.name.replace("case_", "")
                key = f"class{class_num}_case_{case_id}"
                add_case(key, class_num, case_id, "t2", case_dir)
        
        # ADC cases
        for class_dir in self.config.adc_processed_dir.glob("class*"):
            class_num = int(class_dir.name.replace("class", ""))
            for case_dir in class_dir.glob("case_*"):
                case_id = case_dir.name.replace("case_", "")
                key = f"class{class_num}_case_{case_id}"
                add_case(key, class_num, case_id, "adc", case_dir)
        
        # Calc cases
        for class_dir in self.config.calc_processed_dir.glob("class*"):
            class_num = int(class_dir.name.replace("class", ""))
            for case_dir in class_dir.glob("case_*"):
                case_id = case_dir.name.replace("case_", "")
                key = f"class{class_num}_case_{case_id}"
                add_case(key, class_num, case_id, "calc", case_dir)
        
        # Seg cases
        for class_dir in self.config.seg_processed_dir.glob("class*"):
            class_num = int(class_dir.name.replace("class", ""))
            for case_dir in class_dir.glob("case_*"):
                case_id = case_dir.name.replace("case_", "")
                key = f"class{class_num}_case_{case_id}"
                add_case(key, class_num, case_id, "seg", case_dir)
        
        return dict(cases)
    
    def build_series_info(self, case_dir: Path, class_num: int, 
                          dicom_base_dir: Path) -> List[SeriesInfo]:
        """Build SeriesInfo for all series in a case directory."""
        series_list = []
        
        for series_dir in case_dir.glob("*"):
            if not series_dir.is_dir():
                continue
            
            meta_path = series_dir / "meta.json"
            info = load_series_info_from_meta(meta_path, class_num)
            
            if info:
                # Try to find corresponding DICOM directory for spatial info
                info.processed_dir = series_dir
                series_list.append(info)
        
        return series_list
    
    def find_dicom_series_dir(self, dicom_base: Path, class_num: int, 
                               case_id: str, series_uid: str) -> Optional[Path]:
        """Find DICOM directory for a specific series."""
        class_dir = dicom_base / f"class{class_num}"
        if not class_dir.exists():
            return None
        
        # Search through manifest directories
        for manifest_dir in class_dir.glob("manifest-*"):
            # Look for case directory
            for patient_dir in manifest_dir.rglob(f"*{case_id}*"):
                if patient_dir.is_dir():
                    # Search for series directory
                    for study_dir in patient_dir.iterdir():
                        if study_dir.is_dir():
                            for series_dir in study_dir.iterdir():
                                if series_dir.is_dir():
                                    # Check if this is the right series
                                    dcm_files = list(series_dir.glob("*.dcm"))
                                    if dcm_files:
                                        try:
                                            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                                            if getattr(ds, "SeriesInstanceUID", "") == series_uid:
                                                return series_dir
                                        except:
                                            pass
        return None
    
    def compute_slice_mapping(self, t2_info: SeriesInfo, 
                              other_info: SeriesInfo) -> Dict[int, Dict]:
        """
        Compute slice-by-slice mapping between T2 and another modality.
        
        Args:
            t2_info: T2 series info with slice locations
            other_info: ADC/Calc series info with slice locations
        
        Returns:
            Dict mapping T2 slice index to {z_position, other_slice_idx, distance}
        """
        mapping = {}
        
        if not t2_info.slice_locations or not other_info.slice_locations:
            # Fallback to proportional mapping
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
    
    def build_case_mapping(self, case_key: str, case_data: Dict) -> CaseMapping:
        """Build complete mapping for a single case."""
        case_id = case_data["case_id"]
        class_num = case_data["class_num"]
        
        mapping = CaseMapping(
            case_id=case_id,
            class_num=class_num,
            study_uid=""
        )
        
        # Load T2 series info
        if "t2" in case_data:
            t2_series_list = self.build_series_info(
                case_data["t2"], class_num, self.config.t2_dicom_dir
            )
            if t2_series_list:
                mapping.t2 = t2_series_list[0]  # Take first series
                mapping.study_uid = mapping.t2.study_uid
                
                # Extract slice locations from DICOM if available
                dicom_dir = self.find_dicom_series_dir(
                    self.config.t2_dicom_dir, class_num, case_id, mapping.t2.series_uid
                )
                if dicom_dir:
                    slice_locs, meta = extract_slice_locations_from_dicom(dicom_dir)
                    mapping.t2.slice_locations = slice_locs
                    mapping.t2.dicom_dir = dicom_dir
                    if "PixelSpacing" in meta:
                        ps = meta["PixelSpacing"]
                        st = meta.get("SliceThickness", 1.0)
                        mapping.t2.spacing = (float(ps[0]), float(ps[1]), float(st))
                    if "Origin" in meta:
                        mapping.t2.origin = tuple(meta["Origin"])
                    mapping.t2.rescale_slope = meta.get("RescaleSlope", 1.0)
                    mapping.t2.rescale_intercept = meta.get("RescaleIntercept", 0.0)
        
        # Load ADC series info
        if "adc" in case_data:
            adc_series_list = self.build_series_info(
                case_data["adc"], class_num, self.config.adc_dicom_dir
            )
            # Match by StudyInstanceUID
            for series in adc_series_list:
                if series.study_uid == mapping.study_uid or not mapping.study_uid:
                    mapping.adc = series
                    
                    # Extract slice locations
                    dicom_dir = self.find_dicom_series_dir(
                        self.config.adc_dicom_dir, class_num, case_id, series.series_uid
                    )
                    if dicom_dir:
                        slice_locs, meta = extract_slice_locations_from_dicom(dicom_dir)
                        mapping.adc.slice_locations = slice_locs
                        mapping.adc.dicom_dir = dicom_dir
                        if "PixelSpacing" in meta:
                            ps = meta["PixelSpacing"]
                            st = meta.get("SliceThickness", 1.0)
                            mapping.adc.spacing = (float(ps[0]), float(ps[1]), float(st))
                        if "Origin" in meta:
                            mapping.adc.origin = tuple(meta["Origin"])
                        mapping.adc.rescale_slope = meta.get("RescaleSlope", 1.0)
                        mapping.adc.rescale_intercept = meta.get("RescaleIntercept", 0.0)
                    break
        
        # Load Calc series info
        if "calc" in case_data:
            calc_series_list = self.build_series_info(
                case_data["calc"], class_num, self.config.calc_dicom_dir
            )
            for series in calc_series_list:
                if series.study_uid == mapping.study_uid or not mapping.study_uid:
                    mapping.calc = series
                    
                    dicom_dir = self.find_dicom_series_dir(
                        self.config.calc_dicom_dir, class_num, case_id, series.series_uid
                    )
                    if dicom_dir:
                        slice_locs, meta = extract_slice_locations_from_dicom(dicom_dir)
                        mapping.calc.slice_locations = slice_locs
                        mapping.calc.dicom_dir = dicom_dir
                        if "PixelSpacing" in meta:
                            ps = meta["PixelSpacing"]
                            st = meta.get("SliceThickness", 1.0)
                            mapping.calc.spacing = (float(ps[0]), float(ps[1]), float(st))
                        if "Origin" in meta:
                            mapping.calc.origin = tuple(meta["Origin"])
                        mapping.calc.rescale_slope = meta.get("RescaleSlope", 1.0)
                        mapping.calc.rescale_intercept = meta.get("RescaleIntercept", 0.0)
                    break
        
        # Load segmentation info
        if "seg" in case_data:
            seg_dir = case_data["seg"]
            # Find series directory that matches T2
            for series_dir in seg_dir.glob("*"):
                if series_dir.is_dir() and mapping.t2:
                    if series_dir.name == mapping.t2.series_uid:
                        mapping.seg_series_uid = series_dir.name
                        break
            # Fallback: use first available
            if not mapping.seg_series_uid:
                for series_dir in seg_dir.glob("*"):
                    if series_dir.is_dir():
                        mapping.seg_series_uid = series_dir.name
                        break
        
        # Compute slice mappings
        if mapping.t2:
            if mapping.adc:
                adc_mapping = self.compute_slice_mapping(mapping.t2, mapping.adc)
                for idx, m in adc_mapping.items():
                    if idx not in mapping.slice_mapping:
                        mapping.slice_mapping[idx] = {}
                    mapping.slice_mapping[idx]["adc"] = m
            
            if mapping.calc:
                calc_mapping = self.compute_slice_mapping(mapping.t2, mapping.calc)
                for idx, m in calc_mapping.items():
                    if idx not in mapping.slice_mapping:
                        mapping.slice_mapping[idx] = {}
                    mapping.slice_mapping[idx]["calc"] = m
        
        # Check completeness
        mapping.is_complete = (
            mapping.t2 is not None and
            mapping.adc is not None and
            mapping.calc is not None and
            mapping.seg_series_uid is not None
        )
        
        return mapping
    
    def build_all_mappings(self, class_filter: Optional[int] = None) -> Dict[str, CaseMapping]:
        """Build mappings for all discovered cases."""
        cases = self.discover_cases()
        
        logger.info(f"Discovered {len(cases)} cases across all modalities")
        
        for case_key, case_data in tqdm(cases.items(), desc="Building mappings"):
            if class_filter is not None:
                if case_data.get("class_num") != class_filter:
                    continue
            
            mapping = self.build_case_mapping(case_key, case_data)
            self.mappings[case_key] = mapping
        
        return self.mappings


# =============================================================================
# Resampling and Alignment
# =============================================================================

class VolumeResampler:
    """Handles resampling of volumes to a reference grid."""
    
    def __init__(self, config: MappingConfig):
        self.config = config
    
    def load_png_series_as_sitk(self, images_dir: Path, 
                                 spacing: Tuple[float, float, float],
                                 origin: Tuple[float, float, float] = (0, 0, 0)) -> sitk.Image:
        """Load PNG series as SimpleITK volume."""
        png_files = sorted(images_dir.glob("*.png"))
        if not png_files:
            raise ValueError(f"No PNG files found in {images_dir}")
        
        # Load all slices
        slices = []
        for png_file in png_files:
            img = np.array(Image.open(png_file))
            slices.append(img)
        
        # Stack into 3D volume (Z, Y, X)
        volume = np.stack(slices, axis=0)
        
        # Convert to SimpleITK
        sitk_img = sitk.GetImageFromArray(volume)
        sitk_img.SetSpacing(spacing)
        sitk_img.SetOrigin(origin)
        
        return sitk_img
    
    def resample_to_reference(self, moving: sitk.Image, 
                               reference: sitk.Image,
                               interpolator: int = sitk.sitkLinear) -> sitk.Image:
        """Resample moving image to reference geometry."""
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform())
        
        return resampler.Execute(moving)
    
    def save_sitk_as_pngs(self, sitk_img: sitk.Image, output_dir: Path):
        """Save SimpleITK volume as PNG series."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        volume = sitk.GetArrayFromImage(sitk_img)  # (Z, Y, X)
        
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            # Normalize to 0-255 if needed
            if slice_img.max() > 255:
                slice_img = ((slice_img - slice_img.min()) / 
                            (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
            else:
                slice_img = slice_img.astype(np.uint8)
            
            Image.fromarray(slice_img).save(output_dir / f"{i:04d}.png")
    
    def align_case(self, mapping: CaseMapping, output_base: Path) -> bool:
        """
        Create aligned multi-channel output for a case.
        
        Args:
            mapping: CaseMapping with all series info
            output_base: Base output directory
        
        Returns:
            True if successful
        """
        if not mapping.t2 or not mapping.t2.processed_dir:
            logger.warning(f"No T2 data for case {mapping.case_id}")
            return False
        
        # Create output directory
        output_dir = output_base / f"class{mapping.class_num}" / f"case_{mapping.case_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load T2 as reference
            t2_images_dir = mapping.t2.processed_dir / "images"
            t2_volume = self.load_png_series_as_sitk(
                t2_images_dir, 
                mapping.t2.spacing,
                mapping.t2.origin
            )
            
            # Save T2 (already in correct format)
            self.save_sitk_as_pngs(t2_volume, output_dir / "t2")
            
            # Resample and save ADC
            if mapping.adc and mapping.adc.processed_dir:
                adc_images_dir = mapping.adc.processed_dir / "images"
                adc_volume = self.load_png_series_as_sitk(
                    adc_images_dir,
                    mapping.adc.spacing,
                    mapping.adc.origin
                )
                
                # Resample to T2 grid
                adc_resampled = self.resample_to_reference(adc_volume, t2_volume)
                self.save_sitk_as_pngs(adc_resampled, output_dir / "adc")
            else:
                # Create empty ADC if missing
                empty_volume = sitk.Image(t2_volume.GetSize(), sitk.sitkUInt8)
                empty_volume.CopyInformation(t2_volume)
                self.save_sitk_as_pngs(empty_volume, output_dir / "adc")
            
            # Resample and save Calc
            if mapping.calc and mapping.calc.processed_dir:
                calc_images_dir = mapping.calc.processed_dir / "images"
                calc_volume = self.load_png_series_as_sitk(
                    calc_images_dir,
                    mapping.calc.spacing,
                    mapping.calc.origin
                )
                
                calc_resampled = self.resample_to_reference(calc_volume, t2_volume)
                self.save_sitk_as_pngs(calc_resampled, output_dir / "calc")
            else:
                empty_volume = sitk.Image(t2_volume.GetSize(), sitk.sitkUInt8)
                empty_volume.CopyInformation(t2_volume)
                self.save_sitk_as_pngs(empty_volume, output_dir / "calc")
            
            # Copy masks (already aligned to T2)
            if mapping.seg_series_uid:
                seg_base = (self.config.seg_processed_dir / 
                           f"class{mapping.class_num}" / 
                           f"case_{mapping.case_id}" /
                           mapping.seg_series_uid)
                
                for struct_dir in seg_base.glob("*"):
                    if struct_dir.is_dir():
                        out_mask_dir = output_dir / f"mask_{struct_dir.name}"
                        out_mask_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Copy mask PNGs, filling missing slices with zeros
                        for i in range(mapping.t2.num_slices):
                            src_mask = struct_dir / f"{i:04d}.png"
                            dst_mask = out_mask_dir / f"{i:04d}.png"
                            
                            if src_mask.exists():
                                # Copy existing mask
                                mask = np.array(Image.open(src_mask))
                                # Resize if needed
                                if mask.shape != (t2_volume.GetSize()[1], t2_volume.GetSize()[0]):
                                    mask_img = Image.fromarray(mask)
                                    mask_img = mask_img.resize(
                                        (t2_volume.GetSize()[0], t2_volume.GetSize()[1]),
                                        Image.NEAREST
                                    )
                                    mask = np.array(mask_img)
                                Image.fromarray(mask).save(dst_mask)
                            else:
                                # Create empty mask
                                empty_mask = np.zeros(
                                    (t2_volume.GetSize()[1], t2_volume.GetSize()[0]),
                                    dtype=np.uint8
                                )
                                Image.fromarray(empty_mask).save(dst_mask)
            
            # Save mapping JSON
            mapping_json = output_dir / "mapping.json"
            with open(mapping_json, 'w') as f:
                json.dump(mapping.to_dict(), f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to align case {mapping.case_id}: {e}")
            return False


# =============================================================================
# Main Pipeline
# =============================================================================

class MappingPipeline:
    """Main pipeline for multi-modal mapping."""
    
    def __init__(self, config: MappingConfig):
        self.config = config
        self.mapper = MultiModalMapper(config)
        self.resampler = VolumeResampler(config)
    
    def run(self, class_filter: Optional[int] = None, 
            dry_run: bool = False,
            skip_alignment: bool = False) -> Dict:
        """
        Run the full mapping pipeline.
        
        Args:
            class_filter: Only process this class (1-4)
            dry_run: Only show what would be done
            skip_alignment: Skip the resampling/alignment step
        
        Returns:
            Statistics dict
        """
        stats = {
            "cases_discovered": 0,
            "cases_complete": 0,
            "cases_partial": 0,
            "cases_aligned": 0,
            "cases_failed": 0,
        }
        
        # Build mappings
        logger.info("Building multi-modal mappings...")
        mappings = self.mapper.build_all_mappings(class_filter)
        stats["cases_discovered"] = len(mappings)
        
        for key, mapping in mappings.items():
            if mapping.is_complete:
                stats["cases_complete"] += 1
            else:
                stats["cases_partial"] += 1
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("MAPPING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total cases: {stats['cases_discovered']}")
        logger.info(f"Complete (T2+ADC+Calc+Seg): {stats['cases_complete']}")
        logger.info(f"Partial: {stats['cases_partial']}")
        
        if dry_run:
            logger.info("\n[DRY RUN] Skipping alignment step")
            self._print_sample_mappings(mappings)
            return stats
        
        # Save mapping index
        mapping_index_path = self.config.output_dir / "mapping_index.json"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            key: mapping.to_dict() 
            for key, mapping in mappings.items()
        }
        with open(mapping_index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"\nSaved mapping index to: {mapping_index_path}")
        
        if skip_alignment:
            logger.info("Skipping alignment step (--skip-alignment)")
            return stats
        
        # Run alignment
        logger.info(f"\n{'='*60}")
        logger.info("ALIGNING VOLUMES")
        logger.info(f"{'='*60}")
        
        for key, mapping in tqdm(mappings.items(), desc="Aligning cases"):
            if not mapping.t2:
                continue
            
            success = self.resampler.align_case(mapping, self.config.output_dir)
            if success:
                stats["cases_aligned"] += 1
            else:
                stats["cases_failed"] += 1
        
        logger.info(f"\n{'='*60}")
        logger.info("ALIGNMENT COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Cases aligned: {stats['cases_aligned']}")
        logger.info(f"Cases failed: {stats['cases_failed']}")
        logger.info(f"Output directory: {self.config.output_dir}")
        
        return stats
    
    def _print_sample_mappings(self, mappings: Dict[str, CaseMapping], n: int = 3):
        """Print sample mappings for inspection."""
        logger.info(f"\nSample mappings (first {n}):")
        
        for i, (key, mapping) in enumerate(mappings.items()):
            if i >= n:
                break
            
            logger.info(f"\n  {key}:")
            logger.info(f"    StudyUID: {mapping.study_uid[:40]}...")
            logger.info(f"    T2: {mapping.t2.num_slices if mapping.t2 else 'N/A'} slices")
            logger.info(f"    ADC: {mapping.adc.num_slices if mapping.adc else 'N/A'} slices")
            logger.info(f"    Calc: {mapping.calc.num_slices if mapping.calc else 'N/A'} slices")
            logger.info(f"    Seg: {'Yes' if mapping.seg_series_uid else 'No'}")
            logger.info(f"    Complete: {mapping.is_complete}")
    
    def validate(self) -> bool:
        """Validate existing aligned data."""
        if not self.config.output_dir.exists():
            logger.error(f"Output directory not found: {self.config.output_dir}")
            return False
        
        issues = []
        valid_cases = 0
        
        for class_dir in self.config.output_dir.glob("class*"):
            for case_dir in class_dir.glob("case_*"):
                # Check required directories
                t2_dir = case_dir / "t2"
                adc_dir = case_dir / "adc"
                calc_dir = case_dir / "calc"
                
                if not t2_dir.exists():
                    issues.append(f"{case_dir.name}: Missing t2/")
                    continue
                
                t2_count = len(list(t2_dir.glob("*.png")))
                
                if adc_dir.exists():
                    adc_count = len(list(adc_dir.glob("*.png")))
                    if adc_count != t2_count:
                        issues.append(f"{case_dir.name}: ADC slice count mismatch ({adc_count} vs {t2_count})")
                
                if calc_dir.exists():
                    calc_count = len(list(calc_dir.glob("*.png")))
                    if calc_count != t2_count:
                        issues.append(f"{case_dir.name}: Calc slice count mismatch ({calc_count} vs {t2_count})")
                
                valid_cases += 1
        
        if issues:
            logger.warning(f"Found {len(issues)} issues:")
            for issue in issues[:10]:
                logger.warning(f"  - {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more")
        
        logger.info(f"Valid cases: {valid_cases}")
        return len(issues) == 0


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal MRI Mapping Service"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all cases"
    )
    parser.add_argument(
        "--class",
        dest="class_num",
        type=int,
        choices=[1, 2, 3, 4],
        help="Process specific class only"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing"
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Skip the resampling/alignment step (only build mapping)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing aligned data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/aligned",
        help="Output directory (default: data/aligned)"
    )
    
    args = parser.parse_args()
    
    config = MappingConfig(output_dir=Path(args.output))
    pipeline = MappingPipeline(config)
    
    if args.validate:
        success = pipeline.validate()
        return 0 if success else 1
    
    if args.all or args.class_num:
        stats = pipeline.run(
            class_filter=args.class_num,
            dry_run=args.dry_run,
            skip_alignment=args.skip_alignment
        )
        return 0
    
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

