#!/usr/bin/env python3
"""
Generate training metadata for 2.5D multi-modal segmentation.

This script scans the aligned_v2 directory and creates a comprehensive
metadata.json file for PyTorch DataLoader consumption.

Output structure:
- Global and per-case intensity statistics
- Slice-level samples with T2 context indices (edge replication for boundaries)
- Pre-computed label presence flags

Usage:
    python tools/generate_training_metadata.py
    python tools/generate_training_metadata.py --data-dir /path/to/aligned_v2
    python tools/generate_training_metadata.py --t2-context 7  # Use 7 T2 slices instead of 5
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_sorted_png_files(directory: Path) -> list[str]:
    """Get sorted list of PNG filenames in a directory."""
    if not directory.exists():
        return []
    files = sorted([f.name for f in directory.glob("*.png")])
    return files


def compute_image_stats(image_paths: list[Path]) -> dict[str, float]:
    """Compute intensity statistics from a list of image paths."""
    all_pixels = []
    for path in image_paths:
        if path.exists():
            img = np.array(Image.open(path))
            all_pixels.append(img.flatten())
    
    if not all_pixels:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "p1": 0, "p99": 0}
    
    all_pixels = np.concatenate(all_pixels)
    return {
        "mean": float(np.mean(all_pixels)),
        "std": float(np.std(all_pixels)),
        "min": int(np.min(all_pixels)),
        "max": int(np.max(all_pixels)),
        "p1": float(np.percentile(all_pixels, 1)),
        "p99": float(np.percentile(all_pixels, 99)),
    }


def compute_case_stats(case_dir: Path, modalities: list[str]) -> dict[str, dict]:
    """Compute per-modality statistics for a single case."""
    stats = {}
    for modality in modalities:
        mod_dir = case_dir / modality
        if mod_dir.exists():
            image_paths = list(mod_dir.glob("*.png"))
            stats[modality] = compute_image_stats(image_paths)
    return stats


def check_mask_positive(mask_path: Path, positive_value: int = 255) -> bool:
    """Check if a mask contains any positive labels."""
    if not mask_path.exists():
        return False
    mask = np.array(Image.open(mask_path))
    return np.any(mask >= positive_value)


def get_t2_context_indices(
    center_idx: int, 
    num_slices: int, 
    context_size: int = 5
) -> list[int]:
    """
    Get T2 context slice indices with edge replication padding.
    
    Args:
        center_idx: The center slice index
        num_slices: Total number of slices in the volume
        context_size: Number of context slices (must be odd)
    
    Returns:
        List of slice indices for the context window
    """
    half = context_size // 2
    indices = []
    
    for offset in range(-half, half + 1):
        idx = center_idx + offset
        # Edge replication: clamp to valid range
        idx = max(0, min(num_slices - 1, idx))
        indices.append(idx)
    
    return indices


def process_case(
    case_dir: Path,
    case_id: str,
    class_num: int,
    modalities: list[str],
    masks: list[str],
    t2_context_size: int,
    positive_value: int,
) -> tuple[dict, list[dict]]:
    """
    Process a single case and return case info and sample list.
    
    Returns:
        Tuple of (case_info dict, list of sample dicts)
    """
    # Get slice files from t2 directory (reference)
    t2_dir = case_dir / "t2"
    slice_files = get_sorted_png_files(t2_dir)
    num_slices = len(slice_files)
    
    if num_slices == 0:
        logger.warning(f"No slices found in {case_dir}")
        return None, []
    
    # Check modality availability
    modality_available = {}
    for mod in modalities:
        mod_files = get_sorted_png_files(case_dir / mod)
        modality_available[mod] = len(mod_files) == num_slices
        if len(mod_files) != num_slices and len(mod_files) > 0:
            logger.warning(
                f"Slice count mismatch in {case_id}: "
                f"t2={num_slices}, {mod}={len(mod_files)}"
            )
        elif len(mod_files) == 0:
            logger.warning(f"Missing {mod} data in {case_id}")
    
    # Compute per-case statistics
    case_stats = compute_case_stats(case_dir, modalities)
    
    # Find slices with positive labels
    slices_with_prostate = []
    slices_with_target = []
    
    for idx, filename in enumerate(slice_files):
        prostate_path = case_dir / "mask_prostate" / filename
        target_path = case_dir / "mask_target1" / filename
        
        if check_mask_positive(prostate_path, positive_value):
            slices_with_prostate.append(idx)
        if check_mask_positive(target_path, positive_value):
            slices_with_target.append(idx)
    
    # Build case info
    case_info = {
        "class": class_num,
        "num_slices": num_slices,
        "has_adc": modality_available.get("adc", False),
        "has_calc": modality_available.get("calc", False),
        "stats": case_stats,
        "slices_with_prostate": slices_with_prostate,
        "slices_with_target": slices_with_target,
    }
    
    # Build sample list
    samples = []
    for idx, filename in enumerate(slice_files):
        # Extract slice number from filename (e.g., "0030.png" -> 30)
        slice_num = int(filename.replace(".png", ""))
        
        sample = {
            "sample_id": f"{case_id}/slice_{slice_num:04d}",
            "case_id": case_id,
            "class": class_num,
            "slice_idx": idx,
            "slice_num": slice_num,  # Original slice number from filename
            "files": {
                "t2": filename,
                "adc": filename,
                "calc": filename,
                "mask_prostate": filename,
                "mask_target1": filename,
            },
            "t2_context_indices": get_t2_context_indices(idx, num_slices, t2_context_size),
            "has_adc": modality_available.get("adc", False),
            "has_calc": modality_available.get("calc", False),
            "has_prostate": idx in slices_with_prostate,
            "has_target": idx in slices_with_target,
        }
        samples.append(sample)
    
    return case_info, samples


def compute_global_stats(
    data_dir: Path,
    case_ids: list[str],
    modalities: list[str],
    sample_ratio: float = 0.1,
) -> dict[str, dict]:
    """
    Compute global intensity statistics across all cases.
    
    Args:
        data_dir: Base data directory
        case_ids: List of case IDs to process
        modalities: List of modality names
        sample_ratio: Ratio of images to sample for efficiency (0.1 = 10%)
    
    Returns:
        Dict of modality -> stats
    """
    logger.info("Computing global intensity statistics...")
    
    global_stats = {}
    
    for modality in modalities:
        logger.info(f"  Processing {modality}...")
        all_pixels = []
        
        for case_id in tqdm(case_ids, desc=f"  {modality}", leave=False):
            mod_dir = data_dir / case_id / modality
            if not mod_dir.exists():
                continue
            
            png_files = list(mod_dir.glob("*.png"))
            
            # Sample a subset for efficiency
            if sample_ratio < 1.0:
                n_sample = max(1, int(len(png_files) * sample_ratio))
                sampled_files = np.random.choice(png_files, n_sample, replace=False)
            else:
                sampled_files = png_files
            
            for path in sampled_files:
                img = np.array(Image.open(path))
                all_pixels.append(img.flatten())
        
        if all_pixels:
            all_pixels = np.concatenate(all_pixels)
            global_stats[modality] = {
                "mean": float(np.mean(all_pixels)),
                "std": float(np.std(all_pixels)),
                "min": int(np.min(all_pixels)),
                "max": int(np.max(all_pixels)),
                "p1": float(np.percentile(all_pixels, 1)),
                "p99": float(np.percentile(all_pixels, 99)),
            }
    
    return global_stats


def generate_metadata(
    data_dir: Path,
    output_path: Path,
    t2_context_size: int = 5,
    modalities: list[str] | None = None,
    masks: list[str] | None = None,
    positive_value: int = 255,
) -> dict[str, Any]:
    """
    Generate comprehensive training metadata.
    
    Args:
        data_dir: Path to aligned_v2 directory
        output_path: Path to output metadata.json
        t2_context_size: Number of T2 context slices (must be odd)
        modalities: List of modality names
        masks: List of mask names
        positive_value: Pixel value indicating positive label in masks
    
    Returns:
        Generated metadata dict
    """
    if modalities is None:
        modalities = ["t2", "adc", "calc"]
    if masks is None:
        masks = ["mask_prostate", "mask_target1"]
    
    if t2_context_size % 2 == 0:
        raise ValueError("t2_context_size must be odd")
    
    logger.info(f"Scanning data directory: {data_dir}")
    
    # Discover all cases
    case_dirs = []
    for class_dir in sorted(data_dir.glob("class*")):
        if not class_dir.is_dir():
            continue
        class_num = int(class_dir.name.replace("class", ""))
        for case_dir in sorted(class_dir.glob("case_*")):
            if not case_dir.is_dir():
                continue
            case_id = f"{class_dir.name}/{case_dir.name}"
            case_dirs.append((case_dir, case_id, class_num))
    
    logger.info(f"Found {len(case_dirs)} cases")
    
    # Process all cases
    cases = {}
    all_samples = []
    class_distribution = {}
    
    for case_dir, case_id, class_num in tqdm(case_dirs, desc="Processing cases"):
        case_info, samples = process_case(
            case_dir=case_dir,
            case_id=case_id,
            class_num=class_num,
            modalities=modalities,
            masks=masks,
            t2_context_size=t2_context_size,
            positive_value=positive_value,
        )
        
        if case_info is not None:
            cases[case_id] = case_info
            all_samples.extend(samples)
            
            class_key = str(class_num)
            class_distribution[class_key] = class_distribution.get(class_key, 0) + 1
    
    # Compute global statistics
    case_ids = list(cases.keys())
    global_stats = compute_global_stats(data_dir, case_ids, modalities)
    
    # Compute summary statistics
    samples_with_prostate = sum(1 for s in all_samples if s["has_prostate"])
    samples_with_target = sum(1 for s in all_samples if s["has_target"])
    samples_with_adc = sum(1 for s in all_samples if s["has_adc"])
    samples_with_calc = sum(1 for s in all_samples if s["has_calc"])
    samples_complete = sum(1 for s in all_samples if s["has_adc"] and s["has_calc"])
    
    cases_with_adc = sum(1 for c in cases.values() if c["has_adc"])
    cases_with_calc = sum(1 for c in cases.values() if c["has_calc"])
    cases_complete = sum(1 for c in cases.values() if c["has_adc"] and c["has_calc"])
    
    # Determine image size from first sample
    if all_samples:
        first_case_id = all_samples[0]["case_id"]
        first_file = all_samples[0]["files"]["t2"]
        sample_img_path = data_dir / first_case_id / "t2" / first_file
        if sample_img_path.exists():
            sample_img = Image.open(sample_img_path)
            input_size = list(sample_img.size)  # (width, height)
        else:
            input_size = [256, 256]
    else:
        input_size = [256, 256]
    
    # Build final metadata
    metadata = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "config": {
            "input_size": input_size,
            "t2_context_window": t2_context_size,
            "boundary_padding": "edge_replicate",
            "modalities": modalities,
            "masks": masks,
            "mask_positive_value": positive_value,
        },
        "global_stats": global_stats,
        "cases": cases,
        "samples": all_samples,
        "summary": {
            "total_cases": len(cases),
            "total_samples": len(all_samples),
            "cases_with_adc": cases_with_adc,
            "cases_with_calc": cases_with_calc,
            "cases_complete": cases_complete,
            "samples_with_adc": samples_with_adc,
            "samples_with_calc": samples_with_calc,
            "samples_complete": samples_complete,
            "samples_with_prostate": samples_with_prostate,
            "samples_with_target": samples_with_target,
            "class_distribution": class_distribution,
        },
    }
    
    # Save to file
    logger.info(f"Saving metadata to {output_path}")
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("Metadata Generation Complete")
    logger.info("=" * 50)
    logger.info(f"Total cases: {metadata['summary']['total_cases']}")
    logger.info(f"Total samples: {metadata['summary']['total_samples']}")
    logger.info(f"Cases with complete data (ADC+Calc): {metadata['summary']['cases_complete']}/{metadata['summary']['total_cases']}")
    logger.info(f"Samples with complete data: {metadata['summary']['samples_complete']}/{metadata['summary']['total_samples']}")
    logger.info(f"Samples with prostate: {metadata['summary']['samples_with_prostate']}")
    logger.info(f"Samples with target: {metadata['summary']['samples_with_target']}")
    logger.info(f"Class distribution: {metadata['summary']['class_distribution']}")
    logger.info(f"Output: {output_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate training metadata for 2.5D multi-modal segmentation"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/aligned_v2"),
        help="Path to aligned_v2 data directory (default: data/aligned_v2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for metadata.json (default: <data-dir>/metadata.json)",
    )
    parser.add_argument(
        "--t2-context",
        type=int,
        default=5,
        help="Number of T2 context slices (must be odd, default: 5)",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["t2", "adc", "calc"],
        help="List of modality names (default: t2 adc calc)",
    )
    parser.add_argument(
        "--masks",
        nargs="+",
        default=["mask_prostate", "mask_target1"],
        help="List of mask names (default: mask_prostate mask_target1)",
    )
    parser.add_argument(
        "--positive-value",
        type=int,
        default=255,
        help="Pixel value indicating positive label in masks (default: 255)",
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = args.data_dir / "metadata.json"
    
    # Generate metadata
    generate_metadata(
        data_dir=args.data_dir,
        output_path=args.output,
        t2_context_size=args.t2_context,
        modalities=args.modalities,
        masks=args.masks,
        positive_value=args.positive_value,
    )


if __name__ == "__main__":
    main()
