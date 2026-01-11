import logging
from pathlib import Path

import click
import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image
from pydicom.sr.codedict import codes
from tqdm import tqdm

from dicom_mapper.core.geometry import SeriesInfo, compute_slice_mapping
from dicom_mapper.core.highdicom_creation import create_sc_image, create_segmentation
from dicom_mapper.io.dicom import extract_slice_locations_from_dicom, load_dicom_series
from dicom_mapper.io.export import PNGExporter
from dicom_mapper.processing.resampling import VolumeResampler
from dicom_mapper.processing.visualization import AlignedVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """DICOM Mapper: Align and Standardize Multi-Modal MRI."""
    pass


@cli.command()
@click.option(
    "--aligned-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing aligned output (e.g. data/aligned_v2)",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to save visualizations",
)
@click.option("--class-num", type=int, help="Filter by specific class (1-4)")
def visualize(aligned_dir, output_dir, class_num):
    """Visualize aligned data (Overlays on T2/ADC/Calc)."""
    aligned_path = Path(aligned_dir)
    output_path = Path(output_dir)
    visualizer = AlignedVisualizer()

    if class_num:
        class_dirs = [aligned_path / f"class{class_num}"]
    else:
        class_dirs = list(aligned_path.glob("class*"))

    for class_dir in class_dirs:
        if not class_dir.exists():
            continue

        case_dirs = list(class_dir.glob("case_*"))
        for case_dir in tqdm(case_dirs, desc=f"Visualizing {class_dir.name}"):
            visualizer.visualize_case(case_dir, output_path / class_dir.name)


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True),
    help="Root data directory containing 'processed' and 'nbia'",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for aligned data",
)
@click.option("--class-num", type=int, help="Filter by specific class (1-4)")
@click.option("--case-id", type=str, help="Process single case ID")
def process(input_dir, output_dir, class_num, case_id):
    """Process cases: Align -> Create DICOM -> Export PNG."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 1. Discovery
    # We iterate cases found in the T2 processed directory
    processed_root = input_path / "processed"
    if class_num:
        class_dirs = [processed_root / f"class{class_num}"]
    else:
        class_dirs = list(processed_root.glob("class*"))

    for class_dir in class_dirs:
        current_class = int(class_dir.name.replace("class", ""))

        # Filter cases if needed
        case_dirs = list(class_dir.glob("case_*"))
        if case_id:
            case_dirs = [d for d in case_dirs if case_id in d.name]

        for case_dir in tqdm(case_dirs, desc=f"Processing {class_dir.name}"):
            process_single_case(case_dir, input_path, output_path, current_class)


def load_modality_volume(
    case_id: str,
    class_num: int,
    root_dir: Path,
    modality_processed_name: str,
    modality_dicom_name: str,
    resampler: VolumeResampler,
):
    """Helper to load a modality's volume and DICOM datasets."""
    # 1. Find Processed Images
    processed_base = (
        root_dir / modality_processed_name / f"class{class_num}" / f"case_{case_id}"
    )
    if not processed_base.exists():
        logger.debug(f"Processed base not found: {processed_base}")
        return None, None

    # Find series subdir (assuming one series per case/modality)
    series_dirs = [d for d in processed_base.iterdir() if d.is_dir()]
    if not series_dirs:
        logger.debug(f"No series dirs found in {processed_base}")
        return None, None
    processed_dir = series_dirs[0] / "images"

    # 2. Find DICOMs
    dicom_root = root_dir / modality_dicom_name / f"class{class_num}"
    # Robust search for case ID in DICOM structure
    dicom_files = list(dicom_root.rglob(f"**/*{case_id}*/**/*.dcm"))

    if not dicom_files:
        # Try finding by series UID if possible, but for now rely on case_id path matching
        logger.warning(
            f"DICOMs not found for {modality_dicom_name} case {case_id} in {dicom_root}"
        )
        return None, None

    # Filter DICOMs for the specific series if we have multiple
    # (Simplified: take the first series found for this case)
    # Ideally we match SeriesInstanceUID from processed_dir.parent.name
    target_uid = processed_dir.parent.name

    # Find the directory that contains files with this UID
    # We'll just load the first consistent series we find for now
    dicoms = load_dicom_series(dicom_files[0].parent)

    # Verify UID match if strictly needed, but let's trust the case-level folder for now

    # 3. Load Volume
    spacing = (
        float(dicoms[0].PixelSpacing[0]),
        float(dicoms[0].PixelSpacing[1]),
        float(dicoms[0].SliceThickness),
    )
    origin = dicoms[0].ImagePositionPatient

    # Handle direction if available
    direction = getattr(dicoms[0], "ImageOrientationPatient", None)

    volume = resampler.load_png_series_as_sitk(
        processed_dir, spacing, origin, direction
    )

    return volume, dicoms


def save_sc_series(sc_images: list, output_dir: Path, base_name: str):
    """
    Save a list of SCImage objects to a directory.

    Args:
        sc_images: List of SCImage objects (one per frame).
        output_dir: Directory to save to.
        base_name: Base name for the subdirectory (e.g., 't2_aligned').
    """
    series_dir = output_dir / base_name
    series_dir.mkdir(parents=True, exist_ok=True)
    for idx, sc in enumerate(sc_images):
        sc.save_as(series_dir / f"{idx:04d}.dcm")


def export_sc_series_to_png(sc_images: list, output_dir: Path, exporter):
    """
    Export a list of SCImage objects to PNG files.

    Args:
        sc_images: List of SCImage objects.
        output_dir: Directory to save PNG files.
        exporter: PNGExporter instance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, sc in enumerate(sc_images):
        frame = sc.pixel_array
        if frame.dtype != np.uint8:
            min_val, max_val = frame.min(), frame.max()
            if max_val > min_val:
                frame = ((frame - min_val) / (max_val - min_val) * 255.0).astype(
                    np.uint8
                )
            else:
                frame = frame.astype(np.uint8)
        from PIL import Image

        Image.fromarray(frame).save(output_dir / f"{idx:04d}.png")


def process_single_case(
    case_dir: Path, root_dir: Path, output_root: Path, class_num: int
):
    """
    Orchestrate full pipeline for one case.
    """
    case_id = case_dir.name.replace("case_", "")
    output_case_dir = output_root / f"class{class_num}" / case_dir.name
    output_case_dir.mkdir(parents=True, exist_ok=True)

    resampler = VolumeResampler()
    exporter = PNGExporter()

    # --- 1. Process T2 (Reference) ---
    t2_volume, t2_datasets = load_modality_volume(
        case_id, class_num, root_dir, "processed", "nbia", resampler
    )

    if not t2_volume:
        logger.warning(f"Skipping case {case_id}: Missing T2 data")
        return

    # Save T2 DICOM
    t2_arr = sitk.GetArrayFromImage(t2_volume)
    sc_t2_list = create_sc_image(t2_datasets, t2_arr, "Aligned T2", 101, modality="MR")
    save_sc_series(sc_t2_list, output_case_dir, "t2_aligned")

    # Export T2 PNG
    export_sc_series_to_png(sc_t2_list, output_case_dir / "t2", exporter)

    # --- 2. Process ADC ---
    adc_volume, adc_datasets = load_modality_volume(
        case_id, class_num, root_dir, "processed_ep2d_adc", "nbia_ep2d_adc", resampler
    )

    if adc_volume:
        # Resample to T2
        adc_resampled = resampler.resample_to_reference(adc_volume, t2_volume)
        adc_arr = sitk.GetArrayFromImage(adc_resampled)

        # Create DICOM (using T2 datasets as reference for geometry/frame of reference)
        # But we might want to keep some ADC metadata?
        # For 'Aligned' series, they share the Frame of Reference of T2.
        sc_adc_list = create_sc_image(
            t2_datasets, adc_arr, "Aligned ADC", 102, modality="MR"
        )
        save_sc_series(sc_adc_list, output_case_dir, "adc_aligned")

        # Export PNG
        export_sc_series_to_png(sc_adc_list, output_case_dir / "adc", exporter)
    else:
        logger.warning(f"Case {case_id}: Missing ADC")

    # --- 3. Process Calc ---
    calc_volume, calc_datasets = load_modality_volume(
        case_id, class_num, root_dir, "processed_ep2d_calc", "nbia_ep2d_calc", resampler
    )

    if calc_volume:
        # Resample to T2
        calc_resampled = resampler.resample_to_reference(calc_volume, t2_volume)
        calc_arr = sitk.GetArrayFromImage(calc_resampled)

        sc_calc_list = create_sc_image(
            t2_datasets, calc_arr, "Aligned Calc", 103, modality="MR"
        )
        save_sc_series(sc_calc_list, output_case_dir, "calc_aligned")

        export_sc_series_to_png(sc_calc_list, output_case_dir / "calc", exporter)
    else:
        logger.warning(f"Case {case_id}: Missing Calc")

    # --- 4. Handle Segments ---
    processed_seg_root = (
        root_dir / "processed_seg" / f"class{class_num}" / f"case_{case_id}"
    )
    if processed_seg_root.exists():
        # Iterate over series UIDs (usually just one that matches T2 or similar)
        # We look for subdirectories that contain mask folders
        # Structure: processed_seg/classX/case_Y/SERIES_UID/MASK_NAME/*.png

        # We need to find the series directory.
        # In the old logic, it matched T2 series UID or took the first one.
        seg_series_dirs = [d for d in processed_seg_root.iterdir() if d.is_dir()]

        if seg_series_dirs:
            # Try to find one matching T2
            target_seg_dir = None
            t2_uid = t2_datasets[0].SeriesInstanceUID

            for d in seg_series_dirs:
                if d.name == t2_uid:
                    target_seg_dir = d
                    break

            if not target_seg_dir:
                target_seg_dir = seg_series_dirs[0]  # Fallback

            # Get T2 dimensions for creating full-volume masks
            t2_num_slices = t2_arr.shape[0]
            t2_height, t2_width = t2_arr.shape[1], t2_arr.shape[2]

            # Iterate over masks (e.g., "prostate", "target1", "target2")
            for mask_name_dir in target_seg_dir.iterdir():
                if not mask_name_dir.is_dir():
                    continue

                mask_name = mask_name_dir.name

                try:
                    # Create full-volume mask array (all zeros initially)
                    full_mask = np.zeros(
                        (t2_num_slices, t2_height, t2_width), dtype=np.uint8
                    )

                    # Load individual mask slices and place them at correct indices
                    mask_files = sorted(mask_name_dir.glob("*.png"))
                    for mask_file in mask_files:
                        # Get slice index from filename (e.g., "0021.png" -> 21)
                        slice_idx = int(mask_file.stem)
                        if 0 <= slice_idx < t2_num_slices:
                            mask_slice = np.array(Image.open(mask_file).convert("L"))
                            # Resize if necessary
                            if mask_slice.shape != (t2_height, t2_width):
                                mask_slice = np.array(
                                    Image.fromarray(mask_slice).resize(
                                        (t2_width, t2_height), Image.NEAREST
                                    )
                                )
                            full_mask[slice_idx] = mask_slice

                    # Export mask PNGs directly (matching T2 slice numbering)
                    mask_output_dir = output_case_dir / f"mask_{mask_name}"
                    mask_output_dir.mkdir(parents=True, exist_ok=True)

                    for idx in range(t2_num_slices):
                        mask_slice = full_mask[idx]
                        Image.fromarray(mask_slice).save(
                            mask_output_dir / f"{idx:04d}.png"
                        )

                    logger.info(f"Exported mask {mask_name} for case {case_id}")

                except Exception as e:
                    logger.warning(
                        f"Failed to process mask {mask_name} for case {case_id}: {e}"
                    )


if __name__ == "__main__":
    cli()
