from typing import List, Optional, Sequence, Tuple, Union

import highdicom as hd
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.valuerep import PersonName


def _format_person_name(name: Union[str, PersonName, None]) -> Optional[str]:
    """
    Format a person name to avoid DICOM warnings about single-component names.

    If the name is a single component (no carets), add a trailing caret
    to indicate it's a family name only.
    """
    if name is None:
        return None
    name_str = str(name)
    if name_str and "^" not in name_str:
        return f"{name_str}^"
    return name_str


from highdicom.content import (
    AlgorithmIdentificationSequence,
    PixelMeasuresSequence,
    PlaneOrientationSequence,
    PlanePositionSequence,
)
from highdicom.sc import SCImage
from highdicom.seg import Segmentation, SegmentDescription
from pydicom.dataset import Dataset
from pydicom.sr.codedict import codes


def create_sc_image(
    source_images: List[Dataset],
    pixel_array: np.ndarray,
    series_description: str,
    series_number: int,
    instance_number: int = 1,
    modality: str = "MR",
) -> List[SCImage]:
    """
    Create Secondary Capture Images from a processed volume.

    For 3D arrays (multi-frame), creates one SCImage per slice since highdicom's
    SCImage only supports 2D grayscale arrays with MONOCHROME2 photometric interpretation.

    Args:
        source_images: List of source DICOM datasets (one per slice or single multi-frame).
                       Used for patient/study metadata.
        pixel_array: The 2D or 3D numpy array. For 3D: (frames, rows, cols).
        series_description: Description for the new series.
        series_number: Number for the new series.

    Returns:
        List of SCImage objects (one per frame for 3D input, single-element list for 2D).
    """
    if not source_images:
        raise ValueError("No source images provided for metadata reference.")

    ref_dcm = source_images[0]

    # Ensure pixel array is compatible with MONOCHROME2 8-bit
    if pixel_array.dtype != np.uint8:
        # Normalize to 0-255 if needed, but for now just cast if range is safe
        # Ideally caller handles normalization, but safety cast is good
        if pixel_array.max() > 255:
            # Normalize 16-bit to 8-bit? Or change bits_allocated?
            # User specified bits_allocated=8 in original code
            # Let's assume input is displayable range or normalize it
            pixel_array = (
                (pixel_array - pixel_array.min())
                / (pixel_array.max() - pixel_array.min())
                * 255
            ).astype(np.uint8)
        else:
            pixel_array = pixel_array.astype(np.uint8)

    # Handle 3D arrays by creating one SCImage per slice
    # highdicom SCImage only supports 2D grayscale with MONOCHROME2
    if pixel_array.ndim == 3:
        frames = [pixel_array[i] for i in range(pixel_array.shape[0])]
    else:
        frames = [pixel_array]

    # Use a shared series UID for all frames
    series_uid = hd.UID()
    sc_images = []

    for idx, frame in enumerate(frames):
        sc_image = SCImage(
            pixel_array=frame,
            series_instance_uid=series_uid,
            series_number=series_number,
            sop_instance_uid=hd.UID(),
            instance_number=idx + 1,
            series_description=series_description,
            # Required arguments
            photometric_interpretation="MONOCHROME2",
            bits_allocated=8,
            coordinate_system=hd.CoordinateSystemNames.PATIENT,
            study_instance_uid=ref_dcm.StudyInstanceUID,
            manufacturer="TCIA-Handler",
            # Metadata from reference
            patient_id=ref_dcm.PatientID,
            patient_name=_format_person_name(ref_dcm.PatientName),
            patient_birth_date=getattr(ref_dcm, "PatientBirthDate", None),
            patient_sex=getattr(ref_dcm, "PatientSex", None),
            accession_number=getattr(ref_dcm, "AccessionNumber", None),
            study_id=getattr(ref_dcm, "StudyID", None),
            study_date=getattr(ref_dcm, "StudyDate", None),
            study_time=getattr(ref_dcm, "StudyTime", None),
            referring_physician_name=_format_person_name(
                getattr(ref_dcm, "ReferringPhysicianName", None)
            ),
            # Orientation and spacing
            patient_orientation=getattr(ref_dcm, "PatientOrientation", ("L", "P")),
            pixel_spacing=getattr(ref_dcm, "PixelSpacing", None),
        )
        sc_images.append(sc_image)

    return sc_images


def create_segmentation(
    source_images: List[Dataset],
    mask_array: np.ndarray,
    category_code: Tuple[str, str, str],  # (Value, Scheme, Meaning)
    type_code: Tuple[str, str, str],
    series_description: str = "Segmentation",
    series_number: int = 100,
    instance_number: int = 1,
) -> Segmentation:
    """
    Create a DICOM Segmentation object from a mask.

    Args:
        source_images: List of source DICOM datasets corresponding to the reference geometry.
        mask_array: 3D numpy array (frames, rows, cols) where >0 is the segment.
                    Must match geometry of source_images.
        category_code: Tuple for segment category (e.g., codes.CID7150.TissueSeg)
        type_code: Tuple for segment type (e.g., codes.CID7150.Tumor)
    """
    if not source_images:
        raise ValueError("No source images provided.")

    # Convert mask to binary (0 or 1) - highdicom expects segment labels to match descriptions
    # For BINARY segmentation, values should be 0 (background) or 1 (segment 1)
    mask_array = (mask_array > 0).astype(np.uint8)

    # Define segment description
    # highdicom expects pydicom Code objects or CodedConcept
    # We construct simplified descriptions here

    # Algorithm identification is required for non-MANUAL algorithm types
    algorithm_id = AlgorithmIdentificationSequence(
        name="TCIA-Handler",
        family=hd.sr.CodedConcept(
            value="123109",
            scheme_designator="DCM",
            meaning="Manual Processing",
        ),
        version="1.0",
    )

    description = SegmentDescription(
        segment_number=1,
        segment_label=series_description,
        segmented_property_category=hd.sr.CodedConcept(
            value=category_code[0],
            scheme_designator=category_code[1],
            meaning=category_code[2],
        ),
        segmented_property_type=hd.sr.CodedConcept(
            value=type_code[0], scheme_designator=type_code[1], meaning=type_code[2]
        ),
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.SEMIAUTOMATIC,
        algorithm_identification=algorithm_id,
    )

    # Create Segmentation
    seg = Segmentation(
        source_images=source_images,
        pixel_array=mask_array,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[description],
        series_instance_uid=hd.UID(),
        series_number=series_number,
        sop_instance_uid=hd.UID(),
        instance_number=instance_number,
        series_description=series_description,
        # Required device/software info
        manufacturer="TCIA-Handler",
        manufacturer_model_name="TCIA-Handler",
        software_versions="1.0",
        device_serial_number="N/A",
    )

    return seg
