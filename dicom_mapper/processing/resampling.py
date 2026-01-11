import SimpleITK as sitk
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class VolumeResampler:
    """Handles resampling of volumes to a reference grid using SimpleITK."""

    def load_png_series_as_sitk(
        self, 
        images_dir: Path, 
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float] = (0, 0, 0),
        direction: Optional[Tuple[float, ...]] = None
    ) -> sitk.Image:
        """Load a sequence of PNG files as a 3D SimpleITK volume."""
        png_files = sorted(images_dir.glob("*.png"))
        if not png_files:
            raise ValueError(f"No PNG files found in {images_dir}")

        slices = []
        for png_file in png_files:
            img = np.array(Image.open(png_file))
            slices.append(img)

        # Stack into (Z, Y, X)
        volume = np.stack(slices, axis=0)
        
        sitk_img = sitk.GetImageFromArray(volume)
        sitk_img.SetSpacing(spacing)
        sitk_img.SetOrigin(origin)
        if direction:
            if len(direction) == 6:
                # DICOM ImageOrientationPatient is 6 values
                # SimpleITK requires a 9-element flattened 3x3 matrix
                # The columns of the matrix are the direction vectors
                row_dir = np.array(direction[:3])
                col_dir = np.array(direction[3:])
                slice_dir = np.cross(row_dir, col_dir)
                
                # Construct matrix with vectors as columns: [row_dir, col_dir, slice_dir]
                # Then flatten row-major
                matrix = np.stack([row_dir, col_dir, slice_dir], axis=1)
                direction = tuple(matrix.flatten())
                
            sitk_img.SetDirection(direction)
        
        return sitk_img

    def resample_to_reference(
        self, 
        moving: sitk.Image, 
        reference: sitk.Image,
        interpolation: str = "linear"
    ) -> sitk.Image:
        """Resample a moving image to the reference image's grid."""
        interpolator = sitk.sitkLinear if interpolation == "linear" else sitk.sitkNearestNeighbor
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(sitk.Transform())
        
        return resampler.Execute(moving)

    def create_reference_from_meta(self, meta: dict, num_slices: int) -> sitk.Image:
        """Create an empty reference SimpleITK image from DICOM metadata."""
        size = (meta["Columns"], meta["Rows"], num_slices)
        spacing = (meta["PixelSpacing"][0], meta["PixelSpacing"][1], meta.get("SliceThickness", 1.0))
        origin = meta.get("Origin", (0, 0, 0))
        direction = meta.get("ImageOrientationPatient", (1, 0, 0, 0, 1, 0))
        
        # sitk direction is 3x3 matrix (flattened)
        # DICOM orientation is two 3D vectors
        if len(direction) == 6:
            v1 = np.array(direction[:3])
            v2 = np.array(direction[3:])
            v3 = np.cross(v1, v2)
            direction_matrix = np.stack([v1, v2, v3]).flatten().tolist()
        else:
            direction_matrix = direction

        ref_img = sitk.Image(size, sitk.sitkFloat32)
        ref_img.SetSpacing(spacing)
        ref_img.SetOrigin(origin)
        ref_img.SetDirection(direction_matrix)
        
        return ref_img
