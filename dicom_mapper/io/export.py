from pathlib import Path
import numpy as np
from PIL import Image
import pydicom
from typing import Union
import logging

logger = logging.getLogger(__name__)

class PNGExporter:
    """Handles export of DICOM pixel data to PNG sequences."""

    def export_dicom_to_png(self, dcm_dataset: pydicom.dataset.Dataset, output_dir: Path):
        """
        Export frames from a Multi-frame DICOM dataset to a sequence of PNG files.
        
        Args:
            dcm_dataset: The loaded pydicom dataset (e.g. SC Image or SEG).
            output_dir: Directory where PNGs will be saved.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pixel_array = dcm_dataset.pixel_array
        
        # Handle 2D vs 3D
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]
            
        num_frames = pixel_array.shape[0]
        
        logger.info(f"Exporting {num_frames} frames to {output_dir}")
        
        for i in range(num_frames):
            frame = pixel_array[i]
            
            # Normalize if needed (for 16-bit or float images)
            if frame.dtype != np.uint8:
                min_val = frame.min()
                max_val = frame.max()
                if max_val > min_val:
                    frame = ((frame - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Save
            # Use 4-digit index for sorting
            out_path = output_dir / f"{i:04d}.png"
            Image.fromarray(frame).save(out_path)

    def export_mask_to_png(self, dcm_seg: pydicom.dataset.Dataset, output_dir: Path):
        """
        Export Segmentation frames to PNG.
        
        Note: Standard DICOM SEG stores frames packed or one bit per pixel.
        pydicom's pixel_array handler should unpack this.
        If the SEG has multiple segments, they might be in different frames.
        This simplified exporter assumes 1 segment per file or unpacks all frames sequentially.
        """
        self.export_dicom_to_png(dcm_seg, output_dir)
