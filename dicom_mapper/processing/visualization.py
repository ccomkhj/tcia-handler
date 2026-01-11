import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Color scheme for different structures (RGBA)
STRUCTURE_COLORS = {
    "prostate": (255, 255, 0, 100),   # Yellow
    "target1": (255, 0, 0, 150),      # Red
    "target2": (255, 128, 0, 150),    # Orange
    "target3": (255, 0, 255, 150),    # Magenta
    "default": (0, 255, 0, 100),      # Green
}

def create_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Create an RGB overlay of a mask on a grayscale image.
    """
    # Normalize image to 0-1
    if image.dtype == np.uint8:
        img_norm = image.astype(np.float32) / 255.0
    else:
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
    # Convert to RGB
    img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    
    # Process Mask
    r, g, b, a = color
    alpha = a / 255.0
    
    mask_bool = mask > 0
    if mask_bool.any():
        img_rgb[mask_bool, 0] = (1 - alpha) * img_rgb[mask_bool, 0] + alpha * (r / 255.0)
        img_rgb[mask_bool, 1] = (1 - alpha) * img_rgb[mask_bool, 1] + alpha * (g / 255.0)
        img_rgb[mask_bool, 2] = (1 - alpha) * img_rgb[mask_bool, 2] + alpha * (b / 255.0)
        
    return (img_rgb * 255).astype(np.uint8)

class AlignedVisualizer:
    """Visualizer for aligned datasets produced by dicom_mapper."""
    
    def visualize_case(self, case_dir: Path, output_dir: Path, slices_to_viz: int = 5):
        """
        Generate visualizations for a processed case.
        
        Args:
            case_dir: Directory containing aligned data (t2/, adc/, masks/, etc.)
            output_dir: Directory to save visualization images.
            slices_to_viz: Number of equally spaced slices to visualize.
        """
        case_id = case_dir.name
        
        # Identify available modalities and masks
        modalities = ["t2", "adc", "calc"]
        available_mods = [m for m in modalities if (case_dir / m).exists()]
        
        mask_dirs = list(case_dir.glob("mask_*"))
        if not available_mods:
            logger.warning(f"No image data found in {case_dir}")
            return

        # Get slice list from T2 (reference)
        t2_dir = case_dir / "t2"
        if not t2_dir.exists():
            # Fallback to first available
            t2_dir = case_dir / available_mods[0]
            
        slice_files = sorted(t2_dir.glob("*.png"))
        if not slice_files:
            return
            
        # Select slices indices
        num_slices = len(slice_files)
        indices = np.linspace(0, num_slices - 1, slices_to_viz, dtype=int)
        
        # Create output dir for this case
        case_vis_dir = output_dir / case_id
        case_vis_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in indices:
            slice_filename = slice_files[idx].name
            
            # Setup plot: One row per modality
            fig, axes = plt.subplots(len(available_mods), 2, figsize=(10, 5 * len(available_mods)))
            if len(available_mods) == 1:
                axes = axes.reshape(1, -1)
                
            fig.suptitle(f"Case {case_id} - {slice_filename}", fontsize=16)
            
            for i, mod in enumerate(available_mods):
                img_path = case_dir / mod / slice_filename
                if not img_path.exists():
                    continue
                    
                image = np.array(Image.open(img_path))
                
                # Create overlay
                overlay = image.copy()
                # Need to convert overlay to RGB first? create_overlay handles it.
                # But create_overlay takes a base image.
                # We will iteratively apply masks.
                
                # Base for overlay
                # We reuse the create_overlay logic which returns RGB uint8
                if image.ndim == 2:
                    current_overlay = np.stack([image, image, image], axis=-1)
                    # Normalize for display consistency if needed, but keeping original intensity is better
                    # create_overlay expects raw image and returns RGB
                    # Let's pass the raw image to the first call
                    
                    # We need a fresh start for the composite
                    # Hack: create_overlay creates a new RGB from grayscale every time.
                    # We should maintain the RGB buffer.
                    
                    # Re-implement simple loop here for efficiency
                    # Normalize base
                    if image.dtype == np.uint8:
                         base_norm = image.astype(np.float32) / 255.0
                    else:
                         base_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
                    
                    composite = np.stack([base_norm, base_norm, base_norm], axis=-1)
                else:
                    composite = image.astype(np.float32) / 255.0

                # Apply all masks
                masks_applied = []
                for mask_dir in mask_dirs:
                    mask_path = mask_dir / slice_filename
                    if mask_path.exists():
                        mask = np.array(Image.open(mask_path))
                        if mask.max() > 0: # If mask is not empty
                            struct_name = mask_dir.name.replace("mask_", "")
                            color = STRUCTURE_COLORS.get(struct_name, STRUCTURE_COLORS["default"])
                            
                            r, g, b, a = color
                            alpha = a / 255.0
                            mask_bool = mask > 0
                            
                            composite[mask_bool, 0] = (1 - alpha) * composite[mask_bool, 0] + alpha * (r / 255.0)
                            composite[mask_bool, 1] = (1 - alpha) * composite[mask_bool, 1] + alpha * (g / 255.0)
                            composite[mask_bool, 2] = (1 - alpha) * composite[mask_bool, 2] + alpha * (b / 255.0)
                            
                            masks_applied.append(struct_name)
                
                # Plot Original
                axes[i, 0].imshow(image, cmap="gray")
                axes[i, 0].set_title(f"{mod.upper()} Original")
                axes[i, 0].axis("off")
                
                # Plot Overlay
                axes[i, 1].imshow(composite)
                label = f"{mod.upper()} + {' + '.join(masks_applied)}" if masks_applied else f"{mod.upper()} (No Mask)"
                axes[i, 1].set_title(label)
                axes[i, 1].axis("off")
            
            plt.tight_layout()
            plt.savefig(case_vis_dir / f"viz_{slice_filename}", dpi=100)
            plt.close(fig)
