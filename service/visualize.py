#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np

def get_case_path(base_dir, case_id):
    """Find the case directory across all class folders."""
    # Handle "0144" vs "case_0144"
    if not case_id.startswith("case_"):
        search_name = f"case_{case_id}"
    else:
        search_name = case_id
        
    for class_dir in base_dir.glob("class*"):
        candidate = class_dir / search_name
        if candidate.exists():
            return candidate
    return None

def get_images(case_dir):
    """Get sorted list of image paths from the first series folder."""
    if not case_dir:
        return []
    
    # Find first series directory
    series_dirs = [d for d in case_dir.iterdir() if d.is_dir()]
    if not series_dirs:
        return []
        
    # Use the first series
    target_dir = series_dirs[0] / "images"
    if not target_dir.exists():
        return []
        
    return sorted(list(target_dir.glob("*.png")))

def main():
    parser = argparse.ArgumentParser(description="Visualize MRI sequences (T2, ADC, Calc) for a patient.")
    parser.add_argument("patient_id", nargs="?", help="Patient ID (e.g., 0144 or case_0144). If omitted, picks a random one.")
    parser.add_argument("--output", "-o", default="visualization.png", help="Output filename")
    
    args = parser.parse_args()
    
    data_root = Path("data")
    t2_root = data_root / "processed"
    adc_root = data_root / "processed_ep2d_adc"
    calc_root = data_root / "processed_ep2d_calc"
    
    # Pick patient if not provided
    if not args.patient_id:
        # List all available cases in T2
        all_cases = []
        if t2_root.exists():
            for d in t2_root.glob("class*/case_*"):
                all_cases.append(d.name)
        
        if not all_cases:
            print("No processed cases found in data/processed/")
            return
            
        args.patient_id = random.choice(all_cases)
        print(f"No patient ID provided. Selected random case: {args.patient_id}")
    
    print(f"Visualizing patient: {args.patient_id}")
    
    # Find paths
    t2_case_path = get_case_path(t2_root, args.patient_id)
    adc_case_path = get_case_path(adc_root, args.patient_id)
    calc_case_path = get_case_path(calc_root, args.patient_id)
    
    t2_imgs = get_images(t2_case_path)
    adc_imgs = get_images(adc_case_path)
    calc_imgs = get_images(calc_case_path)
    
    print(f"Found images: T2={len(t2_imgs)}, ADC={len(adc_imgs)}, Calc={len(calc_imgs)}")
    
    if not any([t2_imgs, adc_imgs, calc_imgs]):
        print("No images found for this patient.")
        return

    n = max(len(t2_imgs), len(adc_imgs), len(calc_imgs))
    
    # Plotting
    print(f"Generating plot with {n} rows...")
    fig, axes = plt.subplots(n, 3, figsize=(10, 3*n))
    
    # Ensure axes is always 2D array (n, 3)
    if n == 1:
        axes = np.array([axes])
    elif axes.ndim == 1: # Should not happen with subplots(n, 3) unless n=1, but safety check
        axes = axes.reshape(n, 3)
    
    cols = ["T2", "ADC", "Calc"]
    
    for row in range(n):
        # T2
        if row < len(t2_imgs):
            try:
                axes[row, 0].imshow(mpimg.imread(t2_imgs[row]), cmap='gray')
            except Exception as e:
                print(f"Error reading T2 image {t2_imgs[row]}: {e}")
        axes[row, 0].axis('off')
        if row == 0: axes[row, 0].set_title("T2")

        # ADC
        if row < len(adc_imgs):
            try:
                axes[row, 1].imshow(mpimg.imread(adc_imgs[row]), cmap='gray')
            except Exception as e:
                print(f"Error reading ADC image {adc_imgs[row]}: {e}")
        axes[row, 1].axis('off')
        if row == 0: axes[row, 1].set_title("ADC")

        # Calc
        if row < len(calc_imgs):
            try:
                axes[row, 2].imshow(mpimg.imread(calc_imgs[row]), cmap='gray')
            except Exception as e:
                print(f"Error reading Calc image {calc_imgs[row]}: {e}")
        axes[row, 2].axis('off')
        if row == 0: axes[row, 2].set_title("Calc")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
