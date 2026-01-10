# DICOM Reference Guide

This document provides a comprehensive overview of DICOM sequences used in this project (T2, ADC, Calc) and a practical reference for commonly encountered DICOM tags.

## 1. MRI Sequences in this Project

In MRI DICOM naming, we primarily deal with three types of series:

### T2-Weighted (T2)
- **Meaning:** T2-weighted MRI anatomical image (fluid-bright contrast).
- **Use:** Structural anatomy and lesion visualization.
- **Typical DICOM confirmation:**
  - `SeriesDescription (0008,103E)` contains “T2”
  - Long `Repetition Time (0018,0080)` and long `Echo Time (0018,0081)`

### Apparent Diffusion Coefficient Map (ADC)
- **Meaning:** Map derived from diffusion-weighted imaging (DWI).
- **Use:** Quantitative diffusion assessment (e.g., tumors, stroke).
- **Typical DICOM confirmation:**
  - `ImageType (0008,0008)` includes `DERIVED`
  - `SeriesDescription (0008,103E)` contains “ADC”
  - **Note:** Often requires scaling via `RescaleSlope (0028,1053)` and `RescaleIntercept (0028,1052)` to interpret values correctly.

### Calculated Series (Calc)
- **Meaning:** “Calculated” / “Derived” series produced by scanner or workstation output.
- **Common outputs:** parametric maps, synthetic reconstructions, or high b-value diffusion images.
- **Typical DICOM confirmation:**
  - `ImageType (0008,0008)` includes `DERIVED` or `SECONDARY`
  - `DerivationDescription (0008,2111)` may describe the computation.

---

## 2. Practical DICOM Tag Reference

The complete list of DICOM tags (thousands of attributes) is defined in the [DICOM Standard PS3.6 - Data Dictionary](https://www.dicomstandard.org/current). Below are the most relevant sets for medical imaging pipelines.

### A) Organization & Identification
Used for organizing data in PACS and processing pipelines:
- **(0010,0010)** Patient’s Name
- **(0010,0020)** Patient ID
- **(0020,000D)** Study Instance UID
- **(0008,0020)** Study Date
- **(0008,1030)** Study Description
- **(0020,000E)** Series Instance UID
- **(0008,103E)** Series Description
- **(0008,0060)** Modality
- **(0020,0013)** Instance Number (Slice index)

### B) Image Interpretation (Pixel Module)
Describes how to interpret the pixel array:
- **(0028,0010/0011)** Rows / Columns
- **(0028,0100/0101)** Bits Allocated / Stored
- **(0028,1052/1053)** Rescale Intercept / Slope (Crucial for ADC/Calc)
- **(0028,1050/1051)** Window Center / Width
- **(7FE0,0010)** Pixel Data

### C) Geometry & Spatial Positioning
Critical for 3D reconstruction and multi-modal alignment:
- **(0028,0030)** Pixel Spacing (x, y)
- **(0018,0050)** Slice Thickness
- **(0020,0032)** Image Position (Patient) - [x, y, z] coordinates of the first transmitted voxel.
- **(0020,0037)** Image Orientation (Patient) - Direction cosines of rows and columns.
- **(0018,5100)** Patient Position (e.g., HFS - Head First Supine)

### D) MR-Specific Parameters
- **(0018,0080)** Repetition Time (TR)
- **(0018,0081)** Echo Time (TE)
- **(0018,0082)** Inversion Time (TI)
- **(0018,1314)** Flip Angle
- **(0018,1310)** Acquisition Matrix

---

## 3. Programmatic Inspection

### Listing all tags in a file (Python/pydicom)
To see exactly what tags are present in a specific DICOM file:

```python
import pydicom

ds = pydicom.dcmread("file.dcm", stop_before_pixels=True)

# Dump every tag with name and value
for elem in ds.iterall():
    if elem.VR != "SQ":  # Sequences can be very long
        print(f"({elem.tag.group:04X},{elem.tag.element:04X}) {elem.name:30} | {elem.value}")
    else:
        print(f"({elem.tag.group:04X},{elem.tag.element:04X}) {elem.name:30} | [Sequence]")
```

### Accessing the Global Dictionary
To list the entire standard DICOM dictionary programmatically:

```python
from pydicom.datadict import DicomDictionary

for tag, entry in DicomDictionary.items():
    vr, vm, name, is_retired, keyword = entry
    print(tag, name, keyword, vr, vm)
```