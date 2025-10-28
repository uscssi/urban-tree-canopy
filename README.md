# urban-tree-canopy

## Tree Canopy Segmentation and Detection

This repository provides toolbox for:
1. **U-Net** – Tree canopy segmentation from NAIP aerial imagery  
                and python code for:
2. **YOLO** – Tree object detection and canopy bounding box extraction  


# 🌳 U-Net Segmentation Toolbox

This repository provides a **U-Net-based segmentation GUI** for detecting and segmenting urban tree canopies using aerial or satellite imagery.  
The toolbox supports **GeoTIFF I/O**, **data augmentation**, and **spatially referenced outputs** for GIS applications.

---

## ⚙️ Environment Setup

### 1. Install Miniconda (if not installed)
Download and install Miniconda from:  
👉 [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

After installation, open the **Command Prompt (cmd)**.

---

### 2. Clone the Repository
```bash
git clone https://github.com/uscssi/urban-tree-canopy.git
cd urban-tree-canopy

### 3. Create and Activate the Conda Environment
```bash
conda create --n tree --file environment.yml

The `environment.yml` file includes all dependencies required for the U-Net segmentation toolbox:

- `python=3.8`
- `pytorch`, `torchvision`, `torchaudio`
- `rasterio`, `albumentations`
- `segmentation-models-pytorch`, `matplotlib`
- `ultralytics`


## 🌱 Running the U-Net Segmentation Toolbox (GUI)
Activate the conda environment:
```bash
conda activate tree

Execute the U-Net segmentation GUI:
```bash
cd unet
python unet_tree_canopy_model.py

## 🧭 Features

- Interactive GUI with tabs for **Train**, **Test**, and **Predict**
- Browse buttons to easily select image and label directories
- Adjustable hyperparameters:
    - Learning Rate (`-lr`)
    - Batch Size (`-batch_size`)
    - Epochs (`-epochs`)
- Optional **data augmentation** (`-use_aug`)
- Real-time progress and log display
<img width="800" height="1108" alt="unet_ui" src="https://github.com/user-attachments/assets/c1124811-f986-4de8-9186-09f6a3c13051" />

All training results, model checkpoints, and plots (loss, F1, precision-recall) are automatically saved in structured directories.

*Updated: October 2025*
Tree Canopy Project —
**U-Net Segmentation Toolbox**
Maintained by
**USC Spatial Sciences Institute**

