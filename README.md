# urban-tree-canopy

## Tree Canopy Segmentation and Detection

This repository provides toolbox for:

1. **U-Net** – Tree canopy segmentation from NAIP aerial imagery  
   and python code for:
2. **YOLO** – Tree object detection and canopy bounding box extraction

# 🌳 U-Net Segmentation Toolbox

This repository provides a **U-Net-based segmentation toolbox** for detecting and segmenting urban tree canopies using aerial or satellite imagery.  
The toolbox supports **GeoTIFF I/O**, **data augmentation**, and **spatially referenced outputs** for GIS applications.

The U-Net toolbox can be run in two modes:

- **GUI mode** (default) – interactive graphical interface with PyQt5
- **Headless / CLI mode** (`--headless`) – interactive command-line interface for servers or remote sessions

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
```

### 3. Create and Activate the Conda Environment

```bash
conda create --n tree --file environment.yml
```

The `environment.yml` file includes all dependencies required for the U-Net segmentation toolbox:

- `python=3.8`
- `pytorch`, `torchvision`, `torchaudio`
- `rasterio`, `albumentations`
- `segmentation-models-pytorch`, `matplotlib`
- `ultralytics`

---

### Download Dataset & Pre-trained Models

```bash
# Download everything (dataset + pre-trained models)
python dataset_download.py

# Download only the dataset (BH_CT_Data/)
python dataset_download.py --dataset

# Download only pre-trained U-Net models (unet/train/, unet/test/, unet/predict/)
python dataset_download.py --models
```

---

## Default Dataset Structure

The repository expects the following dataset layout under `BH_CT_Data/`:

```
BH_CT_Data/
├── unet_dataset/
│   ├── train/
│   │   ├── images/    # Training images (.tif)
│   │   └── labels/    # Training labels (.tif)
│   ├── val/
│   │   ├── images/    # Validation images
│   │   └── labels/    # Validation labels
│   └── test/
│       ├── images/    # Test images
│       └── labels/    # Test labels
└── yolo_dataset/
    ├── train/
    │   ├── images/    # Training images
    │   └── labels/    # Training labels (.txt)
    ├── val/
    │   ├── images/    # Validation images
    │   └── labels/    # Validation labels (.txt)
    └── test/
        ├── images/    # Test images
        └── labels/    # Test labels (.txt)
```

Both the U-Net and YOLO scripts automatically resolve these default paths.

---

## Running the U-Net Segmentation Toolbox

### GUI Mode (default)

```bash
conda activate tree
cd unet
python unet_tree_canopy_model.py
```

### Headless / CLI Mode

```bash
conda activate tree
cd unet
python unet_tree_canopy_model.py --headless
```

In headless mode, you will see an interactive menu:

```
============================================================
  U-Net Segmentation Toolbox  —  Headless (CLI) Mode
============================================================

--- Current Configuration ---
  Train Images : .../BH_CT_Data/unet_dataset/train/images
  Train Labels : .../BH_CT_Data/unet_dataset/train/labels
  ...

Select mode:
  [1] Train
  [2] Test
  [3] Predict
  [4] Show current config
  [5] Modify config
  [q] Quit
>>>
```

### Specifying a Custom Config File

Both GUI and headless modes accept a `--config` flag:

```bash
python unet_tree_canopy_model.py --config my_custom_config.yaml
python unet_tree_canopy_model.py --headless --config my_custom_config.yaml
```

### Jupyter Notebook Mode

An interactive notebook with **ipywidgets** buttons is also available:

```bash
conda activate tree
cd unet
jupyter notebook unet_tree_canopy_notebook.ipynb
```

The notebook provides:

- Editable text fields for all data paths and model path
- Sliders and input boxes for hyperparameters (LR, batch size, epochs, threshold)
- **Train**, **Test**, **Predict** buttons to launch each mode
- **Stop** button to kill a running process
- Real-time streaming output log
- Shared `dataset_path.yaml` configuration with GUI and headless modes

> **Requirement:** `ipywidgets` must be installed. You can install it with `pip install ipywidgets` or `conda install ipywidgets`.

---

## 🔧 Configuration via YAML

All settings are stored in `unet/dataset_path.yaml`. You can edit this file directly to change paths, hyperparameters, and other options. Changes made through the GUI or headless mode are automatically saved back to this file.

### Example `dataset_path.yaml`

```yaml
# U-Net Segmentation Configuration
# Paths can be absolute or relative to the project root

# --- Training Data ---
train_images: "../BH_CT_Data/unet_dataset/train/images"
train_labels: "../BH_CT_Data/unet_dataset/train/labels"

# --- Validation Data ---
val_images: "../BH_CT_Data/unet_dataset/val/images"
val_labels: "../BH_CT_Data/unet_dataset/val/labels"

# --- Test Data ---
test_images: "../BH_CT_Data/unet_dataset/test/images"
test_labels: "../BH_CT_Data/unet_dataset/test/labels"

# --- Prediction ---
predict_images: ""
output_tif: ""

# --- Model ---
model_path: ""

# --- Hyperparameters ---
learning_rate: 0.001
batch_size: 8
epochs: 10
threshold: 0.5
use_aug: false
```

### Configurable Options

| Key              | Type  | Default                                | Description                                            |
| ---------------- | ----- | -------------------------------------- | ------------------------------------------------------ |
| `train_images`   | path  | `BH_CT_Data/unet_dataset/train/images` | Training images directory                              |
| `train_labels`   | path  | `BH_CT_Data/unet_dataset/train/labels` | Training labels directory                              |
| `val_images`     | path  | `BH_CT_Data/unet_dataset/val/images`   | Validation images directory                            |
| `val_labels`     | path  | `BH_CT_Data/unet_dataset/val/labels`   | Validation labels directory                            |
| `test_images`    | path  | `BH_CT_Data/unet_dataset/test/images`  | Test images directory                                  |
| `test_labels`    | path  | `BH_CT_Data/unet_dataset/test/labels`  | Test labels directory                                  |
| `predict_images` | path  | `BH_CT_Data/unet_dataset/test/images`  | Directory for prediction input (defaults to test data) |
| `output_tif`     | path  | `predict_output.tif`                   | Output TIFF file path for prediction                   |
| `model_path`     | path  | _(empty)_                              | Path to a trained model (`.pt`)                        |
| `learning_rate`  | float | `0.001`                                | Learning rate for training                             |
| `batch_size`     | int   | `8`                                    | Batch size for training                                |
| `epochs`         | int   | `10`                                   | Number of training epochs                              |
| `threshold`      | float | `0.5`                                  | Sigmoid threshold for binary segmentation              |
| `use_aug`        | bool  | `false`                                | Enable data augmentation (Albumentations)              |

---

## 🧭 U-Net Features

- Interactive GUI with tabs for **Train**, **Test**, and **Predict**
- **Headless CLI mode** for servers and automated pipelines
- Browse buttons to easily select image and label directories
- Adjustable hyperparameters:
  - Learning Rate (`-lr`)
  - Batch Size (`-batch_size`)
  - Epochs (`-epochs`)
- Optional **data augmentation** (`-use_aug`)
- Real-time progress and log display
- YAML-based configuration (shared between GUI and CLI)
  <img width="800" height="1108" alt="unet_ui" src="https://github.com/user-attachments/assets/c1124811-f986-4de8-9186-09f6a3c13051" />

All training results, model checkpoints, and plots (loss, F1, precision-recall) are automatically saved in structured directories.

---

## 🌲 Running the YOLO Scripts

The YOLO scripts use a dataset configuration YAML file that defaults to `yolo/yolo_dataset.yaml`.

### YOLO Dataset YAML (`yolo/yolo_dataset.yaml`)

```yaml
path: ../BH_CT_Data/yolo_dataset # dataset root dir
train: train/images # train images (relative to 'path')
val: val/images # val images
test: test/images # test images

names:
  0: tree
```

---

### 🏋️ Training

#### Basic (uses all defaults)

```bash
conda activate tree
cd yolo
python yolo_tree_canopy_model_train.py
```

#### With Custom Options

```bash
python yolo_tree_canopy_model_train.py \
    --yaml-file yolo_dataset.yaml \
    --img-sizes 320 640 \
    --code-base-folder .
```

#### Training Arguments

| Argument             | Default                  | Description                        |
| -------------------- | ------------------------ | ---------------------------------- |
| `--yaml-file`        | `yolo/yolo_dataset.yaml` | YOLO dataset config YAML           |
| `--img-sizes`        | `320`                    | List of image sizes for training   |
| `--code-base-folder` | `yolo/`                  | Base folder for YOLO runs and logs |

---

### 🔍 Prediction

#### Basic Prediction

```bash
cd yolo
python yolo_tree_canopy_model_predict.py \
    --model runs/detect/<train_name>/weights/best.pt
```

This will predict on the default test images (`BH_CT_Data/yolo_dataset/test/images`).

#### Predict on Custom Images

```bash
python yolo_tree_canopy_model_predict.py \
    --model runs/detect/<train_name>/weights/best.pt \
    --source /path/to/your/images \
    --img-sizes 640 \
    --confs 0.3 \
    --ious 0.5 \
    --save-txt --save-conf
```

#### Batch Prediction (multiple thresholds)

You can sweep across multiple image sizes, confidence thresholds, and IoU thresholds:

```bash
python yolo_tree_canopy_model_predict.py \
    --model runs/detect/<train_name>/weights/best.pt \
    --img-sizes 320 640 \
    --confs 0.3 0.5 0.7 \
    --ious 0.3 0.5 0.7 \
    --save-txt --save-crop
```

This creates a separate prediction run for each combination.

#### Prediction Arguments

| Argument             | Default                               | Description                                 |
| -------------------- | ------------------------------------- | ------------------------------------------- |
| `--model`            | _(required)_                          | Path to trained YOLO model weights (`.pt`)  |
| `--source`           | `BH_CT_Data/yolo_dataset/test/images` | Image source (directory or single file)     |
| `--img-sizes`        | `320`                                 | List of image sizes for prediction          |
| `--confs`            | `0.5`                                 | List of confidence thresholds               |
| `--ious`             | `0.5`                                 | List of IoU thresholds for NMS              |
| `--device`           | `0`                                   | CUDA device number or `cpu`                 |
| `--save-txt`         | `False`                               | Save detection results as `.txt` files      |
| `--save-conf`        | `False`                               | Include confidence scores in `.txt` results |
| `--save-crop`        | `False`                               | Save cropped detection images               |
| `--code-base-folder` | `yolo/`                               | Base folder for prediction runs and logs    |

All prediction results and logs are saved under `predict_logs/` in the code base folder.

---

_Updated: March 2026_  
Tree Canopy Project —  
**U-Net Segmentation Toolbox**  
Maintained by  
**USC Spatial Sciences Institute**
