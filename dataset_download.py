import requests
import zipfile
import os
import sys
import argparse


def download_file(url, save_path, desc=""):
    """Download a file with progress display."""
    print(f"Downloading {desc}...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 / total
                        sys.stdout.write(f"\r  {pct:.1f}% ({downloaded/1024/1024:.1f} MB / {total/1024/1024:.1f} MB)")
                        sys.stdout.flush()
    print("\n  Download complete.")


def download_gdrive(file_id, save_path, desc=""):
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("  Installing gdown...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    print(f"Downloading {desc} from Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, save_path, quiet=False)
    print("  Download complete.")


def extract_zip(zip_path, extract_dir):
    """Extract a zip file."""
    print(f"Extracting {zip_path} -> {extract_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("  Extraction complete.")


def download_dataset():
    """Download the BH_CT_Data dataset from Zenodo."""
    url = "https://zenodo.org/records/17459767/files/BH_CT_Data.zip?download=1"
    zip_path = "BH_CT_Data.zip"
    extract_dir = "./BH_CT_Data"

    if os.path.exists(extract_dir):
        print(f"{extract_dir}/ already exists, skipping dataset download.")
        print(f"  (Delete {extract_dir}/ folder to re-download.)")
        return

    download_file(url, zip_path, desc="BH_CT_Data.zip (dataset)")
    extract_zip(zip_path, extract_dir)
    print("Dataset ready: BH_CT_Data/")


def download_models():
    """Download pre-trained U-Net and YOLO models independently."""
    
    # --- U-Net Section ---
    file_id_unet = "1hWFd7PUgJ7G9tBFsMSlj6LmD_1AF1ivt"
    zip_path_unet = "unet_model_ex.zip"
    extract_dir_unet = "unet"

    if os.path.exists(os.path.join(extract_dir_unet, "train")):
        print("unet/train/ already exists, skipping model download.")
        print("  (Delete unet/train/ folder to re-download.)")
    else:
        download_gdrive(file_id_unet, zip_path_unet, desc="unet_model_ex.zip (pre-trained models)")
        extract_zip(zip_path_unet, extract_dir_unet)
        if os.path.exists(zip_path_unet):
            os.remove(zip_path_unet)
            print(f"  Removed {zip_path_unet}")
        print("Unet Sample Model ready: unet/train/, unet/test/, unet/predict/")

    # --- YOLO Section (No return, continues even if U-Net skipped) ---
    file_id_yolo = "175g8OV_Uc-I7-ecg3F0X5nL2r0meqLY0"
    yolo_file_path = os.path.join("yolo", "yolo_model.pt")

    if os.path.exists(yolo_file_path):
        print(f"YOLO model already exists at: {os.path.abspath(yolo_file_path)}")
        print("  Skipping YOLO model download.")
    else:
        download_gdrive(file_id_yolo, yolo_file_path, desc="yolo_model.pt")
        print(f"YOLO model saved to: {os.path.abspath(yolo_file_path)}")
        print("YOLO model ready: yolo/yolo_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset and/or pre-trained models")
    parser.add_argument("--dataset", action="store_true", help="Download BH_CT_Data dataset")
    parser.add_argument("--models",  action="store_true", help="Download pre-trained U-Net models")
    parser.add_argument("--all",     action="store_true", help="Download everything")
    args = parser.parse_args()

    if not (args.dataset or args.models or args.all):
        args.all = True

    if args.all or args.dataset:
        download_dataset()
        print()

    if args.all or args.models:
        download_models()

    print("\nDone.")