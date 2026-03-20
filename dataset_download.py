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
    extract_dir = "."

    if os.path.exists("BH_CT_Data"):
        print("BH_CT_Data/ already exists, skipping dataset download.")
        print("  (Delete BH_CT_Data/ folder to re-download.)")
        return

    download_file(url, zip_path, desc="BH_CT_Data.zip (dataset)")
    extract_zip(zip_path, extract_dir)
    print("Dataset ready: BH_CT_Data/")


def download_models():
    """Download pre-trained U-Net model examples from Google Drive."""
    file_id = "1hWFd7PUgJ7G9tBFsMSlj6LmD_1AF1ivt"
    zip_path = "unet_model_ex.zip"
    extract_dir = "unet"

    if os.path.exists(os.path.join("unet", "train")):
        print("unet/train/ already exists, skipping model download.")
        print("  (Delete unet/train/ folder to re-download.)")
        return

    download_gdrive(file_id, zip_path, desc="unet_model_ex.zip (pre-trained models)")
    extract_zip(zip_path, extract_dir)

    # Clean up zip
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"  Removed {zip_path}")

    print("Models ready: unet/train/, unet/test/, unet/predict/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset and/or pre-trained models")
    parser.add_argument("--dataset", action="store_true", help="Download BH_CT_Data dataset")
    parser.add_argument("--models",  action="store_true", help="Download pre-trained U-Net models")
    parser.add_argument("--all",     action="store_true", help="Download everything")
    args = parser.parse_args()

    # If no flags, download everything
    if not (args.dataset or args.models or args.all):
        args.all = True

    if args.all or args.dataset:
        download_dataset()
        print()

    if args.all or args.models:
        download_models()

    print("\nDone.")
