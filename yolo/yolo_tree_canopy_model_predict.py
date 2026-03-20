import subprocess
import time
import os
import argparse
from datetime import datetime

# Resolve project root: one level up from the yolo/ directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Default paths
DEFAULT_CODE_BASE = SCRIPT_DIR
DEFAULT_SOURCE = os.path.join(PROJECT_ROOT, "BH_CT_Data", "yolo_dataset", "test", "images")


def check_environment():
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available")


def predict_with_subprocess(model_path, source, img_size, conf, iou, device_num,
                            save_txt, save_conf, save_crop, code_base_folder, name):
    """Run YOLO predict via subprocess, logging output to file."""

    # Convert paths to absolute for reliability
    model_path = os.path.abspath(model_path).replace('\\', '/')
    source = os.path.abspath(source).replace('\\', '/')

    base_log_dir = os.path.join(code_base_folder, "predict_logs")
    os.makedirs(base_log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not name:
        name = f"predict_{timestamp}"

    original_name = name
    log_dir = os.path.join(base_log_dir, name)
    counter = 1
    
    while os.path.exists(log_dir):
        name = f"{original_name}_{counter}"
        log_dir = os.path.join(base_log_dir, name)
        counter += 1
        
    os.makedirs(log_dir)
    # ---------------------------------------------------

    predict_log_file = os.path.join(log_dir, f"{name}.log")
    time_log_file = os.path.join(log_dir, f"{name}_execution_times.txt")
    
    yolo_result_dir = os.path.abspath(os.path.join(code_base_folder, "runs", "detect", name)).replace('\\', '/')

    print("=" * 72)
    print(f"Prediction started with model={model_path}")
    print(f"  source    : {source}")
    print(f"  imgsz     : {img_size}")
    print(f"  conf      : {conf}")
    print(f"  iou       : {iou}")
    print(f"  device    : {device_num}")
    print(f"  save_txt  : {save_txt}")
    print(f"  save_conf : {save_conf}")
    print(f"  save_crop : {save_crop}")
    print(f"  name      : {name}") 
    print("=" * 72)

    predict_command = [
        "yolo", "detect", "predict",
        f"model={model_path}",
        f"source={source}",
        f"imgsz={img_size}",
        f"conf={conf}",
        f"iou={iou}",
        f"device={device_num}",
        f"name={name}",
        f"save_txt={save_txt}",
        f"save_conf={save_conf}",
        f"save_crop={save_crop}",
        "save=True",
    ]

    start_time = time.time()
    start_time_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(predict_log_file, "w") as file:
        subprocess.run(predict_command, stdout=file, stderr=subprocess.STDOUT)

    end_time = time.time()
    end_time_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    clean_log_path = os.path.abspath(predict_log_file).replace('\\', '/') # f-string 밖으로 빼기!
    
    print("=" * 72)
    print(f"Prediction completed for {name}.")
    print(f"  Script Logs saved to : {clean_log_path}")
    print(f"  YOLO Results saved to: {yolo_result_dir}")
    print("-" * 72)
    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("=" * 72)

    with open(time_log_file, "a") as time_log:
        time_log.write(
            f"{name}: Start Time: {start_time_formatted}, "
            f"End Time: {end_time_formatted}, "
            f"Elapsed Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n"
        )


def batch_predict(model_path, source, img_sizes, confs, ious, device_num,
                  save_txt, save_conf, save_crop, code_base_folder):
    """Run predictions across multiple image sizes, confidence, and IoU thresholds."""

    for img_size in img_sizes:
        for conf in confs:
            for iou in ious:
                name = f"predict_imgsz{img_size}_conf{conf}_iou{iou}"
                predict_with_subprocess(
                    model_path=model_path,
                    source=source,
                    img_size=img_size,
                    conf=conf,
                    iou=iou,
                    device_num=device_num,
                    save_txt=save_txt,
                    save_conf=save_conf,
                    save_crop=save_crop,
                    code_base_folder=code_base_folder,
                    name=name,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO Prediction Script for Tree Canopy Detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained YOLO model weights (.pt)\n"
             "Example: runs/detect/<train_name>/weights/best.pt"
    )
    parser.add_argument(
        "--source", type=str, default=DEFAULT_SOURCE,
        help=f"Path to images (directory or single file) for prediction.\n"
             f"(default: {DEFAULT_SOURCE})"
    )
    parser.add_argument(
        "--img-sizes", nargs="+", default=[320], type=int,
        help="List of image sizes for prediction (default: 320)"
    )
    parser.add_argument(
        "--confs", nargs="+", default=[0.5], type=float,
        help="List of confidence thresholds (default: 0.5)"
    )
    parser.add_argument(
        "--ious", nargs="+", default=[0.5], type=float,
        help="List of IoU thresholds for NMS (default: 0.5)"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="CUDA device number or 'cpu' (default: 0)"
    )
    parser.add_argument(
        "--save-txt", action="store_true", default=False,
        help="Save detection results as .txt files"
    )
    parser.add_argument(
        "--save-conf", action="store_true", default=False,
        help="Include confidence scores in saved .txt results"
    )
    parser.add_argument(
        "--save-crop", action="store_true", default=False,
        help="Save cropped detection images"
    )
    parser.add_argument(
        "--code-base-folder", type=str, default=DEFAULT_CODE_BASE,
        help=f"Base folder for storing prediction runs and logs.\n"
             f"(default: {DEFAULT_CODE_BASE})"
    )

    args = parser.parse_args()

    check_environment()

    batch_predict(
        model_path=args.model,
        source=args.source,
        img_sizes=args.img_sizes,
        confs=args.confs,
        ious=args.ious,
        device_num=args.device,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        code_base_folder=args.code_base_folder,
    )