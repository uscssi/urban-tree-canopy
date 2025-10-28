import subprocess
import time
import os
import argparse
from datetime import datetime

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

def train_with_subprocess(yaml_file, img_sizes, code_base_folder):
    batch_size = 8
    learning_rate = 0.0001
    # lf = 0.01
    optimizer = "SGD"
    model_name = "yolov8x.pt"
    # model_name = "yolo12x.pt"
    epoch_num = 200
    device_num = "0"
    # os.makedirs("logs", exist_ok=True)
    dataset = os.path.splitext(os.path.basename(yaml_file))[0]

    for image_size in img_sizes:
        name = f"{dataset}_{optimizer}_{batch_size}_{learning_rate}_{model_name}_{image_size}_{epoch_num}"
        os.makedirs(f"train_logs/{name}", exist_ok=True)
        train_log_file = f"train_logs/{name}/{name}_train.log"
        val_log_file = f"train_logs/{name}/{name}_val.log"
        time_log_file = f"train_logs/{name}/{name}_execution_times.txt"
        print("=" * 72)
        print(f"Training with {yaml_file}, BatchSize={batch_size}, LearningRate={learning_rate}, ImageSize={image_size}, Model={model_name}, Name={name}")

        train_command = [
            "yolo", "detect", "train",
            f"data={yaml_file}",
            f"model={model_name}",
            f"epochs={epoch_num}",
            f"imgsz={image_size}",
            f"device={device_num}",
            f"batch={batch_size}",
            f"name={name}",
            f"lr0={learning_rate}",
            # f"lrf={lf}",
            f"optimizer={optimizer}",
            "workers=0",
            # "cos_lr=True",
            # "amp=True",
            # "val=True",
            "plots=True",
            # "close_mosaic=10"
            "save_period=50",
            # "patience=0"
        ]

        start_time = time.time()
        start_time_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(train_log_file, "w") as file:
            subprocess.run(train_command, stdout=file, stderr=subprocess.STDOUT)

        print(f"Training completed for {name}. Log saved to {train_log_file}")

        weights_file_path = f"{code_base_folder}/runs/detect/{name}/weights/best.pt"
        ious = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        confs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        for iou in ious:
            for conf in confs:
                val_command = [
                    "yolo", "val",
                    f"split='test'",
                    f"model={weights_file_path}",
                    f"data={yaml_file}",
                    f"imgsz={image_size}",
                    f"conf={conf}",
                    f"iou={iou}",
                    f"device={device_num}",
                    f"name={name}_val",
                    "save_json=True",
                    "plots=True"
                ]

                with open(val_log_file, "a") as file:
                    file.write(f"####Validation started with iou={iou} and conf={conf}###\n\n")
                    subprocess.run(val_command, stdout=file, stderr=subprocess.STDOUT)

                print(f"Validation completed for {name}. Log saved to {val_log_file}")

        end_time = time.time()
        end_time_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print("#" * 69)
        print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print("#" * 69)
        with open(time_log_file, "a") as time_log:
            time_log.write(f"{name}: Start Time: {start_time_formatted}, End Time: {end_time_formatted}, Elapsed Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO Training Script with a Single YAML File')
    parser.add_argument('--yaml-file', type=str, required=True, help='Path to a single YAML file')
    parser.add_argument('--img-sizes', nargs='+', default=[320], type=int, help='List of image sizes for training and validation')
    parser.add_argument('--code-base-folder', type=str, required=True, help='Path to YOLO base folder for storing runs and logs')
    
    args = parser.parse_args()

    check_environment()
    train_with_subprocess(args.yaml_file, args.img_sizes, args.code_base_folder)
