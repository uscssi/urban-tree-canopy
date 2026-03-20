#!/usr/bin/env python
import sys as _sys
# -*- coding: utf-8 -*-
"""
UNet (TIF) Training / Testing / Prediction Script - Binary Segmentation Edition
With Albumentations-based augmentation

Updates v8 (2026-01-26)
- Added Albumentations augmentation matching configured parameters: rotate, brightness, contrast, zoom, crop
- Fixed tensor dtype mismatch by normalizing and converting to FloatTensor
- Augmentations applied consistently across train/val/test/predict
- predict_mosaic now supports transforms
"""

import os
import time
import json
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
# --- Helpers ---
def make_unique_dir(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    idx = 1
    while True:
        new_path = f"{path}_{idx}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        idx += 1


def make_unique_file(path: str) -> str:
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        cand = f"{base}_{idx}{ext}"
        if not os.path.exists(cand):
            return cand
        idx += 1

# --- Albumentations Transforms ---
def get_transforms(mode="train"):
    """Return an Albumentations Compose for each mode."""
    if mode == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),

            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),  # CLAHE: Contrast Limited Adaptive Histogram Equalization
            A.RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=5, p=0.3),

            A.Normalize(mean=(0.0, 0.0, 0.0),
                        std=(1.0, 1.0, 1.0),
                        max_pixel_value=255.0),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})

    else:
        # For val/test/predict: normalize only (no augmentation)
        return A.Compose([
            A.Normalize(mean=(0.0, 0.0, 0.0),
                        std=(1.0, 1.0, 1.0),
                        max_pixel_value=255.0),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})

# --- Dataset ---
class TIFDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith('.tif')
        )
        if not self.image_files:
            raise ValueError(f"No .tif files found in {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fn = self.image_files[idx]
        img_path = os.path.join(self.images_dir, fn)
        lbl_path = os.path.join(self.labels_dir, fn)

        # Load image and mask from GeoTIFF files
        with rasterio.open(img_path) as src:
            img_np = src.read().astype(np.uint8)  # [C,H,W]
        with rasterio.open(lbl_path) as src:
            mask_np = src.read(1).astype(np.uint8)  # [H,W]

        img_np = np.transpose(img_np, (1, 2, 0))  # to HWC
        mask_np = (mask_np > 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_t = augmented['image']           # FloatTensor CHW
            mask_t = augmented['mask'].float()   # FloatTensor HW
        else:
            # Fallback: manual normalization when no transform is provided
            img_t = torch.from_numpy(img_np.transpose(2,0,1) / 255.0).float()
            mask_t = torch.from_numpy(mask_np).float()

        return img_t, mask_t

# --- Metrics ---
def compute_metrics(pred, target):
    p = pred.cpu().numpy().flatten()
    t = target.cpu().numpy().flatten()
    TP = np.sum((p == 1) & (t == 1))
    TN = np.sum((p == 0) & (t == 0))
    FP = np.sum((p == 1) & (t == 0))
    FN = np.sum((p == 0) & (t == 1))
    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = TP / (TP + FP + FN + eps)
    dice      = 2 * TP / (2 * TP + FP + FN + eps)
    return {'TP':TP,'TN':TN,'FP':FP,'FN':FN,
            'precision':precision,'recall':recall,
            'f1':f1,'iou':iou,'dice':dice}

# --- Model Loader ---
def get_unet_model(encoder='resnet34', weights='imagenet', pretrained=True, in_channels=3):
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights if pretrained else None,
        in_channels=in_channels,
        classes=1
    )

# --- Train / Eval ---
def train_one_epoch(model, loader, criterion, optimizer, device, thr):
    model.train()
    total_loss = 0.0
    agg = {'TP':0,'TN':0,'FP':0,'FN':0}
    n_batches = len(loader)
    for i, (imgs, lbls) in enumerate(loader, 1):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, lbls.unsqueeze(1))
        loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) > thr).long().squeeze(1)
        m = compute_metrics(preds, lbls.long())
        for k in agg: agg[k] += m[k]
        pct = i / n_batches * 100
        _sys.stdout.write(f'\r  [Train] {i}/{n_batches} ({pct:.0f}%) loss={loss.item():.4f}')
        _sys.stdout.flush()
    print()  # newline after progress

    P = agg['TP']/(agg['TP']+agg['FP']+1e-8)
    R = agg['TP']/(agg['TP']+agg['FN']+1e-8)
    f1 = 2*P*R/(P+R+1e-8)
    iou = agg['TP']/(agg['TP']+agg['FP']+agg['FN']+1e-8)
    dice = 2*agg['TP']/(2*agg['TP']+agg['FP']+agg['FN']+1e-8)
    avg_loss = total_loss/len(loader.dataset)
    return avg_loss, {'precision':P,'recall':R,'f1':f1,'iou':iou,'dice':dice, **agg}

@torch.no_grad()
def evaluate(model, loader, criterion, device, thr, phase="Val"):
    model.eval()
    total_loss = 0.0
    agg = {'TP':0,'TN':0,'FP':0,'FN':0}
    n_batches = len(loader)
    for i, (imgs, lbls) in enumerate(loader, 1):
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        loss   = criterion(logits, lbls.unsqueeze(1))
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) > thr).long().squeeze(1)
        m = compute_metrics(preds, lbls.long())
        for k in agg: agg[k] += m[k]
        pct = i / n_batches * 100
        _sys.stdout.write(f'\r  [{phase}] {i}/{n_batches} ({pct:.0f}%)')
        _sys.stdout.flush()
    print()  # newline after progress

    P = agg['TP']/(agg['TP']+agg['FP']+1e-8)
    R = agg['TP']/(agg['TP']+agg['FN']+1e-8)
    f1 = 2*P*R/(P+R+1e-8)
    iou = agg['TP']/(agg['TP']+agg['FP']+agg['FN']+1e-8)
    dice = 2*agg['TP']/(2*agg['TP']+agg['FP']+agg['FN']+1e-8)
    avg_loss = total_loss/len(loader.dataset)
    return avg_loss, {'precision':P,'recall':R,'f1':f1,'iou':iou,'dice':dice, **agg}

# --- Plot and CSV utilities ---
def plot_curves(train_losses, val_losses, train_metrics, val_metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses)+1)
    plt.figure(); plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.savefig(os.path.join(out_dir,'loss_curve.png')); plt.close()
    plt.figure()
    plt.plot(epochs, train_metrics['precision'], label='Train P')
    plt.plot(epochs, val_metrics['precision'],   label='Val P')
    plt.plot(epochs, train_metrics['recall'],    label='Train R')
    plt.plot(epochs, val_metrics['recall'],      label='Val R')
    plt.xlabel('Epoch'); plt.ylabel('Value'); plt.legend(); plt.title('Precision & Recall')
    plt.savefig(os.path.join(out_dir,'precision_recall.png')); plt.close()
    plt.figure(); plt.plot(epochs, train_metrics['f1'], label='Train F1')
    plt.plot(epochs, val_metrics['f1'],   label='Val F1')
    plt.xlabel('Epoch'); plt.ylabel('F1'); plt.legend(); plt.title('F1 Score')
    plt.savefig(os.path.join(out_dir,'f1_curve.png')); plt.close()

def write_confusion_csv(path, epochs, losses, metrics_list):
    head = ["epoch","loss","TP","TN","FP","FN","precision","recall","f1","iou","dice"]
    with open(path, 'w', newline='') as cf:
        w = csv.DictWriter(cf, fieldnames=head); w.writeheader()
        for ep,l,m in zip(epochs, losses, metrics_list):
            row = {"epoch":ep,"loss":l}; row.update({k:m[k] for k in head[2:]}); w.writerow(row)
    print(f"Saved CSV to {path}")
    
# --- GeoTIFF Helpers ---
def write_world_file(tif_path, transform):
    a,b,d,e,c,f = transform.a,transform.b,transform.d,transform.e,transform.c,transform.f
    lines = [f"{a:.10f}",f"{b:.10f}",f"{d:.10f}",f"{e:.10f}",
             f"{c + a/2:.10f}",f"{f + e/2:.10f}"]
    with open(os.path.splitext(tif_path)[0]+".tfw",'w') as fw: fw.write("\n".join(lines))

def predict_mosaic(model, input_dir, output_tif, device, thr, transform=None):
    model.eval()
    tif_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.tif'))
    if not tif_files:
        raise ValueError(f"No tif files in {input_dir}")
    datasets = []
    for fn in tif_files:
        with rasterio.open(os.path.join(input_dir, fn)) as src:
            img_np = src.read().astype(np.uint8)  # [C,H,W]
            meta, tr = src.meta.copy(), src.transform
        img_np = np.transpose(img_np, (1,2,0))  # HWC
        if transform:
            img_t = transform(image=img_np)['image']
        else:
            img_t = torch.from_numpy(img_np.transpose(2,0,1) / 255.0).float()
        inp = img_t.unsqueeze(0).to(device)
        logits = model(inp)
        prob   = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
        pred   = (prob > thr).astype(np.uint8)

        mem = MemoryFile()
        prof = dict(driver='GTiff', height=pred.shape[0], width=pred.shape[1],
                    count=1, dtype=pred.dtype, crs=meta.get('crs'), transform=tr)
        ds = mem.open(**prof); ds.write(pred, 1); datasets.append(ds)

    mosaic, m_transform = merge(datasets)
    out_meta = datasets[0].meta.copy()
    out_meta.update(height=mosaic.shape[1], width=mosaic.shape[2],
                    transform=m_transform, count=1)
    output_tif = make_unique_file(output_tif); os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    with rasterio.open(output_tif, 'w', **out_meta) as dst: dst.write(mosaic)
    print(f"Mosaic saved to {output_tif}")
    write_world_file(output_tif, m_transform)
    for ds in datasets: ds.close()

# --- Main ---
def main():
    import yaml as _yaml

    parser = argparse.ArgumentParser(description="UNet binary segmentation with augmentation")
    subs = parser.add_subparsers(dest="mode", required=True)

    def add_common_args(p):
        p.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (dataset_path.yaml). "
                             "Values from YAML are used as defaults; CLI flags override.")
        p.add_argument("--threshold", type=float, default=None, help="sigmoid threshold")
        p.add_argument("--use_aug", action="store_true", help="Use Albumentations augmentation")

    tr = subs.add_parser("train")
    tr.add_argument("--train_images", default=None)
    tr.add_argument("--train_labels", default=None)
    tr.add_argument("--val_images",   default=None)
    tr.add_argument("--val_labels",   default=None)
    tr.add_argument("--test_images",  default=None)
    tr.add_argument("--test_labels",  default=None)
    tr.add_argument("--epochs",       type=int,   default=None)
    tr.add_argument("--batch_size",   type=int,   default=None)
    tr.add_argument("--lr",           type=float, default=None)
    tr.add_argument("--scheduler_on", action="store_true")
    tr.add_argument("--encoder",      type=str,   default="resnet34")
    tr.add_argument("--weights",      type=str,   default="imagenet")
    tr.add_argument("--in_channels",  type=int,   default=3)
    add_common_args(tr)

    te = subs.add_parser("test")
    te.add_argument("--test_images", default=None)
    te.add_argument("--test_labels", default=None)
    te.add_argument("--batch_size",  type=int, default=None)
    te.add_argument("--model_path",  default=None)
    te.add_argument("--encoder",     type=str, default="resnet34")
    te.add_argument("--in_channels", type=int, default=3)
    add_common_args(te)

    pp = subs.add_parser("predict")
    pp.add_argument("--predict_images", default=None)
    pp.add_argument("--output_tif",     default=None)
    pp.add_argument("--model_path",     default=None)
    pp.add_argument("--encoder",        type=str, default="resnet34")
    pp.add_argument("--in_channels",    type=int, default=3)
    add_common_args(pp)

    args = parser.parse_args()

    # --- Load YAML config and use as defaults ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    config_path = args.config
    if config_path is None:
        # Auto-detect dataset_path.yaml in the same directory
        default_cfg = os.path.join(script_dir, "dataset_path.yaml")
        if os.path.exists(default_cfg):
            config_path = default_cfg

    yaml_cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            yaml_cfg = _yaml.safe_load(f) or {}
        # Resolve relative paths in YAML
        path_keys = ["train_images", "train_labels", "val_images", "val_labels",
                      "test_images", "test_labels", "predict_images", "output_tif", "model_path"]
        for k in path_keys:
            v = str(yaml_cfg.get(k, ""))
            if v and not os.path.isabs(v):
                yaml_cfg[k] = os.path.normpath(os.path.join(project_root, v))
        print(f"Config loaded from: {config_path}")

    # Apply YAML values as defaults (CLI flags override)
    yaml_map = {
        "train_images": "train_images", "train_labels": "train_labels",
        "val_images": "val_images", "val_labels": "val_labels",
        "test_images": "test_images", "test_labels": "test_labels",
        "predict_images": "predict_images", "output_tif": "output_tif",
        "model_path": "model_path",
        "learning_rate": "lr", "batch_size": "batch_size",
        "epochs": "epochs", "threshold": "threshold",
        "use_aug": "use_aug",
    }
    for yaml_key, arg_name in yaml_map.items():
        if yaml_key in yaml_cfg and getattr(args, arg_name, None) is None:
            setattr(args, arg_name, yaml_cfg[yaml_key])

    # Final fallbacks for essential values
    if getattr(args, "lr", None) is None: args.lr = 1e-4
    if getattr(args, "batch_size", None) is None: args.batch_size = 4
    if getattr(args, "epochs", None) is None: args.epochs = 50
    if getattr(args, "threshold", None) is None: args.threshold = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare transforms based on augmentation flag
    
    if args.use_aug:
        train_tf = get_transforms("train")
    else:
        train_tf = get_transforms("val")  # no aug

    val_tf   = get_transforms("val")
    test_tf  = get_transforms("val")
    pred_tf  = get_transforms("val")


    # Determine output directories based on mode
    if args.mode == "train":
        base = f"{args.encoder}_bs{args.batch_size}_lr{args.lr}"
    elif args.mode == "test":
        base = f"{args.encoder}_bs{args.batch_size}"
    else:
        base = f"{args.encoder}_predict"
    root_dir = make_unique_dir(os.path.join(args.mode, base))

    # --- TRAIN MODE ---
    if args.mode == "train":
        print("\n" + "=" * 60)
        print("  MODE: TRAINING")
        print("=" * 60)
        csv_dir, plots_dir = [os.path.join(root_dir, p) for p in ("csv","plots")]
        models_dir = os.path.join(root_dir, "models")
        for d in (csv_dir, plots_dir, models_dir): os.makedirs(d, exist_ok=True)

        train_ds = TIFDataset(args.train_images, args.train_labels, transform=train_tf)
        val_ds   = TIFDataset(args.val_images,   args.val_labels,   transform=val_tf)
        train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
        val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_ld  = None
        if args.test_images and args.test_labels:
            td = TIFDataset(args.test_images, args.test_labels, transform=test_tf)
            test_ld = DataLoader(td, batch_size=args.batch_size, shuffle=False, num_workers=4)

        model     = get_unet_model(args.encoder, args.weights, True, args.in_channels).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1) if args.scheduler_on else None

        tlosses, vlosses, tmetrics, vmetrics = [], [], [], []
        best_dice, best_ep, best_info = 0.0, 0, {}

        start = time.time()
        for ep in range(1, args.epochs+1):
            print(f"\nEpoch {ep}/{args.epochs}")
            tl, tm = train_one_epoch(model, train_ld, criterion, optimizer, device, args.threshold)
            vl, vm = evaluate(model,   val_ld,   criterion, device, args.threshold, phase="Val")
            tlosses.append(tl); vlosses.append(vl)
            tmetrics.append(tm); vmetrics.append(vm)

            print(f"  Train Loss {tl:.4f}")
            print(f"  Val   Loss {vl:.4f} | Dice {vm['dice']:.4f} IoU {vm['iou']:.4f}"  
                  f" P {vm['precision']:.4f} R {vm['recall']:.4f}")

            if test_ld:
                tsl, tsm = evaluate(model, test_ld, criterion, device, args.threshold, phase="Test")
                print(f"  Test  Loss {tsl:.4f} | Dice {tsm['dice']:.4f} IoU {tsm['iou']:.4f}"
                      f" P {tsm['precision']:.4f} R {tsm['recall']:.4f}")

            if vm['dice'] > best_dice:
                best_dice, best_ep = vm['dice'], ep
                best_info = {'train_loss':tl,'val_loss':vl,'val_dice':vm['dice'],
                             'val_f1':vm['f1'],'val_iou':vm['iou']}
                if test_ld:
                    best_info.update({'test_loss':tsl,'test_dice':tsm['dice']})
                torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pt"))
                print("  >>> Best model saved.")
            if scheduler: scheduler.step()

        torch.save(model.state_dict(), os.path.join(models_dir, "last_model.pt"))
        with open(os.path.join(models_dir,"best_epoch_info.txt"),'w') as f:
            f.write(f"Best Epoch: {best_ep}\n")
            for k,v in best_info.items(): f.write(f"{k.replace('_',' ').title()}: {v:.4f}\n")

        plot_curves(tlosses, vlosses,
                    {k:[m[k] for m in tmetrics] for k in ('precision','recall','f1')},
                    {k:[m[k] for m in vmetrics] for k in ('precision','recall','f1')},
                    plots_dir)
        write_confusion_csv(os.path.join(csv_dir,"confusion_train.csv"),
                            range(1,args.epochs+1), tlosses, tmetrics)
        write_confusion_csv(os.path.join(csv_dir,"confusion_val.csv"),
                            range(1,args.epochs+1), vlosses, vmetrics)

        h,m,s = time.time() - start, 0, 0
        run_info = {
            'train_samples':len(train_ds),'val_samples':len(val_ds),
            'test_samples':len(test_ld.dataset) if test_ld else 0,
            'encoder':args.encoder,'batch_size':args.batch_size,
            'learning_rate':args.lr,'epochs':args.epochs,
            'threshold':args.threshold
        }
        with open(os.path.join(root_dir,"run_info.json"),"w") as jf:
            json.dump(run_info, jf, indent=4)
        print("Training complete.")

    # --- TEST MODE ---
    elif args.mode == "test":
        print("\n" + "=" * 60)
        print("  MODE: TESTING")
        print("=" * 60)
        test_ds = TIFDataset(args.test_images, args.test_labels, transform=test_tf)
        test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        model = get_unet_model(args.encoder, None, False, args.in_channels).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        criterion = nn.BCEWithLogitsLoss()
        loss, m = evaluate(model, test_ld, criterion, device, args.threshold, phase="Test")
        print(f"Test  Loss {loss:.4f} | Dice {m['dice']:.4f} IoU {m['iou']:.4f}"  
              f" P {m['precision']:.4f} R {m['recall']:.4f}")
        write_confusion_csv(os.path.join(root_dir,"confusion_test.csv"), [1], [loss], [m])

    # --- PREDICT MODE ---
    else:
        print("\n" + "=" * 60)
        print("  MODE: PREDICTION")
        print("=" * 60)
        model = get_unet_model(args.encoder, None, False, args.in_channels).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        out_name = os.path.basename(args.output_tif)
        out_path = make_unique_file(os.path.join(root_dir, out_name))
        predict_mosaic(model, args.predict_images, out_path,
                       device, args.threshold, transform=pred_tf)

if __name__ == "__main__":
    import warnings
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    main()
