import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score

from evaluate import Detector
from utils import get_list, seed_torch
from test_adm import ADMTestDataset


def evaluate_ensemble(y_pred, y_true):
    y_pred_arr = np.array(y_pred)
    y_true_arr = np.array(y_true)
    y_pred_bin = (y_pred_arr > 0.5).astype(int)
    ap = average_precision_score(y_true_arr, y_pred_arr)
    acc = accuracy_score(y_true_arr, y_pred_bin)
    prec = precision_score(y_true_arr, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_bin, zero_division=0)
    return ap, acc, prec, rec, f1


def main(args):
    seed_torch(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    print("\n" + "="*65)
    print("      CO-SPY DUAL-EXPERT ENSEMBLE (ProGAN + SD-v1.4)     ")
    print("="*65)
    print(f" Device          : {device}")
    print(f" Checkpoint Dir  : {args.ckpt}")
    print(f" Strategy        : {args.strategy.upper()} (e.g. max(p_progan, p_sd))")
    print(f" Batch Size      : {args.batch_size}")
    print(f" Target Model    : {args.model}")
    print("="*65 + "\n")

    # Resolve dataset path
    data_path = args.data_path
    if data_path is None:
        data_path = f"data/test/AIGCDetectionBenchMark/test/{args.model}"
    print(f"[1/4] Scanning dataset at: {data_path}")
    test_dataset = ADMTestDataset(data_path)
    print(f"       Found {len(test_dataset.real_list)} Real, {len(test_dataset.fake_list)} Fake (Total: {len(test_dataset)})")

    # Load Detector 1: ProGAN expert
    print("\n[2/4] Loading Detector 1 (ProGAN Expert - GAN Specialist)...")
    detector_progan = Detector(
        device=device,
        mode="fusion",
        fusion_type=args.fusion_type,
        train_dataset="progan",
        ckpt=args.ckpt,
        batch_size=args.batch_size,
    )

    # Load Detector 2: SD-v1.4 expert
    print("[3/4] Loading Detector 2 (SD-v1.4 Expert - Diffusion Specialist)...")
    detector_sd = Detector(
        device=device,
        mode="fusion",
        fusion_type=args.fusion_type,
        train_dataset="sd-v1_4",
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        use_freq=True,
    )

    # Inference loop
    print("\n[4/4] Running Dual-Expert Ensemble inference...")
    y_true = []
    y_pred_progan = []
    y_pred_sd = []

    # Using test_dataset with raw PIL loading in batch
    loader = DataLoader(test_dataset.total_list, batch_size=args.batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_paths in tqdm(loader, desc=f"Evaluating Ensemble on {args.model}"):
            # Batch transform for progan detector
            imgs_progan = []
            imgs_sd = []
            batch_labels = []
            for p in batch_paths:
                img_pil = Image.open(p).convert("RGB")
                imgs_progan.append(detector_progan.model.test_transform(img_pil))
                imgs_sd.append(detector_sd.model.test_transform(img_pil))
                batch_labels.append(test_dataset.labels_dict[p])

            tensor_progan = torch.stack(imgs_progan).to(device)
            tensor_sd = torch.stack(imgs_sd).to(device)

            preds_1 = detector_progan.model.predict(tensor_progan)
            preds_2 = detector_sd.model.predict(tensor_sd)

            y_pred_progan.extend(preds_1)
            y_pred_sd.extend(preds_2)
            y_true.extend(batch_labels)

    # Calculate individual metrics
    ap1, acc1, prec1, rec1, f1_1 = evaluate_ensemble(y_pred_progan, y_true)
    ap2, acc2, prec2, rec2, f1_2 = evaluate_ensemble(y_pred_sd, y_true)

    # Calculate ensemble
    arr1 = np.array(y_pred_progan)
    arr2 = np.array(y_pred_sd)
    if args.strategy == "max":
        arr_ens = np.maximum(arr1, arr2)
    elif args.strategy == "mean":
        arr_ens = (arr1 + arr2) / 2.0
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    ap_ens, acc_ens, prec_ens, rec_ens, f1_ens = evaluate_ensemble(arr_ens, y_true)

    # Print comparison table
    print("\n" + "="*70)
    print(f"             COMPARISON RESULTS ON {args.model.upper()}")
    print("="*70)
    print(f" {'Detector':25s} | {'AP':8s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':8s}")
    print("-" * 70)
    print(f" {'1. ProGAN Detector':25s} | {ap1*100:6.2f}% | {acc1*100:7.2f}% | {prec1*100:8.2f}% | {rec1*100:7.2f}%")
    print(f" {'2. SD-v1.4 Detector':25s} | {ap2*100:6.2f}% | {acc2*100:7.2f}% | {prec2*100:8.2f}% | {rec2*100:7.2f}%")
    print("-" * 70)
    print(f" {'3. ENSEMBLE (MAX)':25s} | {ap_ens*100:6.2f}% | {acc_ens*100:7.2f}% | {prec_ens*100:8.2f}% | {rec_ens*100:7.2f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dual-Expert Ensemble (ProGAN + SD-v1.4)")
    parser.add_argument("--model", type=str, default="stable_diffusion_v_1_4",
                        help="Target generative model (e.g. stable_diffusion_v_1_4, Midjourney, ADM, progan)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset directory (defaults to data/test/AIGCDetectionBenchMark/test/<model>)")
    parser.add_argument("--ckpt", type=str, default="ckpt_improved",
                        help="Checkpoint root directory")
    parser.add_argument("--fusion_type", type=str, default="attention",
                        choices=["attention", "linear"],
                        help="Fusion type for both detectors")
    parser.add_argument("--strategy", type=str, default="max",
                        choices=["max", "mean"],
                        help="Ensemble strategy: max (highest score) or mean (average)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id")
    parser.add_argument("--seed", type=int, default=1024,
                        help="Random seed")

    args = parser.parse_args()
    main(args)
