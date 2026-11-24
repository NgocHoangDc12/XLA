import os
import sys
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

from evaluate import Detector
from utils import get_list, seed_torch


class ADMTestDataset(Dataset):
    """Dataset dedicated to testing the ADM subset of AIGCDetectionBenchMark."""
    def __init__(self, adm_dir, transform=None):
        self.adm_dir = adm_dir
        self.transform = transform

        if not os.path.exists(adm_dir):
            raise FileNotFoundError(f"ADM directory not found: {adm_dir}")

        # Collect 0_real and 1_fake images
        real_list = get_list(adm_dir, must_contain='0_real')
        fake_list = get_list(adm_dir, must_contain='1_fake')

        if len(real_list) == 0 and len(fake_list) == 0:
            # Fallback if folders are named 'real' and 'fake'
            real_list = get_list(adm_dir, must_contain='real')
            fake_list = get_list(adm_dir, must_contain='fake')

        self.real_list = real_list
        self.fake_list = fake_list

        self.labels_dict = {}
        for p in self.real_list:
            self.labels_dict[p] = 0
        for p in self.fake_list:
            self.labels_dict[p] = 1

        self.total_list = self.real_list + self.fake_list
        np.random.shuffle(self.total_list)

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        path = self.total_list[idx]
        label = self.labels_dict[path]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def run_adm_test(args):
    seed_torch(args.seed)
    if args.data_path is None:
        args.data_path = f"data/test/AIGCDetectionBenchMark/test/{args.model}"
    
    target_name = args.model if args.data_path.endswith(args.model) else os.path.basename(args.data_path.rstrip("/\\"))
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    print(f"\n=======================================================")
    print(f"            Co-Spy Evaluation on {target_name.upper()}            ")
    print(f"=======================================================")
    print(f" Target Model    : {target_name}")
    print(f" Device          : {device}")
    print(f" Checkpoint Dir  : {args.ckpt}")
    print(f" Train Dataset   : {args.train_dataset}")
    print(f" Mode            : {args.mode}")
    if args.mode == "fusion":
        print(f" Fusion Type     : {args.fusion_type}")
    elif args.mode == "branch":
        print(f" Branch          : {args.branch}")
    print(f" Frequency FFT   : {args.use_freq}")
    print(f" Dataset Path    : {args.data_path}")
    print(f" Batch Size      : {args.batch_size}")
    print(f"=======================================================\n")

    # 1. Initialize Detector
    print("[1/3] Loading trained weights into Detector...")
    try:
        detector = Detector(
            device=device,
            mode=args.mode,
            branch=args.branch,
            train_dataset=args.train_dataset,
            pretrain=args.pretrain,
            ckpt=args.ckpt,
            batch_size=args.batch_size,
            fusion_type=args.fusion_type,
            use_freq=args.use_freq,
        )
    except Exception as e:
        print(f"[Error] Failed to initialize Detector or load weights: {e}")
        print("Please verify your --ckpt path and folder structure.")
        return

    # 2. Load ADM dataset
    print(f"[2/3] Scanning dataset in: {args.data_path}")
    try:
        test_dataset = ADMTestDataset(args.data_path, transform=detector.model.test_transform)
    except Exception as e:
        print(f"[Error] {e}")
        return

    num_real = len(test_dataset.real_list)
    num_fake = len(test_dataset.fake_list)
    total_imgs = len(test_dataset)

    print(f"       Found {num_real} real images (0_real)")
    print(f"       Found {num_fake} fake images (1_fake)")
    print(f"       Total: {total_imgs} images")

    if total_imgs == 0:
        print("[Error] No images found. Make sure images contain '0_real' and '1_fake' in their paths.")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.startswith("cuda") else False,
    )

    # 3. Inference
    print("\n[3/3] Running inference...")
    y_pred, y_true = [], []
    for images, labels in tqdm(test_loader, desc="Testing ADM"):
        preds = detector.model.predict(images)
        y_pred.extend(preds)
        y_true.extend(labels.tolist())

    y_pred_arr = np.array(y_pred)
    y_true_arr = np.array(y_true)
    y_pred_binary = (y_pred_arr > 0.5).astype(int)

    # Metrics
    ap = average_precision_score(y_true_arr, y_pred_arr)
    acc = (y_pred_binary == y_true_arr).mean()
    prec = precision_score(y_true_arr, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_binary, zero_division=0)

    print(f"\n==================== Evaluation Results ====================")
    print(f" Dataset         : {target_name} (AIGCDetectionBenchMark)")
    print(f" Total Samples   : {total_imgs} (Real: {num_real}, Fake: {num_fake})")
    print(f" Average Precision (AP) : {ap * 100:.2f}%")
    print(f" Accuracy (ACC)         : {acc * 100:.2f}%")
    print(f" Precision              : {prec * 100:.2f}%")
    print(f" Recall                 : {rec * 100:.2f}%")
    print(f" F1-Score               : {f1 * 100:.2f}%")
    print(f"============================================================\n")

    # Save results if requested
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        results = {
            "dataset": target_name,
            "samples": total_imgs,
            "real_samples": num_real,
            "fake_samples": num_fake,
            "AP": float(ap),
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to: {args.save_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Co-Spy model on ADM or other generative models")
    parser.add_argument("--model", type=str, default="ADM",
                        help="Generator model name in AIGCDetectionBenchMark (e.g. ADM, Midjourney, DALLE2, Glide, progan, biggan, etc.)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset directory (defaults to data/test/AIGCDetectionBenchMark/test/<model>)")
    parser.add_argument("--ckpt", type=str, default="ckpt_improved",
                        help="Root directory of checkpoints (e.g., ckpt_improved or ckpt)")
    parser.add_argument("--train_dataset", type=str, default="progan",
                        choices=["progan", "sd-v1_4", "writing"],
                        help="Model architecture / trained dataset (progan / sd-v1_4 / writing)")
    parser.add_argument("--mode", type=str, default="fusion",
                        choices=["fusion", "branch", "end2end"],
                        help="Evaluation mode (fusion / branch / end2end)")
    parser.add_argument("--branch", type=str, default="artifact",
                        choices=["artifact", "semantic"],
                        help="Branch name if mode=branch")
    parser.add_argument("--fusion_type", type=str, default="attention",
                        choices=["attention", "linear"],
                        help="Fusion type (attention / linear)")
    parser.add_argument("--use_freq", action="store_true", default=False,
                        help="Whether artifact branch uses frequency FFT")
    parser.add_argument("--pretrain", action="store_true", default=False,
                        help="Use pretrain weights instead of ckpt")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Dataloader workers")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID (use -1 for CPU)")
    parser.add_argument("--seed", type=int, default=1024,
                        help="Random seed")
    parser.add_argument("--save_json", type=str, default="adm_eval_result.json",
                        help="Path to output result JSON file")

    args = parser.parse_args()
    run_adm_test(args)
