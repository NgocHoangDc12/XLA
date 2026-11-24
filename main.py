import torch
from train import Trainer
from evaluate import Detector
from utils import seed_torch


def main(args):
    #########################################
    # Phase 1: Training
    #########################################
    if args.phase == "train":
        # Initialize Trainer
        trainer = Trainer(
            mode=args.mode,
            device=args.device,
            branch=args.branch,
            train_dataset=args.train_dataset,
            ckpt=args.ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            feat_interp=args.feat_interp,
            feat_interp_alpha=args.feat_interp_alpha,
            feat_interp_ratio=args.feat_interp_ratio,
            fusion_type=args.fusion_type,
            use_freq=args.use_freq,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
        )
        # Start training
        trainer.train()
    #########################################
    # Phase 2: Evaluation
    #########################################
    elif args.phase == "eval":
        # Initialize Detector
        detector = Detector(
            device=args.device,
            mode=args.mode,
            train_dataset=args.train_dataset,
            pretrain=args.pretrain,
            ckpt=args.ckpt,
            batch_size=args.batch_size,
            branch=args.branch,
            fusion_type=args.fusion_type,
            use_freq=args.use_freq,
        )
        # Start evaluation
        detector.evaluate_benchmark(eval_models=args.eval_models, data_root=args.data_root, eval_datasets=args.eval_datasets, no_save=args.no_save)
    ##########################################
    # Phase 3: Test on a single image
    ##########################################
    elif args.phase == "test":
        # Initialize Detector
        detector = Detector(
            device=args.device,
            mode=args.mode,
            train_dataset=args.train_dataset,
            pretrain=args.pretrain,
            ckpt=args.ckpt,
            batch_size=args.batch_size,
            branch=args.branch,
            fusion_type=args.fusion_type,
            use_freq=args.use_freq,
            gradcam=args.gradcam,
            gradcam_save_dir=args.gradcam_save_dir,
            test_image=args.test_image,
        )
        # Test on a single image
        score = detector.scan(args.test_image)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="GPU id to use")
    parser.add_argument("--phase",
                        type=str,
                        default="test",
                        choices=["train", "eval", "test"],
                        help="Select the phase to run Co-Spy: train / eval / test")
    parser.add_argument("--mode",
                        type=str,
                        default="fusion",
                        choices=["branch", "fusion", "end2end"],
                        help="Select the mode of Co-Spy training")
    parser.add_argument("--train_dataset",
                        type=str,
                        default="sd-v1_4",
                        help="Training dataset")
    parser.add_argument("--branch",
                        type=str,
                        default="artifact",
                        choices=["artifact", "semantic"],
                        help="Branch detector (for branch mode)")
    parser.add_argument("--pretrain",
                        action="store_true",
                        help="Whether to use pre-trained weights for evaluation")
    parser.add_argument("--ckpt",
                        type=str,
                        default="ckpt",
                        help="Checkpoint directory")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size")
    parser.add_argument("--seed",
                        type=int,
                        default=1024,
                        help="Random seed")
    parser.add_argument("--feat_interp",
                        action="store_true",
                        default=False,
                        help="Feature-space interpolation for semantic branch.")
    parser.add_argument("--feat_interp_alpha",
                        type=float,
                        default=0.2,
                        help="Beta(alpha, alpha) shape for feat_interp.")
    parser.add_argument("--feat_interp_ratio",
                        type=float,
                        default=0.5,
                        help="Fraction of batch to apply feat_interp.")

    # Novel improvements
    parser.add_argument("--fusion_type",
                        type=str,
                        default="linear",
                        choices=["linear", "attention"],
                        help="Fusion architecture: linear (baseline) or attention (Cross-Attention Gated Fusion)")
    parser.add_argument("--use_freq",
                        action="store_true",
                        default=False,
                        help="Enable 2D FFT frequency branch in artifact detector")
    parser.add_argument("--loss_type",
                        type=str,
                        default="bce",
                        choices=["bce", "focal"],
                        help="Loss function: bce (baseline) or focal (Focal Loss)")
    parser.add_argument("--focal_gamma",
                        type=float,
                        default=2.0,
                        help="Gamma focusing parameter for Focal Loss")
    parser.add_argument("--focal_alpha",
                        type=float,
                        default=0.25,
                        help="Alpha balance parameter for Focal Loss")
    parser.add_argument("--gradcam",
                        action="store_true",
                        default=False,
                        help="Enable Grad-CAM visual explanation generation on test images")
    parser.add_argument("--gradcam_save_dir",
                        type=str,
                        default="gradcam_results",
                        help="Directory to save Grad-CAM visualizations")
    parser.add_argument("--test_image",
                        type=str,
                        default="",
                        help="Direct path to an image for test phase (avoids interactive prompt)")
    parser.add_argument("--eval_models",
                        nargs="+",
                        default=None,
                        help="Specific model(s) to evaluate in benchmark (e.g. ADM)")
    parser.add_argument("--eval_datasets",
                        nargs="+",
                        default=None,
                        help="Specific source dataset(s) to evaluate in benchmark (e.g. cc3m, mscoco, flickr)")
    parser.add_argument("--no_save",
                        action="store_true",
                        default=False,
                        help="Do not save output JSON and result files, just print results to console")
    parser.add_argument("--data_root",
                        type=str,
                        default=None,
                        help="Custom root directory for test benchmark (e.g. on Google Drive)")


    args = parser.parse_args()

    # Set random seed
    seed_torch(args.seed)

    # Set GPU device
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Run the experiment
    main(args)
