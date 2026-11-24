from test_adm import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Co-Spy model on benchmark datasets (ADM, Midjourney, DALLE2, etc.)")
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
    parser.add_argument("--save_json", type=str, default=None,
                        help="Path to output result JSON file")
    parser.add_argument("--no_save", action="store_true", default=False,
                        help="Do not save result to JSON file")

    args = parser.parse_args()
    if args.no_save:
        args.save_json = None
    elif args.save_json is None:
        args.save_json = f"{args.model.lower()}_eval_result.json"
    run_adm_test(args)
