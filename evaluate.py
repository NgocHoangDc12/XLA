import os
import json
import torch
import importlib
import numpy as np
from tqdm import tqdm
from PIL import Image
from loguru import logger

from dataSets import *
from utils import evaluate

import warnings
warnings.filterwarnings("ignore")


# Load pre-trained models for evaluation
class Detector:
    def __init__(self,
                 device: str,
                 mode: str = "fusion",
                 branch: str = "artifact",
                 train_dataset: str = "sd-v1_4",
                 pretrain: bool = False,
                 ckpt: str = "ckpt",
                 batch_size: int = 32,
                 fusion_type: str = "linear",
                 use_freq: bool = False,
                 gradcam: bool = False,
                 gradcam_save_dir: str = "gradcam_results",
                 test_image: str = ""):

        # Device
        self.device = device
        self.mode = mode
        self.branch = branch
        self.train_dataset = train_dataset
        self.pretrain = pretrain
        self.ckpt = ckpt
        self.batch_size = batch_size
        self.fusion_type = fusion_type
        self.use_freq = use_freq
        self.gradcam = gradcam
        self.gradcam_save_dir = gradcam_save_dir
        self.test_image = test_image

        # Dynamically import the detector module from either "progan" or "sd-v1_4"
        detector_arch = "progan" if train_dataset == "writing" else train_dataset
        detector_module = importlib.import_module(f"detectors.{detector_arch}")

        # Auto-detect use_freq from artifact checkpoint if trained with frequency branch
        if not self.use_freq and not self.pretrain:
            art_w = os.path.join(self.ckpt, self.train_dataset, "artifact", "best_model.pth")
            if os.path.exists(art_w):
                try:
                    chk = torch.load(art_w, map_location='cpu')
                    if 'fc.weight' in chk and chk['fc.weight'].shape[1] == 640:
                        self.use_freq = True
                except Exception:
                    pass

        # Get the detector and load weights based on mode
        if self.mode == "branch":
            if self.branch == "artifact":
                DetectorClass = getattr(detector_module, "ArtifactDetector")
                self.model = DetectorClass(use_freq=self.use_freq)
            elif self.branch == "semantic":
                DetectorClass = getattr(detector_module, "SemanticDetector")
                self.model = DetectorClass()
            else:
                raise ValueError(f"Unknown branch: {self.branch}")
            if pretrain:
                weights_path = f"pretrained/{self.train_dataset}/{self.branch}_weights.pth"
            else:
                weights_path = os.path.join(self.ckpt, self.train_dataset, self.branch, "best_model.pth")
            if not os.path.exists(weights_path):
                raise ValueError(f"{self.branch} weights path does not exist: {weights_path}")
            self.model.load_weights(weights_path)
        elif self.mode == "fusion":
            fusion_subdir = f"fusion_{self.fusion_type}" if self.fusion_type != "linear" else "fusion"
            if pretrain:
                semantic_weights_path = f"pretrained/{self.train_dataset}/semantic_weights.pth"
                artifact_weights_path = f"pretrained/{self.train_dataset}/artifact_weights.pth"
                fusion_weights_filename = "fusion_attention_weights.pth" if self.fusion_type == "attention" else "fusion_weights.pth"
                fusion_weights_path = f"pretrained/{self.train_dataset}/{fusion_weights_filename}"
                if self.fusion_type == "attention" and not os.path.exists(fusion_weights_path):
                    fusion_weights_path = f"pretrained/{self.train_dataset}/fusion_weights.pth"
            else:
                semantic_weights_path = os.path.join(self.ckpt, self.train_dataset, "semantic", "best_model.pth")
                artifact_weights_path = os.path.join(self.ckpt, self.train_dataset, "artifact", "best_model.pth")
                fusion_weights_path = os.path.join(self.ckpt, self.train_dataset, fusion_subdir, "best_model.pth")
                if not os.path.exists(fusion_weights_path):
                    # Check for hyphenated directory name (e.g. fusion-attention)
                    alt_fusion_path = os.path.join(self.ckpt, self.train_dataset, f"fusion-{self.fusion_type}", "best_model.pth")
                    if os.path.exists(alt_fusion_path):
                        fusion_weights_path = alt_fusion_path
                    else:
                        for fname in ["final_model.pth", "best_model.pth"]:
                            p1 = os.path.join(self.ckpt, self.train_dataset, fusion_subdir, fname)
                            p2 = os.path.join(self.ckpt, self.train_dataset, f"fusion-{self.fusion_type}", fname)
                            if os.path.exists(p1):
                                fusion_weights_path = p1
                                break
                            elif os.path.exists(p2):
                                fusion_weights_path = p2
                                break
            if not os.path.exists(semantic_weights_path) or not os.path.exists(artifact_weights_path) or not os.path.exists(fusion_weights_path):
                raise ValueError(f"Weights path missing for {fusion_subdir} mode: check {semantic_weights_path}, {artifact_weights_path}, {fusion_weights_path}")

            if self.fusion_type == "attention":
                CoSpyAttentionFusionDetector = getattr(detector_module, "CoSpyAttentionFusionDetector")
                self.model = CoSpyAttentionFusionDetector(
                    semantic_weights_path=semantic_weights_path,
                    artifact_weights_path=artifact_weights_path,
                    use_freq=self.use_freq)
            else:
                CoSpyFusionDetector = getattr(detector_module, "CoSpyFusionDetector")
                self.model = CoSpyFusionDetector(
                    semantic_weights_path=semantic_weights_path,
                    artifact_weights_path=artifact_weights_path,
                    use_freq=self.use_freq)
            self.model.load_weights(fusion_weights_path)
        elif self.mode == "end2end":
            End2EndDetector = getattr(detector_module, "End2EndDetector")
            self.model = End2EndDetector()
            end2end_weights_path = os.path.join(self.ckpt, self.train_dataset, "end2end", "best_model.pth")
            if not os.path.exists(end2end_weights_path):
                raise ValueError("End2End weights path does not exist for end2end mode")
            self.model.load_weights(end2end_weights_path)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Put the model on the device and set to eval
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_benchmark(self, eval_models: list = None, data_root: str = None, eval_datasets: list = None, no_save: bool = False):
        # Select the appropriate test dataset and evaluation lists
        if self.train_dataset in ["progan", "writing"]:
            benchmark_name = "AIGCDetectionBenchMark"
            TestDataset = AIGCDetectTestDataset
            eval_dataset_list = eval_datasets if eval_datasets else AIGCDetectionBenchMark_DATASET_LIST
            eval_model_list = eval_models if eval_models else AIGCDetectionBenchMark_MODEL_LIST
        elif self.train_dataset == "sd-v1_4":
            benchmark_name = "Co-Spy-Bench"
            TestDataset = CoSpyBenchTestDataset
            eval_dataset_list = eval_datasets if eval_datasets else CoSpyBench_DATASET_LIST
            eval_model_list = eval_models if eval_models else CoSpyBench_MODEL_LIST
        else:
            raise ValueError(f"Unknown train dataset: {self.train_dataset}")

        # Set the saving directory (branch mode writes under <branch>/, others under <mode>/)
        if self.mode == "branch":
            subdir = self.branch
        elif self.mode == "fusion":
            subdir = f"fusion_{self.fusion_type}" if self.fusion_type != "linear" else "fusion"
        else:
            subdir = self.mode
        if self.pretrain:
            save_dir = os.path.join(self.ckpt, self.train_dataset, subdir, f"pretrain_{benchmark_name}")
        else:
            save_dir = os.path.join(self.ckpt, self.train_dataset, subdir, f"eval_{benchmark_name}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Setup logger
        log_path = f"{save_dir}/evaluation.log"
        if os.path.exists(log_path):
            os.remove(log_path)
        
        logger_id = logger.add(
            log_path,
            format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
            level="DEBUG",
        )

        # Save raw model prediction
        save_output_path = os.path.join(save_dir, "output.json")
        # Save summarized evaluation result
        save_result_path = os.path.join(save_dir, "result.json")

        # Begin the evaluation
        result_all = {}
        output_all = {}
        evaluated_pairs = []
        for dataset_name in eval_dataset_list:
            result_all[dataset_name] = {}
            output_all[dataset_name] = {}
            for model_name in eval_model_list:
                try:
                    if data_root is not None:
                        test_dataset = TestDataset(dataset=dataset_name, model=model_name, transform=self.model.test_transform, root_path=data_root)
                    else:
                        test_dataset = TestDataset(dataset=dataset_name, model=model_name, transform=self.model.test_transform)
                except Exception as e:
                    logger.warning(f"Failed to load dataset for {benchmark_name} - {dataset_name} - {model_name}: {e}")
                    continue

                if len(test_dataset) == 0:
                    logger.warning(f"No test images found for {benchmark_name} - {dataset_name} - {model_name}. Skipping...")
                    continue

                test_loader = torch.utils.data.DataLoader(test_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          num_workers=4,
                                                          pin_memory=True)

                # Evaluate the model
                y_pred, y_true = [], []
                with torch.no_grad():
                    for (images, labels) in tqdm(test_loader, desc=f"Evaluating {benchmark_name} - {dataset_name} - {model_name}"):
                        y_pred.extend(self.model.predict(images))
                        y_true.extend(labels.tolist())

                ap, accuracy = evaluate(y_pred, y_true)
                logger.info(f"Evaluate on {benchmark_name} - {dataset_name} - {model_name} | Size {len(y_true)} | AP {ap*100:.2f}% | Accuracy {accuracy*100:.2f}%")

                result_all[dataset_name][model_name] = {"size": len(y_true), "AP": ap, "Accuracy": accuracy}
                output_all[dataset_name][model_name] = {"y_pred": y_pred, "y_true": y_true}
                evaluated_pairs.append((dataset_name, model_name))

        # Save the results if not disabled
        if not no_save:
            with open(save_result_path, "w") as f:
                json.dump(result_all, f, indent=4)

            with open(save_output_path, "w") as f:
                json.dump(output_all, f, indent=4)
        
        # Print the average performance across evaluated datasets and models
        if evaluated_pairs:
            print("\n" + "="*60)
            print(f"       SUMMARY EVALUATION - {benchmark_name}")
            print("="*60)
            for d, m in evaluated_pairs:
                res = result_all[d][m]
                print(f" {d} - {m:35s} | AP: {res['AP']*100:.2f}% | Acc: {res['Accuracy']*100:.2f}%")
            print("-"*60)
            avg_ap = np.mean([result_all[d][m]["AP"] for d, m in evaluated_pairs])
            avg_accuracy = np.mean([result_all[d][m]["Accuracy"] for d, m in evaluated_pairs])
            print(f" AVERAGE AP: {avg_ap*100:.2f}% | AVERAGE ACC: {avg_accuracy*100:.2f}%")
            print("="*60 + "\n")
            logger.info(f"Average Performance on {benchmark_name} | AP {avg_ap*100:.2f}% | Accuracy {avg_accuracy*100:.2f}%")
        else:
            logger.warning("No models were successfully evaluated.")

    def scan(self, image_filepath=None):
        # Load the image (interactive or from argument)
        if image_filepath is None or image_filepath == "":
            image_filepath = self.test_image
        if image_filepath is None or image_filepath == "":
            image_filepath = input("Please enter the image filepath for scanning: ")
            if not os.path.exists(image_filepath):
                print(f"Image file not found: {image_filepath}")
                image_filepath = input("Please enter the image filepath for scanning: ")

        if not os.path.exists(image_filepath):
            print(f"Image file not found: {image_filepath}")
            return None

        # Grad-CAM visualization if enabled
        if self.gradcam:
            from gradcam import explain_image
            os.makedirs(self.gradcam_save_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_filepath))[0]
            save_cam_path = os.path.join(self.gradcam_save_dir, f"{base_name}_gradcam.png")
            result = explain_image(
                model=self.model,
                image_path=image_filepath,
                save_path=save_cam_path,
                device=self.device,
                transform=self.model.test_transform,
            )
            print(f"\n================ Co-Spy Inspection ================")
            print(f"Image File     : {image_filepath}")
            print(f"Prediction     : {result['verdict']} (Confidence: {result['score']*100:.2f}%)")
            print(f"Grad-CAM Saved : {save_cam_path}")
            print(f"===================================================\n")
            return result['score']

        image = Image.open(image_filepath).convert("RGB")
        image = self.model.test_transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        # Make the prediction
        prediction = self.model.predict(image)[0]

        if prediction > 0.5:
            print(f"Co-Spy Prediction: {prediction:.3f} - AI-Generated")
        else:
            print(f"Co-Spy Prediction: {prediction:.3f} - Real")

        return prediction
