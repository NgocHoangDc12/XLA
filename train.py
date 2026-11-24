import os
import time
import torch
import importlib
import numpy as np
from PIL import Image
from loguru import logger

from utils import evaluate
from dataSets import TrainDataset, FusionTrainDataset

import warnings
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self,
                 mode: str,
                 device: str,
                 branch: str,
                 train_dataset: str,
                 ckpt: str = "ckpt",
                 epochs: int = 10,
                 batch_size: int = 32,
                 feat_interp: bool = False,
                 feat_interp_alpha: float = 0.2,
                 feat_interp_ratio: float = 0.5,
                 fusion_type: str = "linear",
                 use_freq: bool = False,
                 loss_type: str = "bce",
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25):

        self.device = device
        self.mode = mode
        self.branch = branch
        self.train_dataset = train_dataset
        self.ckpt = ckpt
        self.epochs = epochs
        self.batch_size = batch_size
        self.feat_interp = feat_interp
        self.feat_interp_alpha = feat_interp_alpha
        self.feat_interp_ratio = feat_interp_ratio
        self.fusion_type = fusion_type
        self.use_freq = use_freq
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Dynamically import the detector module from either "progan" or "sd-v1_4"
        detector_arch = "progan" if train_dataset == "writing" else train_dataset
        detector_module = importlib.import_module(f"detectors.{detector_arch}")

        # Get the detector based on mode
        if self.mode == "branch":
            ArtifactDetector = getattr(detector_module, "ArtifactDetector")
            SemanticDetector = getattr(detector_module, "SemanticDetector")
            if self.branch == "artifact":
                self.model = ArtifactDetector(use_freq=self.use_freq)
            elif self.branch == "semantic":
                self.model = SemanticDetector()
            else:
                raise ValueError(f"Unknown detector: {self.branch}")
        elif self.mode == "fusion":
            semantic_weights_path = os.path.join(self.ckpt, self.train_dataset, "semantic", "best_model.pth")
            artifact_weights_path = os.path.join(self.ckpt, self.train_dataset, "artifact", "best_model.pth")
            if not os.path.exists(semantic_weights_path) or not os.path.exists(artifact_weights_path):
                raise ValueError("Semantic or Artifact weights path does not exist for fusion mode")

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
                self.weight_init = 0.5 if self.train_dataset == "sd-v1_4" else 0.0
                self.bias_floor = 2.0 if self.train_dataset == "sd-v1_4" else 0.0
        elif self.mode == "end2end":
            End2EndDetector = getattr(detector_module, "End2EndDetector")
            self.model = End2EndDetector()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.model.to(self.device)

        # Initialize the fc layer (if model has fc attribute)
        if hasattr(self.model, "fc") and isinstance(self.model.fc, torch.nn.Linear):
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)
        if self.mode == "end2end":
            torch.nn.init.normal_(self.model.sem.fc.weight.data, 0.0, 0.02)
            torch.nn.init.normal_(self.model.art.fc.weight.data, 0.0, 0.02)

        if self.mode == "fusion" and self.fusion_type == "linear":
            with torch.no_grad():
                self.model.fc.weight.data += self.weight_init
                self.model.fc.bias.data.fill_(self.bias_floor)

        # Optimizer
        _beta1 = 0.9
        _weight_decay = 0.0
        params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {len(params)}")

        if self.mode == "fusion":
            self._lr = 1e-4 if self.fusion_type == "attention" else 1e-1
        else:
            self._lr = 1e-4
        self._lr_min = self._lr / self.epochs
        self.optimizer = torch.optim.AdamW(params, lr=self._lr, betas=(_beta1, 0.999), weight_decay=_weight_decay)

        # Loss function
        if self.loss_type == "focal":
            from utils import FocalBCEWithLogitsLoss
            self.criterion = FocalBCEWithLogitsLoss(gamma=self.focal_gamma, alpha=self.focal_alpha)
            logger.info(f"Using Focal Loss (gamma={self.focal_gamma}, alpha={self.focal_alpha})")
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()
            logger.info("Using BCEWithLogitsLoss")

        self.lr_scheduler = None

        # Set lr_step_per_batch based on the training dataset
        if self.train_dataset == "sd-v1_4":
            self.lr_step_per_batch = True
        elif self.train_dataset in ["progan", "writing"]:
            self.lr_step_per_batch = False
        else:
            raise ValueError(f"Unknown train dataset: {self.train_dataset}")

    def _save_training_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        next_index: int,
        permutation,
        batches_completed: int,
        best_acc: float,
    ):
        checkpoint = {
            "epoch": epoch,
            "next_index": next_index,
            "permutation": permutation,
            "batches_completed": batches_completed,
            "best_acc": float(best_acc),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            ),
        }
        temporary_path = f"{checkpoint_path}.tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

    def _load_training_checkpoint(self, checkpoint_path: str):
        numpy_multiarray = (
            np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
        )
        with torch.serialization.safe_globals([
            numpy_multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float64)),
            np.float64,
        ]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.lr_scheduler is not None and checkpoint["scheduler"] is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

        return (
            checkpoint["epoch"],
            checkpoint["next_index"],
            checkpoint["permutation"],
            checkpoint["batches_completed"],
            checkpoint["best_acc"],
        )

    def _mix_features(self, feats, labels):
        """Feature-space Interpolation on a fraction of the batch.

        For self.feat_interp_ratio of the batch, interpolate:
            feat_mixed  = delta * feat_a + (1 - delta) * feat_b
            label_mixed = delta * label_a + (1 - delta) * label_b
        with delta ~ Beta(alpha, alpha) sampled independently per mixed sample.
        """
        B = feats.size(0)
        device = feats.device
        labels_f = labels.float().clone()
        mix_mask = torch.rand(B, device=device) < self.feat_interp_ratio
        n_mix = int(mix_mask.sum().item())
        if n_mix == 0:
            return feats, labels_f
        mix_idx = torch.where(mix_mask)[0]
        partner = torch.randint(0, B, (n_mix,), device=device)
        beta = torch.distributions.Beta(self.feat_interp_alpha, self.feat_interp_alpha)
        delta = beta.sample((n_mix,)).to(device)
        feats_new = feats.clone()
        d2 = delta.unsqueeze(1)
        feats_new[mix_idx] = d2 * feats[mix_idx] + (1.0 - d2) * feats[partner]
        labels_new = labels_f.clone()
        labels_new[mix_idx] = delta * labels_f[mix_idx] + (1.0 - delta) * labels_f[partner]
        return feats_new, labels_new

    def train_step(self, batch_data):
        self.optimizer.zero_grad()

        if self.mode == "fusion":
            sem_t, art_t, labels = batch_data
            sem_t = sem_t.to(self.device, non_blocking=True)
            art_t = art_t.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if hasattr(self.model, "forward_from_branches"):
                outputs = self.model.forward_from_branches(sem_t, art_t)
                loss = self.criterion(outputs, labels.unsqueeze(1).float())
            else:
                pred_sem = self.model.sem(sem_t)
                pred_art = self.model.art(art_t)
                feat = torch.cat([pred_sem, pred_art], dim=1)
                outputs = self.model.fc(feat)
                loss = self.criterion(outputs, labels.unsqueeze(1).float())
                b = self.model.fc.bias[0]
                loss = loss + torch.nn.functional.relu(self.bias_floor - b)
        else:
            inputs, labels = batch_data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Feature-space interpolation
            use_feat_interp = (
                self.feat_interp
                and self.mode == "branch"
                and self.branch == "semantic"
            )
            if use_feat_interp:
                with torch.no_grad():
                    # Fallback: sd-v1_4 open_clip uses .encode_image, progan HF CLIPModel uses .get_image_features
                    encode_fn = getattr(self.model.clip, "encode_image", None) or self.model.clip.get_image_features
                    feats = encode_fn(inputs)
                feats_mixed, labels_soft = self._mix_features(feats, labels)
                outputs = self.model.fc(feats_mixed)
                loss = self.criterion(outputs, labels_soft.unsqueeze(1))
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1).float())

        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None and self.lr_step_per_batch:
            self.lr_scheduler.step()

        eval_loss = loss.item()
        y_pred = outputs.sigmoid().flatten().tolist()
        y_true = labels.tolist()
        return eval_loss, y_pred, y_true

    def _build_scheduler(self, steps_per_epoch: int):
        """Construct the LR scheduler appropriate for this training dataset.

        - sd-v1_4: per-step CosineAnnealingLR from self._lr to self._lr_min over (epochs * steps_per_epoch) optimizer steps.

        - progan: _step_lr_per_epoch() adjusts lr *= 0.9 when epoch % 10 == 0 and != 0, applied after each epoch's validation.
        """
        if self.lr_step_per_batch:
            total_steps = self.epochs * steps_per_epoch
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self._lr_min,
            )
            logger.info(
                f"LR cosine {self._lr:.1e} -> {self._lr_min:.1e} over {total_steps} steps"
            )
        else:
            logger.info(
                f"LR step-decay start {self._lr:.1e}, x0.9 every 10 epochs"
            )

    def _step_lr_per_epoch(self, epoch: int):
        """Original Co-Spy rule: lr *= 0.9 when epoch is a non-zero multiple
        of 10. `epoch` is the index just finished (0-based)."""
        if epoch % 10 == 0 and epoch != 0:
            for pg in self.optimizer.param_groups:
                pg["lr"] *= 0.9

    def train(self):
        # Determine save directory
        if self.mode == "branch":
            subdir = self.branch
        elif self.mode == "fusion":
            subdir = f"fusion_{self.fusion_type}" if self.fusion_type != "linear" else "fusion"
        else:
            subdir = self.mode
        # Set the saving directory
        model_dir = os.path.join(self.ckpt, self.train_dataset, subdir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Setup logger
        log_path = f"{model_dir}/training.log"
        logger_id = logger.add(
            log_path,
            format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
            level="DEBUG",
        )

        # Load the training and validation dataset
        if self.mode == "fusion":
            # Match train_fusion_v4: each branch applies its own transform to
            # the raw PIL image (so sem / art see their native aug + resize + normalize).
            base_train = TrainDataset(train_dataset=self.train_dataset, split="train", transform=None)
            base_val = TrainDataset(train_dataset=self.train_dataset, split="val", transform=None)
            train_dataset = FusionTrainDataset(base_train, self.model.sem.train_transform, self.model.art.train_transform)
            val_dataset = FusionTrainDataset(base_val, self.model.sem.test_transform, self.model.art.test_transform)
        else:
            train_dataset = TrainDataset(train_dataset=self.train_dataset,
                                         split="train",
                                         transform=self.model.train_transform)
            val_dataset = TrainDataset(train_dataset=self.train_dataset,
                                       split="val",
                                       transform=self.model.val_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=4,
                                                 pin_memory=True)

        logger.info(f"Train size {len(train_dataset)} | Val size {len(val_dataset)}")
        if self.feat_interp and self.mode == "branch" and self.branch == "semantic":
            logger.info(
                f"feat_interp ON | alpha={self.feat_interp_alpha} "
                f"ratio={self.feat_interp_ratio}"
            )

        # Build LR scheduler now that we know len(train_loader).
        self._build_scheduler(steps_per_epoch=len(train_loader))

        checkpoint_path = os.path.join(model_dir, "last_checkpoint.pth")
        start_epoch = 0
        next_index = 0
        permutation = None
        batches_completed = 0
        best_acc = 0

        if os.path.exists(checkpoint_path):
            (
                start_epoch,
                next_index,
                permutation,
                batches_completed,
                best_acc,
            ) = self._load_training_checkpoint(checkpoint_path)
            logger.info(
                f"Resuming from epoch {start_epoch + 1}, "
                f"image {next_index}/{len(train_dataset)}"
            )
            if next_index < 0 or next_index > len(train_dataset):
                raise ValueError("Checkpoint position is outside the current training dataset")
            if permutation is not None and len(permutation) != len(train_dataset):
                raise ValueError(
                    "Checkpoint dataset size differs from the current training dataset"
                )
            if permutation is None and next_index != 0:
                raise ValueError("Checkpoint is missing the training sample order")

        # Train in shuffled chunks so no more than 10,000 indices are queued at once.
        chunk_size = 10_000
        checkpoint_every_batches = 100
        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            time_start = time.time()
            if permutation is None:
                permutation = torch.randperm(len(train_dataset))
                next_index = 0
                batches_completed = 0
                self._save_training_checkpoint(
                    checkpoint_path,
                    epoch,
                    next_index,
                    permutation,
                    batches_completed,
                    best_acc,
                )

            while next_index < len(train_dataset):
                chunk_end = min(next_index + chunk_size, len(train_dataset))
                chunk_indices = permutation[next_index:chunk_end].tolist()
                chunk_loader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(train_dataset, chunk_indices),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )

                for batch_data in chunk_loader:
                    eval_loss, y_pred, y_true = self.train_step(batch_data)
                    ap, accuracy = evaluate(y_pred, y_true)
                    batch_size = len(y_true)
                    next_index += batch_size
                    batches_completed += 1

                    if batches_completed % checkpoint_every_batches == 0:
                        time_end = time.time()
                        logger.info(
                            f"Epoch {epoch} | Batch {batches_completed}/{len(train_loader)} "
                            f"| Loss {eval_loss:.4f} | AP {ap*100:.2f}% "
                            f"| Accuracy {accuracy*100:.2f}% "
                            f"| Time {time_end-time_start:.2f}s"
                        )
                        time_start = time.time()
                        self._save_training_checkpoint(
                            checkpoint_path,
                            epoch,
                            next_index,
                            permutation,
                            batches_completed,
                            best_acc,
                        )

                self._save_training_checkpoint(
                    checkpoint_path,
                    epoch,
                    next_index,
                    permutation,
                    batches_completed,
                    best_acc,
                )
                logger.info(
                    f"Epoch {epoch} | Trained {next_index}/{len(train_dataset)} images"
                )

            # Evaluate the model
            self.model.eval()
            y_pred, y_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if self.mode == "fusion":
                        sem_t, art_t, labels = batch
                        sem_t = sem_t.to(self.device, non_blocking=True)
                        art_t = art_t.to(self.device, non_blocking=True)
                        if hasattr(self.model, "forward_from_branches"):
                            logits = self.model.forward_from_branches(sem_t, art_t)
                        else:
                            feat = torch.cat([self.model.sem(sem_t), self.model.art(art_t)], dim=1)
                            logits = self.model.fc(feat)
                        y_pred.extend(logits.sigmoid().flatten().cpu().tolist())
                        y_true.extend(labels.tolist())
                    else:
                        images, labels = batch
                        y_pred.extend(self.model.predict(images))
                        y_true.extend(labels.tolist())

            ap, accuracy = evaluate(y_pred, y_true)
            eval_type = "Test" if self.mode == "branch" else "Total"
            cur_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch} | {eval_type} AP {ap*100:.2f}% | {eval_type} Accuracy {accuracy*100:.2f}% | LR {cur_lr:.2e}")

            if not self.lr_step_per_batch:
                self._step_lr_per_epoch(epoch)

            # Save the model
            if accuracy >= best_acc:
                best_acc = accuracy
                self.model.save_weights(f"{model_dir}/best_model.pth")
                logger.info(f"Best model saved with accuracy {best_acc*100:.2f}%")

            if epoch % 5 == 0:
                self.model.save_weights(f"{model_dir}/epoch_{epoch}.pth")
                logger.info(f"Model saved at epoch {epoch}")

            permutation = None
            next_index = 0
            batches_completed = 0
            self._save_training_checkpoint(
                checkpoint_path,
                epoch + 1,
                next_index,
                permutation,
                batches_completed,
                best_acc,
            )

        # Save the final model
        self.model.save_weights(f"{model_dir}/final_model.pth")
        logger.info("Final model saved")

        # Remove the logger
        logger.remove(logger_id)
