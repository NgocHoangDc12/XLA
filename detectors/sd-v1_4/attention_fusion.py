import os
import torch
import torch.nn as nn
from torchvision import transforms
from utils import data_augment
from .base import BaseDetector
from .semantic import SemanticDetector
from .artifact import ArtifactDetector


class CrossAttentionBlock(nn.Module):
    """Multi-Head Cross-Attention block with residual connection and LayerNorm."""

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # q: [B, 1, D], k: [B, 1, D], v: [B, 1, D]
        attn_out, _ = self.mha(q, k, v)
        out = self.norm(q + self.dropout(attn_out))
        return out


class GatedFusion(nn.Module):
    """Adaptive Gated Fusion between semantic and artifact representations."""

    def __init__(self, embed_dim=256):
        super(GatedFusion, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, feat_sem, feat_art):
        # feat_sem, feat_art: [B, D]
        concat_feat = torch.cat([feat_sem, feat_art], dim=1)
        gate = self.gate_net(concat_feat)  # [B, D] in (0, 1)
        fused = gate * feat_sem + (1.0 - gate) * feat_art
        return fused, gate


class CrossAttentionFusionModule(nn.Module):
    """
    Bidirectional Cross-Attention + Gated Fusion Module.
    1. Projects Semantic (1152-d) and Artifact (512-d or 640-d) features into embed_dim.
    2. Computes bidirectional cross-attention:
       - Semantic attends to Artifact (Q=Sem, K=Art, V=Art)
       - Artifact attends to Semantic (Q=Art, K=Sem, V=Sem)
    3. Fuses the contextualized features through an adaptive gating unit.
    4. Combines fused deep representation with branch logits for final classification.
    """

    def __init__(self, sem_dim=1152, art_dim=512, embed_dim=256, num_heads=4, num_classes=1, dropout=0.1):
        super(CrossAttentionFusionModule, self).__init__()
        self.embed_dim = embed_dim

        # Feature projections
        self.proj_sem = nn.Sequential(
            nn.Linear(sem_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.proj_art = nn.Sequential(
            nn.Linear(art_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Bidirectional cross-attention blocks
        self.cross_sem_to_art = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_art_to_sem = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Gated fusion
        self.gated_fusion = GatedFusion(embed_dim=embed_dim)

        # Classifier: combines fused deep representation (embed_dim) + branch logits (2)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, sem_feat, art_feat, pred_sem, pred_art):
        # 1. Project features
        h_sem = self.proj_sem(sem_feat).unsqueeze(1)  # [B, 1, D]
        h_art = self.proj_art(art_feat).unsqueeze(1)  # [B, 1, D]

        # 2. Bidirectional Cross-Attention
        ctx_sem = self.cross_sem_to_art(h_sem, h_art, h_art).squeeze(1)  # [B, D]
        ctx_art = self.cross_art_to_sem(h_art, h_sem, h_sem).squeeze(1)  # [B, D]

        # 3. Gated Fusion
        fused_feat, gate = self.gated_fusion(ctx_sem, ctx_art)  # [B, D]

        # 4. Concatenate with branch logits as calibrated priors
        all_features = torch.cat([fused_feat, pred_sem, pred_art], dim=1)  # [B, D + 2]
        logits = self.classifier(all_features)

        return logits, fused_feat, gate


class CoSpyAttentionFusionDetector(BaseDetector):
    """CO-SPY Detector with Cross-Attention Gated Fusion."""

    def __init__(self, semantic_weights_path, artifact_weights_path, num_classes=1, use_freq=False):
        super(CoSpyAttentionFusionDetector, self).__init__()

        # Auto-detect if artifact weights were trained with use_freq
        if not use_freq and os.path.exists(artifact_weights_path):
            try:
                chk = torch.load(artifact_weights_path, map_location='cpu')
                if 'fc.weight' in chk and chk['fc.weight'].shape[1] == 640:
                    use_freq = True
            except Exception:
                pass

        # Load semantic detector
        self.sem = SemanticDetector()
        self.sem.load_weights(semantic_weights_path)

        # Load artifact detector
        self.art = ArtifactDetector(use_freq=use_freq)
        self.art.load_weights(artifact_weights_path)

        # Freeze the two pre-trained branches
        for param in self.sem.parameters():
            param.requires_grad = False
        for param in self.art.parameters():
            param.requires_grad = False

        # Dimension setup
        sem_dim = self.sem.fc.in_features
        art_dim = self.art.fc.in_features

        # Cross-Attention Fusion Module
        self.fusion = CrossAttentionFusionModule(
            sem_dim=sem_dim,
            art_dim=art_dim,
            embed_dim=256,
            num_heads=4,
            num_classes=num_classes,
            dropout=0.1,
        )

        # Transforms inside the forward function
        self.sem_transform = transforms.Compose([
            transforms.Normalize(self.sem.mean, self.sem.std)
        ])
        self.art_transform = transforms.Compose([
            transforms.Resize(self.art.cropSize, antialias=False),
            transforms.Normalize(self.art.mean, self.art.std)
        ])

        # Build dataset transforms
        self._build_transforms()

    def _build_transforms(self):
        """Build transforms without normalization (normalization done per-branch in forward)."""
        self.loadSize = 384
        self.cropSize = 384

        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        random_crop = transforms.RandomCrop(self.cropSize)
        center_crop = transforms.CenterCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            flip_func,
            aug_func,
            rz_func,
            random_crop,
            transforms.ToTensor(),
        ])

        self.val_transform = transforms.Compose([
            rz_func,
            random_crop,
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            center_crop,
            transforms.ToTensor(),
        ])

    def forward_from_branches(self, sem_t, art_t):
        """Compute fusion prediction directly from branch-specific transformed tensors."""
        with torch.no_grad():
            sem_feat, pred_sem = self.sem(sem_t, return_feat=True)
            art_feat, pred_art = self.art(art_t, return_feat=True)
        logits, _, _ = self.fusion(sem_feat, art_feat, pred_sem, pred_art)
        return logits

    def forward(self, x, return_feat=False):
        x_sem = self.sem_transform(x)
        x_art = self.art_transform(x)
        sem_feat, pred_sem = self.sem(x_sem, return_feat=True)
        art_feat, pred_art = self.art(x_art, return_feat=True)
        logits, fused_feat, gate = self.fusion(sem_feat, art_feat, pred_sem, pred_art)
        if return_feat:
            return fused_feat, logits
        return logits

    def save_weights(self, weights_path):
        # Save fusion module state dict
        torch.save(self.fusion.state_dict(), weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location='cpu')
        self.fusion.load_state_dict(weights, strict=True)
