import os
import torch
import torch.nn as nn
from torchvision import transforms
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
        concat_feat = torch.cat([feat_sem, feat_art], dim=1)
        gate = self.gate_net(concat_feat)
        fused = gate * feat_sem + (1.0 - gate) * feat_art
        return fused, gate


class CrossAttentionFusionModule(nn.Module):
    """Bidirectional Cross-Attention + Gated Fusion Module."""

    def __init__(self, sem_dim=768, art_dim=512, embed_dim=256, num_heads=4, num_classes=1, dropout=0.1):
        super(CrossAttentionFusionModule, self).__init__()
        self.embed_dim = embed_dim

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

        self.cross_sem_to_art = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_art_to_sem = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        self.gated_fusion = GatedFusion(embed_dim=embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, sem_feat, art_feat, pred_sem, pred_art):
        h_sem = self.proj_sem(sem_feat).unsqueeze(1)
        h_art = self.proj_art(art_feat).unsqueeze(1)

        ctx_sem = self.cross_sem_to_art(h_sem, h_art, h_art).squeeze(1)
        ctx_art = self.cross_art_to_sem(h_art, h_sem, h_sem).squeeze(1)

        fused_feat, gate = self.gated_fusion(ctx_sem, ctx_art)

        all_features = torch.cat([fused_feat, pred_sem, pred_art], dim=1)
        logits = self.classifier(all_features)

        return logits, fused_feat, gate


class CoSpyAttentionFusionDetector(BaseDetector):
    """CO-SPY Detector with Cross-Attention Gated Fusion for ProGAN dataset."""

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

        self.sem = SemanticDetector()
        self.sem.load_weights(semantic_weights_path)

        self.art = ArtifactDetector(use_freq=use_freq)
        self.art.load_weights(artifact_weights_path)

        for param in self.sem.parameters():
            param.requires_grad = False
        for param in self.art.parameters():
            param.requires_grad = False

        sem_dim = self.sem.fc.in_features
        art_dim = self.art.fc.in_features

        self.fusion = CrossAttentionFusionModule(
            sem_dim=sem_dim,
            art_dim=art_dim,
            embed_dim=256,
            num_heads=4,
            num_classes=num_classes,
            dropout=0.1,
        )

        self.sem_transform = transforms.Compose([
            transforms.Normalize(self.sem.mean, self.sem.std)
        ])
        self.art_transform = transforms.Compose([
            transforms.Normalize(self.art.mean, self.art.std)
        ])

        self._build_fusion_transforms()

    def _build_fusion_transforms(self):
        from utils import data_augment

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
        torch.save(self.fusion.state_dict(), weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location='cpu')
        self.fusion.load_state_dict(weights, strict=True)
