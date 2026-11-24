import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from torchvision import transforms
from utils import data_augment
from .base import BaseDetector


class ArtifactDetector(BaseDetector):
    def __init__(self, dim_artifact=512, num_classes=1, use_freq=False):
        super(ArtifactDetector, self).__init__()
        self.use_freq = use_freq
        self.dim_artifact = dim_artifact + (128 if use_freq else 0)

        # Load the pre-trained VAE
        model_id = "CompVis/stable-diffusion-v1-4"
        vae = StableDiffusionPipeline.from_pretrained(model_id).vae
        vae.requires_grad_(False)
        self.artifact_encoder = VAEReconEncoder(vae, use_freq=use_freq)

        # Classifier
        self.fc = torch.nn.Linear(self.dim_artifact, num_classes)

        # Build transforms
        self._build_transforms()

    
    def _build_transforms(self):
        # Normalization
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]

        # Resolution
        self.loadSize = 256
        self.cropSize = 224

        # Data augmentation
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        random_crop = transforms.RandomCrop(self.cropSize)
        center_crop = transforms.CenterCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            aug_func,
            rz_func,
            random_crop,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.val_transform = transforms.Compose([
            rz_func,
            random_crop,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            center_crop,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def forward(self, x, return_feat=False):
        feat = self.artifact_encoder(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location='cpu')
        if 'fc.weight' in weights and weights['fc.weight'].shape[1] == 640 and not self.use_freq:
            self.use_freq = True
            self.dim_artifact = 640
            self.fc = torch.nn.Linear(640, self.fc.out_features)
            self.artifact_encoder.use_freq = True
            self.artifact_encoder.freq_branch = FrequencyBranch(in_channels=3, out_dim=128)
        self.load_state_dict(weights, strict=False)


# Helper functions for ResNet
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FrequencyBranch(nn.Module):
    """Extracts frequency-domain representations from artifact maps using 2D FFT."""

    def __init__(self, in_channels=3, out_dim=128):
        super(FrequencyBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # x is artifact map: [B, C, H, W]
        # Compute 2D FFT per channel
        fft_val = torch.fft.fft2(x, norm="ortho")
        fft_shift = torch.fft.fftshift(fft_val, dim=(-2, -1))
        # Log magnitude spectrum
        mag = torch.log(torch.abs(fft_shift) + 1e-8)
        # Normalize across spatial dims for numerical stability
        mean = mag.mean(dim=(-2, -1), keepdim=True)
        std = mag.std(dim=(-2, -1), keepdim=True) + 1e-6
        mag_norm = (mag - mean) / std
        feat = self.conv(mag_norm)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        return feat


class VAEReconEncoder(nn.Module):
    def __init__(self, vae, block=Bottleneck, use_freq=False):
        super(VAEReconEncoder, self).__init__()
        self.use_freq = use_freq

        # Define the ResNet model
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 is [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Frequency branch
        if self.use_freq:
            self.freq_branch = FrequencyBranch(in_channels=3, out_dim=128)

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load the VAE model
        self.vae = vae

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reconstruct(self, x):
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.mean
            decoded = self.vae.decode(latent).sample
        return decoded

    def forward(self, x):
        # Reconstruct
        x_recon = self.reconstruct(x)
        # Compute the artifacts
        artifact_raw = x - x_recon

        freq_feat = None
        if self.use_freq:
            freq_feat = self.freq_branch(artifact_raw)

        # Scale the artifacts
        x = artifact_raw / 7. * 100.

        # Forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        spatial_feat = x.view(x.size(0), -1)

        if self.use_freq:
            return torch.cat([spatial_feat, freq_feat], dim=1)
        return spatial_feat

