import os
import cv2
import torch
import numpy as np
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    for visualizing regions influencing AI-generated image detection.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer = target_layer or self._find_target_layer()
        self.hook_handles = []

        if self.target_layer is not None:
            self._register_hooks()

    def _find_target_layer(self):
        """Automatically find the best convolutional layer for visualization."""
        # 1. Check for artifact_encoder layer2
        if hasattr(self.model, "art"):
            if hasattr(self.model.art, "artifact_encoder"):
                encoder = self.model.art.artifact_encoder
                if hasattr(encoder, "layer2"):
                    return encoder.layer2
        if hasattr(self.model, "artifact_encoder"):
            if hasattr(self.model.artifact_encoder, "layer2"):
                return self.model.artifact_encoder.layer2

        # 2. Fallback to last Conv2d layer
        target = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target = module
        return target

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles = [
            self.target_layer.register_forward_hook(forward_hook),
            self.target_layer.register_full_backward_hook(backward_hook),
        ]

    def generate_cam(self, input_tensor):
        """
        Generate CAM heatmap for input tensor [1, C, H, W].
        Returns 2D numpy float array in [0, 1].
        """
        # Save original requires_grad states and enable grad for CAM
        orig_states = {}
        for name, param in self.model.named_parameters():
            orig_states[name] = param.requires_grad
            param.requires_grad = True

        try:
            self.activations = None
            self.gradients = None
            self.model.zero_grad()

            # Forward pass
            output = self.model(input_tensor)
            score = output.sigmoid().flatten()[0]

            # Backward with respect to the output score
            output.backward(torch.ones_like(output))

            if self.gradients is None or self.activations is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

            # Global average pooling of gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]
            cam = torch.relu(cam)

            cam_np = cam.squeeze().detach().cpu().numpy()

            # If relu eliminated all features, use absolute gradient weighting
            if cam_np.max() <= 1e-8:
                cam_alt = torch.sum(torch.abs(weights) * torch.relu(self.activations), dim=1, keepdim=True)
                cam_np = cam_alt.squeeze().detach().cpu().numpy()

            # Normalize to [0, 1]
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max - cam_min > 1e-8:
                cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_np = np.zeros_like(cam_np)

            return cam_np, score.item()

        finally:
            # Restore original requires_grad states
            for name, param in self.model.named_parameters():
                param.requires_grad = orig_states.get(name, False)
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles.clear()

    @staticmethod
    def overlay_heatmap(img_pil, cam_mask, alpha=0.5):
        """
        Overlay vibrant CAM heatmap onto PIL image.
        Returns a PIL Image with heatmap overlaid.
        """
        img_np = np.array(img_pil.convert("RGB"))
        h, w, _ = img_np.shape

        cam_resized = cv2.resize(cam_mask, (w, h))
        # Gaussian smoothing for natural heatmap transitions
        cam_smooth = cv2.GaussianBlur(cam_resized, (15, 15), 0)
        
        # Color mapping (JET: Blue -> Cyan -> Yellow -> Orange -> Red)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_smooth), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Intensity-adaptive alpha blending: high anomaly regions pop out brightly
        intensity = cam_smooth[:, :, np.newaxis]
        alpha_map = np.clip(intensity * 0.65 + 0.2, 0.2, 0.75)
        blended = np.uint8(alpha_map * heatmap + (1.0 - alpha_map) * img_np)
        return Image.fromarray(blended)


def explain_image(model, image_path, save_path=None, device="cpu", transform=None):
    """
    Generate an explainability report (Original | Heatmap) for an image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    raw_img = Image.open(image_path).convert("RGB")

    if transform is not None:
        tensor = transform(raw_img).unsqueeze(0).to(device)
    else:
        from torchvision import transforms
        default_transform = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
        ])
        tensor = default_transform(raw_img).unsqueeze(0).to(device)

    gradcam = GradCAM(model)
    cam_mask, score = gradcam.generate_cam(tensor)

    overlay = GradCAM.overlay_heatmap(raw_img, cam_mask, alpha=0.5)

    # Side-by-side composite
    w, h = raw_img.size
    banner_h = 45
    composite = Image.new("RGB", (w * 2, h + banner_h), color=(25, 25, 30))
    composite.paste(raw_img, (0, banner_h))
    composite.paste(overlay, (w, banner_h))

    # Add descriptive banner
    comp_np = np.array(composite)
    verdict = "AI-Generated" if score > 0.5 else "Real"
    verdict_color = (60, 220, 100) if verdict == "Real" else (60, 80, 240) # BGR
    
    cv2.putText(comp_np, f"Original Image", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(comp_np, f"Grad-CAM Heatmap: {verdict} ({score * 100:.2f}%)", (w + 20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, verdict_color, 2, cv2.LINE_AA)
    composite = Image.fromarray(comp_np)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        composite.save(save_path)

    return {
        "score": score,
        "verdict": verdict,
        "label": f"{verdict} ({score * 100:.2f}%)",
        "composite_image": composite,
        "overlay_image": overlay,
        "cam_mask": cam_mask,
    }
