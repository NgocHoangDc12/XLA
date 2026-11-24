import os
import io
import time
import base64
import threading
import webbrowser
import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory

from evaluate import Detector
from gradcam import GradCAM

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global detector instances (Tri-Expert Ensemble: SD-v1.4 + ProGAN + Writing)
print("Initializing CO-SPY Tri-Expert Detectors...")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
detector_sd = None
detector_progan = None
detector_writing = None

def get_detectors():
    global detector_sd, detector_progan, detector_writing
    if detector_sd is None:
        print(f"Loading CO-SPY SD-v1.4 Detector (Diffusion Expert) on {DEVICE}...")
        use_ckpt = "ckpt_improved" if os.path.exists("ckpt_improved/sd-v1_4") else "ckpt"
        is_pretrain = not os.path.exists(os.path.join(use_ckpt, "sd-v1_4", "fusion_attention")) and not os.path.exists(os.path.join(use_ckpt, "sd-v1_4", "fusion-attention"))
        try:
            detector_sd = Detector(
                device=DEVICE,
                mode="fusion",
                train_dataset="sd-v1_4",
                fusion_type="attention",
                ckpt=use_ckpt,
                pretrain=is_pretrain,
                use_freq=True,
                gradcam=True
            )
            print("SD-v1.4 Detector loaded successfully!")
        except Exception as e:
            print(f"Fallback loading pretrain for SD-v1.4: {e}")
            detector_sd = Detector(
                device=DEVICE,
                mode="fusion",
                train_dataset="sd-v1_4",
                fusion_type="attention",
                pretrain=True,
                use_freq=True,
                gradcam=True
            )

    if detector_progan is None:
        print(f"Loading CO-SPY ProGAN Detector (GAN Expert) on {DEVICE}...")
        use_ckpt = "ckpt_improved" if os.path.exists("ckpt_improved/progan") else "ckpt"
        is_pretrain = not os.path.exists(os.path.join(use_ckpt, "progan", "fusion_attention")) and not os.path.exists(os.path.join(use_ckpt, "progan", "fusion-attention"))
        try:
            detector_progan = Detector(
                device=DEVICE,
                mode="fusion",
                train_dataset="progan",
                fusion_type="attention",
                ckpt=use_ckpt,
                pretrain=is_pretrain,
                gradcam=True
            )
            print("ProGAN Detector loaded successfully!")
        except Exception as e:
            print(f"Fallback loading pretrain for ProGAN: {e}")
            detector_progan = Detector(
                device=DEVICE,
                mode="fusion",
                train_dataset="progan",
                fusion_type="attention",
                pretrain=True,
                gradcam=True
            )

    if detector_writing is None and os.path.exists("ckpt_improved/writing"):
        print(f"Loading CO-SPY Writing Detector (Writing Specialist) on {DEVICE}...")
        try:
            detector_writing = Detector(
                device=DEVICE,
                mode="fusion",
                train_dataset="writing",
                fusion_type="attention",
                ckpt="ckpt_improved",
                gradcam=True
            )
            print("Writing Detector loaded successfully!")
        except Exception as e:
            print(f"Warning loading Writing Detector: {e}")
            detector_writing = None

    return detector_sd, detector_progan, detector_writing

def image_to_base64(img_pil, format="JPEG", quality=90):
    buffered = io.BytesIO()
    if format.upper() == "JPEG" and img_pil.mode in ("RGBA", "P"):
        img_pil = img_pil.convert("RGB")
    img_pil.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def compute_fft_spectrum(pil_img):
    """Compute 2D Fast Fourier Transform Magnitude Spectrum in color."""
    gray = np.array(pil_img.convert('L'), dtype=np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1e-8)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored)

def compute_freq_score(pil_img):
    """Analyze high-frequency energy distribution."""
    gray = np.array(pil_img.convert('L'), dtype=np.float32)
    h, w = gray.shape
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 > r**2
    high_ratio = float(np.sum(mag * mask) / (np.sum(mag) + 1e-8))
    return min(100.0, round(high_ratio * 125.0, 2))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    samples_dir = os.path.join(app.static_folder, "samples")
    samples = []
    if os.path.exists(samples_dir):
        catalog = [
            {
                "id": "sample_ai_portrait.jpg",
                "name": "AI Portrait (Hyper-Realistic)",
                "type": "AI-Generated",
                "tag": "Portrait / Diffusion",
                "file": "samples/sample_ai_portrait.jpg"
            },
            {
                "id": "sample_real_photo.jpg",
                "name": "Real Photograph (DSLR)",
                "type": "Real Camera",
                "tag": "Natural Photo",
                "file": "samples/sample_real_photo.jpg"
            },
            {
                "id": "sample_ai_scene.png",
                "name": "AI Interior Scene",
                "type": "AI-Generated",
                "tag": "Synthetic Scene",
                "file": "samples/sample_ai_scene.png"
            }
        ]
        for item in catalog:
            if os.path.exists(os.path.join(app.static_folder, item["file"])):
                samples.append(item)
    return jsonify({"samples": samples})

@app.route("/api/scan", methods=["POST"])
def scan_image():
    t_start = time.time()
    pil_img = None

    # Check uploaded file
    if "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            pil_img = Image.open(file.stream).convert("RGB")
    
    # Check sample selection
    if pil_img is None and request.is_json:
        data = request.get_json()
        sample_path = data.get("sample_path")
        if sample_path:
            full_path = os.path.join(app.static_folder, sample_path)
            if os.path.exists(full_path):
                pil_img = Image.open(full_path).convert("RGB")
    
    # Check base64 input
    if pil_img is None and request.is_json:
        data = request.get_json()
        b64_data = data.get("image_base64")
        if b64_data:
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
            img_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if pil_img is None:
        return jsonify({"error": "No valid image provided"}), 400

    det_sd, det_progan, det_writing = get_detectors()

    # Preprocessing for all loaded detectors
    tensor_sd = det_sd.model.test_transform(pil_img).unsqueeze(0).to(det_sd.device)
    tensor_progan = det_progan.model.test_transform(pil_img).unsqueeze(0).to(det_progan.device)
    
    tensor_writing = None
    score_writing = None
    sem_writing = None
    art_writing = None
    if det_writing is not None:
        tensor_writing = det_writing.model.test_transform(pil_img).unsqueeze(0).to(det_writing.device)

    # 1. Predictions from all experts
    with torch.no_grad():
        score_sd = det_sd.model.predict(tensor_sd)[0]
        score_progan = det_progan.model.predict(tensor_progan)[0]

        # Branch scores for SD detector
        sem_sd = det_sd.model.sem(det_sd.model.sem_transform(tensor_sd)).sigmoid().item()
        art_sd = det_sd.model.art(det_sd.model.art_transform(tensor_sd)).sigmoid().item()

        # Branch scores for ProGAN detector
        sem_progan = det_progan.model.sem(det_progan.model.sem_transform(tensor_progan)).sigmoid().item()
        art_progan = det_progan.model.art(det_progan.model.art_transform(tensor_progan)).sigmoid().item()

        # Branch scores for Writing detector
        if det_writing is not None and tensor_writing is not None:
            score_writing = det_writing.model.predict(tensor_writing)[0]
            sem_writing = det_writing.model.sem(det_writing.model.sem_transform(tensor_writing)).sigmoid().item()
            art_writing = det_writing.model.art(det_writing.model.art_transform(tensor_writing)).sigmoid().item()

    # 2. Collaborative Multi-Evidence Forensic Fusion
    fusion_candidates = [s for s in [score_sd, score_progan, score_writing] if s is not None]
    sem_candidates = [s for s in [sem_sd, sem_progan, sem_writing] if s is not None]
    art_candidates = [a for a in [art_sd, art_progan, art_writing] if a is not None]

    max_sem = max(sem_candidates)
    max_art = max(art_candidates)
    max_fusion = max(fusion_candidates)

    # 2D FFT Spectrum
    fft_spec_pil = compute_fft_spectrum(pil_img)
    freq_score = compute_freq_score(pil_img)

    # Collaborative evidence: weighted combination of branches
    branch_cue = 0.55 * max_sem + 0.45 * max_art
    prob_ai = max(max_fusion, branch_cue)
    if (max_sem > 0.5 or max_art > 0.5) and freq_score > 65:
        spectral_boost = (freq_score - 65) / 100.0 * 0.25
        prob_ai = max(prob_ai, 0.60 + spectral_boost)
    elif max_sem > 0.48 and freq_score > 70:
        prob_ai = max(prob_ai, 0.55)

    is_ai = prob_ai >= 0.5
    verdict = "AI-Generated" if is_ai else "Real Image"
    confidence = prob_ai * 100 if is_ai else (1.0 - prob_ai) * 100

    # Pick the winning expert for Grad-CAM explanation
    conf_sd = max(score_sd, (sem_sd + art_sd) / 2.0, sem_sd * 0.85)
    conf_progan = max(score_progan, (sem_progan + art_progan) / 2.0, art_progan * 0.85)
    conf_writing = score_writing if score_writing is not None else 0.0
    if det_writing is not None and sem_writing is not None and art_writing is not None:
        if score_writing > 0.35 or (sem_writing > 0.6 and art_writing > 0.4):
            conf_writing = max(score_writing, (sem_writing + art_writing) / 2.0)

    expert_candidates = [
        (conf_sd, det_sd, tensor_sd, "Diffusion Specialist (SD-v1.4)"),
        (conf_progan, det_progan, tensor_progan, "GAN Specialist (ProGAN)"),
    ]
    if det_writing is not None and tensor_writing is not None:
        expert_candidates.append(
            (conf_writing, det_writing, tensor_writing, "Writing Specialist (Writing)")
        )
    expert_candidates.sort(key=lambda x: x[0], reverse=True)
    _, winning_det, winning_tensor, winning_expert = expert_candidates[0]

    # 3. Grad-CAM from winning expert
    cam_gen = GradCAM(winning_det.model)
    cam_mask, _ = cam_gen.generate_cam(winning_tensor)

    # 4. Create Heatmap Overlays
    w, h = pil_img.size
    overlay_img = GradCAM.overlay_heatmap(pil_img, cam_mask, alpha=0.5)

    # Pure heatmap for separate inspection
    cam_resized = cv2.resize(cam_mask, (w, h))
    cam_smooth = cv2.GaussianBlur(cam_resized, (15, 15), 0)
    heatmap_pure_cv = cv2.applyColorMap(np.uint8(255 * cam_smooth), cv2.COLORMAP_JET)
    heatmap_pure_cv = cv2.cvtColor(heatmap_pure_cv, cv2.COLOR_BGR2RGB)
    heatmap_pure_pil = Image.fromarray(heatmap_pure_cv)

    t_end = time.time()
    inference_time = round((t_end - t_start) * 1000, 1)

    if is_ai:
        if winning_expert.startswith("Writing"):
            origin_tech = "Synthetic Writing / AI Text-Image Generation (Writing Specialist)"
            detail_notes = "Phát hiện vết nét ký tự, bất thường biên cạnh chữ và cấu trúc đồ họa sinh bởi mô hình AI."
        elif max_sem > max_art or score_sd >= score_progan:
            origin_tech = "Latent Diffusion Models (Stable Diffusion, Midjourney, FLUX, SDXL)"
            detail_notes = "Phát hiện dấu vết tái tạo giải mã VAE và bất thường ngữ nghĩa/kết cấu đặc trưng của mô hình Diffusion."
        else:
            origin_tech = "Generative Adversarial Networks (StyleGAN, ProGAN, BigGAN)"
            detail_notes = "Phát hiện bất thường về kết cấu điểm ảnh và đỉnh phổ tần số nhân tạo đặc trưng của mạng GAN."
    else:
        origin_tech = "Ống kính máy ảnh quang học (DSLR / Smartphone Camera)"
        detail_notes = "Đặc trưng tự nhiên, hạt nhiễu cảm biến đồng đều, không phát hiện dấu vết upsampling hay giải mã AI."

    return jsonify({
        "success": True,
        "is_ai": is_ai,
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "overall_score": round(prob_ai, 4),
        "score_sd": round(score_sd * 100, 2),
        "score_progan": round(score_progan * 100, 2),
        "score_writing": round(score_writing * 100, 2) if score_writing is not None else None,
        "winning_expert": winning_expert,
        "sem_score": round(max_sem * 100, 2),
        "art_score": round(max_art * 100, 2),
        "freq_score": freq_score,
        "origin_tech": origin_tech,
        "detail_notes": detail_notes,
        "inference_time_ms": inference_time,
        "image_size": f"{w} x {h}",
        "device": det_sd.device,
        "original_image": image_to_base64(pil_img, format="JPEG", quality=85),
        "overlay_image": image_to_base64(overlay_img, format="JPEG", quality=85),
        "heatmap_image": image_to_base64(heatmap_pure_pil, format="JPEG", quality=85),
        "fft_image": image_to_base64(fft_spec_pil, format="JPEG", quality=85),
    })

def start_server(port=5000, auto_open=True):
    # Pre-warm models in background
    threading.Thread(target=get_detectors, daemon=True).start()
    if auto_open:
        def open_browser():
            time.sleep(1.5)
            print(f"\nOpening Web App at http://127.0.0.1:{port} ...\n")
            webbrowser.open(f"http://127.0.0.1:{port}")
        threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("CO-SPY Web Application")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()
    start_server(port=args.port, auto_open=not args.no_browser)
