document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide icons
    lucide.createIcons();

    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const btnBrowse = document.getElementById("btnBrowse");
    const dropzoneEmpty = document.getElementById("dropzoneEmpty");
    const dropzonePreview = document.getElementById("dropzonePreview");
    const previewImage = document.getElementById("previewImage");
    const btnClear = document.getElementById("btnClear");
    const btnScan = document.getElementById("btnScan");
    const samplesGrid = document.getElementById("samplesGrid");

    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const resultsContent = document.getElementById("resultsContent");

    // Verdict elements
    const verdictBanner = document.getElementById("verdictBanner");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictTitle = document.getElementById("verdictTitle");
    const verdictCaption = document.getElementById("verdictCaption");
    const gaugeFill = document.getElementById("gaugeFill");
    const gaugeNumber = document.getElementById("gaugeNumber");

    // Tri-Expert elements
    const scoreProganEl = document.getElementById("scoreProgan");
    const scoreSDEl = document.getElementById("scoreSD");
    const scoreWritingEl = document.getElementById("scoreWriting");

    // Branch elements
    const semVal = document.getElementById("semVal");
    const semBar = document.getElementById("semBar");
    const artVal = document.getElementById("artVal");
    const artBar = document.getElementById("artBar");
    const freqVal = document.getElementById("freqVal");
    const freqBar = document.getElementById("freqBar");

    // Assessment elements
    const originTech = document.getElementById("originTech");
    const detailNotes = document.getElementById("detailNotes");

    // Viewer elements
    const sliderView = document.getElementById("sliderView");
    const sideView = document.getElementById("sideView");
    const heatmapView = document.getElementById("heatmapView");
    const fftView = document.getElementById("fftView");
    const imgBefore = document.getElementById("imgBefore");
    const imgAfter = document.getElementById("imgAfter");
    const imageAfterContainer = document.getElementById("imageAfter");
    const sliderDivider = document.getElementById("sliderDivider");
    const sideImgOrig = document.getElementById("sideImgOrig");
    const sideImgOverlay = document.getElementById("sideImgOverlay");
    const pureHeatmapImg = document.getElementById("pureHeatmapImg");
    const fftImg = document.getElementById("fftImg");
    const opacityRange = document.getElementById("opacityRange");
    const opacityVal = document.getElementById("opacityVal");

    // Meta elements
    const metaTime = document.getElementById("metaTime");
    const metaSize = document.getElementById("metaSize");
    const metaDevice = document.getElementById("metaDevice");
    const btnDownload = document.getElementById("btnDownload");

    // View buttons
    const btnViewSlider = document.getElementById("btnViewSlider");
    const btnViewSide = document.getElementById("btnViewSide");
    const btnViewHeatmap = document.getElementById("btnViewHeatmap");
    const btnViewFFT = document.getElementById("btnViewFFT");
    const toggleBtns = [btnViewSlider, btnViewSide, btnViewHeatmap, btnViewFFT].filter(Boolean);

    // State
    let currentFile = null;
    let currentSamplePath = null;
    let isDraggingSlider = false;
    let latestForensicData = null;

    // Load samples
    loadSamples();

    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                samplesGrid.innerHTML = data.samples.map(sample => `
                    <div class="sample-card" data-file="${sample.file}">
                        <img src="/static/${sample.file}" class="sample-thumb" alt="${sample.name}">
                        <div class="sample-info">
                            <span class="sample-name">${sample.name}</span>
                            <span class="sample-type ${sample.type.includes('AI') ? 'ai' : 'real'}">
                                ${sample.type}
                            </span>
                            <span class="sample-tag">${sample.tag}</span>
                        </div>
                    </div>
                `).join("");

                // Add click handlers
                document.querySelectorAll(".sample-card").forEach(card => {
                    card.addEventListener("click", () => {
                        const filePath = card.getAttribute("data-file");
                        selectSample(filePath, card.querySelector("img").src);
                    });
                });
            } else {
                samplesGrid.innerHTML = `<p style="color: var(--text-dim); font-size: 13px;">No sample images found.</p>`;
            }
        } catch (err) {
            console.error("Failed to load samples", err);
        }
    }

    function selectSample(filePath, imgSrc) {
        currentFile = null;
        currentSamplePath = filePath;
        previewImage.src = imgSrc;
        dropzoneEmpty.classList.add("hidden");
        dropzonePreview.classList.remove("hidden");
        btnScan.disabled = false;
        
        btnScan.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Drag & Drop
    btnBrowse.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener("click", () => {
        if (!currentFile && !currentSamplePath) {
            fileInput.click();
        }
    });

    ["dragenter", "dragover"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Paste from clipboard
    window.addEventListener("paste", (e) => {
        const items = e.clipboardData?.items;
        if (items) {
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf("image") !== -1) {
                    const blob = items[i].getAsFile();
                    handleFileSelect(blob);
                    break;
                }
            }
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            alert("Vui lòng chọn tệp hình ảnh hợp lệ (JPG, PNG, WEBP).");
            return;
        }
        currentFile = file;
        currentSamplePath = null;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropzoneEmpty.classList.add("hidden");
            dropzonePreview.classList.remove("hidden");
            btnScan.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    btnClear.addEventListener("click", (e) => {
        e.stopPropagation();
        clearInput();
    });

    function clearInput() {
        currentFile = null;
        currentSamplePath = null;
        fileInput.value = "";
        previewImage.src = "";
        dropzonePreview.classList.add("hidden");
        dropzoneEmpty.classList.remove("hidden");
        btnScan.disabled = true;
    }

    // Inspect Action
    btnScan.addEventListener("click", async () => {
        if (!currentFile && !currentSamplePath) return;

        // UI Loading state
        emptyState.classList.add("hidden");
        resultsContent.classList.add("hidden");
        loadingState.classList.remove("hidden");
        btnScan.disabled = true;
        btnScan.innerHTML = `<i data-lucide="loader-2" class="spin"></i><span>Đang giám định CO-SPY v2.0...</span>`;
        lucide.createIcons();

        const formData = new FormData();
        if (currentFile) {
            formData.append("image", currentFile);
        }

        try {
            let response;
            if (currentFile) {
                response = await fetch("/api/scan", {
                    method: "POST",
                    body: formData,
                });
            } else {
                response = await fetch("/api/scan", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ sample_path: currentSamplePath }),
                });
            }

            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || "Không thể phân tích hình ảnh");
            }

            latestForensicData = data;
            renderResults(data);
        } catch (err) {
            alert("Lỗi phân tích: " + err.message);
            emptyState.classList.remove("hidden");
        } finally {
            loadingState.classList.add("hidden");
            btnScan.disabled = false;
            btnScan.innerHTML = `<i data-lucide="scan"></i><span>Giám định với CO-SPY</span>`;
            lucide.createIcons();
        }
    });

    function renderResults(data) {
        resultsContent.classList.remove("hidden");

        // 1. Verdict Banner
        verdictBanner.className = `verdict-banner ${data.is_ai ? "ai" : "real"}`;
        verdictTitle.textContent = data.verdict.toUpperCase();
        verdictCaption.textContent = data.is_ai 
            ? `Phát hiện dấu vết tạo hình nhân tạo AI (${data.confidence}% tin cậy • Mô hình chủ đạo: ${data.winning_expert})`
            : `Đặc tính ảnh chụp tự nhiên được xác minh (${data.confidence}% tin cậy • Chuyên gia ProGAN, SD & Writing đồng thuận)`;

        verdictIcon.setAttribute("data-lucide", data.is_ai ? "alert-triangle" : "shield-check");

        // Tri-Expert Ensemble Badges
        if (scoreProganEl && data.score_progan !== undefined) {
            scoreProganEl.textContent = `${data.score_progan}%`;
        }
        if (scoreSDEl && data.score_sd !== undefined) {
            scoreSDEl.textContent = `${data.score_sd}%`;
        }
        if (scoreWritingEl && data.score_writing !== undefined && data.score_writing !== null) {
            scoreWritingEl.textContent = `${data.score_writing}%`;
        }

        // 2. Circular Gauge
        const circumference = 2 * Math.PI * 42;
        const progressOffset = circumference - (data.confidence / 100) * circumference;
        gaugeFill.style.strokeDashoffset = progressOffset;
        animateNumber(gaugeNumber, 0, data.confidence, 1000);

        // 3. Branch Metrics
        semVal.textContent = `${data.sem_score}%`;
        semBar.style.width = `${data.sem_score}%`;
        artVal.textContent = `${data.art_score}%`;
        artBar.style.width = `${data.art_score}%`;

        if (freqVal && freqBar && data.freq_score !== undefined) {
            freqVal.textContent = `${data.freq_score}%`;
            freqBar.style.width = `${data.freq_score}%`;
        }

        // 4. Assessment Card
        if (originTech && data.origin_tech) {
            originTech.textContent = data.origin_tech;
        }
        if (detailNotes && data.detail_notes) {
            detailNotes.textContent = data.detail_notes;
        }

        // 5. Populate Images
        imgBefore.src = data.original_image;
        imgAfter.src = data.overlay_image;
        sideImgOrig.src = data.original_image;
        sideImgOverlay.src = data.overlay_image;
        pureHeatmapImg.src = data.heatmap_image;
        if (fftImg && data.fft_image) {
            fftImg.src = data.fft_image;
        }

        // Reset Slider to 50%
        setSliderPosition(50);

        // 6. Meta Info
        metaTime.textContent = `${data.inference_time_ms} ms`;
        metaSize.textContent = data.image_size;
        metaDevice.textContent = data.device.toUpperCase();

        lucide.createIcons();
    }

    function animateNumber(element, start, end, duration) {
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const val = (start + (end - start) * progress).toFixed(1);
            element.textContent = val;
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        requestAnimationFrame(update);
    }

    // Interactive Slider Logic
    function setSliderPosition(percent) {
        percent = Math.max(0, Math.min(100, percent));
        imageAfterContainer.style.width = `${percent}%`;
        sliderDivider.style.left = `${percent}%`;
    }

    function onPointerMove(e) {
        if (!isDraggingSlider) return;
        const rect = sliderView.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        const offsetX = clientX - rect.left;
        const percent = (offsetX / rect.width) * 100;
        setSliderPosition(percent);
    }

    sliderDivider.addEventListener("mousedown", () => { isDraggingSlider = true; });
    sliderView.addEventListener("mousedown", (e) => {
        isDraggingSlider = true;
        onPointerMove(e);
    });

    window.addEventListener("mouseup", () => { isDraggingSlider = false; });
    window.addEventListener("mousemove", onPointerMove);

    // Touch support
    sliderDivider.addEventListener("touchstart", () => { isDraggingSlider = true; });
    sliderView.addEventListener("touchstart", (e) => {
        isDraggingSlider = true;
        onPointerMove(e);
    });
    window.addEventListener("touchend", () => { isDraggingSlider = false; });
    window.addEventListener("touchmove", onPointerMove);

    // View Toggles (Slider, Side-by-Side, Heatmap, 2D-FFT)
    toggleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            toggleBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const mode = btn.getAttribute("data-view");
            sliderView.classList.add("hidden");
            sideView.classList.add("hidden");
            heatmapView.classList.add("hidden");
            if (fftView) fftView.classList.add("hidden");

            if (mode === "slider") sliderView.classList.remove("hidden");
            else if (mode === "sideBySide") sideView.classList.remove("hidden");
            else if (mode === "heatmap") heatmapView.classList.remove("hidden");
            else if (mode === "fft" && fftView) fftView.classList.remove("hidden");
        });
    });

    // Opacity Range Slider
    opacityRange.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        opacityVal.textContent = `${Math.round(val * 100)}%`;
        imgAfter.style.opacity = val;
        sideImgOverlay.style.opacity = val;
    });

    // Enhanced Forensic Report Card Generator
    btnDownload.addEventListener("click", async (e) => {
        e.preventDefault();
        if (!latestForensicData) return;

        const data = latestForensicData;

        // Create Offscreen Canvas for high-res forensic report (1280 x 820)
        const canvas = document.createElement("canvas");
        canvas.width = 1280;
        canvas.height = 820;
        const ctx = canvas.getContext("2d");

        // Helper to load image as Promise
        const loadImage = (src) => new Promise((resolve) => {
            if (!src) { resolve(null); return; }
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => resolve(null);
            img.src = src;
        });

        // 1. Background
        const bgGrad = ctx.createLinearGradient(0, 0, 1280, 820);
        bgGrad.addColorStop(0, "#0b0f19");
        bgGrad.addColorStop(1, "#111827");
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, 1280, 820);

        // Outer border
        ctx.strokeStyle = "#1e293b";
        ctx.lineWidth = 2;
        ctx.strokeRect(10, 10, 1260, 800);

        // Header Background
        ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
        ctx.fillRect(20, 20, 1240, 100);
        ctx.strokeStyle = "#334155";
        ctx.strokeRect(20, 20, 1240, 100);

        // Header Title
        ctx.font = "bold 24px 'Segoe UI', system-ui, sans-serif";
        ctx.fillStyle = "#38bdf8";
        ctx.fillText("CO-SPY AI FORENSICS REPORT", 45, 62);

        ctx.font = "14px 'Segoe UI', system-ui, sans-serif";
        ctx.fillStyle = "#94a3b8";
        ctx.fillText("Deep Learning Dual-Expert Ensemble & Spectral Anomaly Verification (CVPR'25)", 45, 92);

        // Verdict Badge on Header Right
        const isAI = data.is_ai;
        const badgeColor = isAI ? "#ef4444" : "#22c55e";
        const badgeBg = isAI ? "rgba(239, 68, 68, 0.15)" : "rgba(34, 197, 94, 0.15)";

        ctx.fillStyle = badgeBg;
        ctx.strokeStyle = badgeColor;
        ctx.lineWidth = 2;
        roundRect(ctx, 920, 36, 320, 68, 8, true, true);

        ctx.font = "bold 20px 'Segoe UI', system-ui, sans-serif";
        ctx.fillStyle = badgeColor;
        ctx.textAlign = "center";
        ctx.fillText(data.verdict.toUpperCase(), 1080, 65);

        ctx.font = "13px 'Segoe UI', system-ui, sans-serif";
        ctx.fillStyle = "#f8fafc";
        ctx.fillText(`Độ tin cậy: ${data.confidence}% | Chuyên gia: ${data.winning_expert}`, 1080, 88);
        ctx.textAlign = "left";

        // Load 3 Images concurrently
        const [origImgObj, camImgObj, fftImgObj] = await Promise.all([
            loadImage(data.original_image),
            loadImage(data.overlay_image),
            loadImage(data.fft_image)
        ]);

        // Panel positions (3 columns)
        const panelW = 380;
        const panelH = 340;
        const panelY = 140;
        const panels = [
            { x: 35, title: "1. ẢNH GỐC ĐẦU VÀO", img: origImgObj, tag: "RGB Input" },
            { x: 450, title: "2. BẢN ĐỒ NHIỆT GRAD-CAM", img: camImgObj, tag: "Spatial Artifacts" },
            { x: 865, title: "3. PHỔ TẦN SỐ 2D-FFT", img: fftImgObj, tag: "Fourier Lattice Peak" }
        ];

        panels.forEach(p => {
            // Box
            ctx.fillStyle = "#0f172a";
            ctx.strokeStyle = "#1e293b";
            ctx.lineWidth = 1;
            roundRect(ctx, p.x, panelY, panelW, panelH, 8, true, true);

            // Title
            ctx.font = "bold 13px 'Segoe UI', system-ui, sans-serif";
            ctx.fillStyle = "#cbd5e1";
            ctx.fillText(p.title, p.x + 15, panelY + 28);

            // Tag badge
            ctx.font = "11px 'Segoe UI', system-ui, sans-serif";
            ctx.fillStyle = "#64748b";
            ctx.textAlign = "right";
            ctx.fillText(p.tag, p.x + panelW - 15, panelY + 28);
            ctx.textAlign = "left";

            // Draw Image inside
            const imgPad = 12;
            const imgBoxX = p.x + imgPad;
            const imgBoxY = panelY + 40;
            const imgBoxW = panelW - imgPad * 2;
            const imgBoxH = panelH - 52;

            ctx.fillStyle = "#020617";
            ctx.fillRect(imgBoxX, imgBoxY, imgBoxW, imgBoxH);

            if (p.img) {
                // Keep aspect ratio
                const scale = Math.min(imgBoxW / p.img.width, imgBoxH / p.img.height);
                const dw = p.img.width * scale;
                const dh = p.img.height * scale;
                const dx = imgBoxX + (imgBoxW - dw) / 2;
                const dy = imgBoxY + (imgBoxH - dh) / 2;
                ctx.drawImage(p.img, dx, dy, dw, dh);
            }
        });

        // 3. Metric Cards (Row 1 under panels: y = 500)
        const metrics = [
            { label: "Nhánh Ngữ Nghĩa (CLIP)", val: `${data.sem_score}%`, color: "#38bdf8" },
            { label: "Nhánh Pixel (ResNet)", val: `${data.art_score}%`, color: "#a855f7" },
            { label: "Nhánh Tần Số (2D-FFT)", val: `${data.freq_score || 0}%`, color: "#ec4899" },
            { label: "GAN Expert (ProGAN)", val: `${data.score_progan || 0}%`, color: "#a5b4fc" },
            { label: "Diffusion (SD-v1.4)", val: `${data.score_sd || 0}%`, color: "#34d399" },
            { label: "Writing Specialist", val: `${data.score_writing || 0}%`, color: "#f472b6" }
        ];

        const metricW = 188;
        const metricH = 75;
        const metricY = 500;
        const metricGap = 16;
        metrics.forEach((m, idx) => {
            const mx = 35 + idx * (metricW + metricGap);
            ctx.fillStyle = "rgba(15, 23, 42, 0.7)";
            ctx.strokeStyle = "#334155";
            roundRect(ctx, mx, metricY, metricW, metricH, 6, true, true);

            ctx.font = "11px 'Segoe UI', system-ui, sans-serif";
            ctx.fillStyle = "#94a3b8";
            ctx.fillText(m.label, mx + 10, metricY + 26);

            ctx.font = "bold 21px 'Segoe UI', system-ui, sans-serif";
            ctx.fillStyle = m.color;
            ctx.fillText(m.val, mx + 10, metricY + 58);
        });

        // 4. Forensic Assessment Box (y = 595, h = 135)
        const assessY = 595;
        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.strokeStyle = isAI ? "rgba(239, 68, 68, 0.4)" : "rgba(34, 197, 94, 0.4)";
        roundRect(ctx, 35, assessY, 1210, 135, 8, true, true);

        ctx.font = "bold 14px 'Segoe UI', system-ui, sans-serif";
        ctx.fillStyle = isAI ? "#f87171" : "#4ade80";
        ctx.fillText("KẾT LUẬN GIÁM ĐỊNH PHÁP CHỨNG & NGUỒN GỐC CÔNG NGHỆ:", 55, assessY + 32);

        ctx.font = "bold 13px 'Segoe UI', system-ui, sans-serif";
        ctx.fillStyle = "#cbd5e1";
        ctx.fillText("Phỏng đoán nguồn gốc: " + (data.origin_tech || "N/A"), 55, assessY + 60);

        ctx.font = "13px 'Segoe UI', system-ui, sans-serif";
        ctx.fillStyle = "#94a3b8";
        // Wrap notes text if needed
        const noteText = data.detail_notes || "";
        ctx.fillText(noteText.slice(0, 160), 55, assessY + 86);
        if (noteText.length > 160) {
            ctx.fillText(noteText.slice(160, 320), 55, assessY + 110);
        }

        // 5. Footer Line (y = 755)
        const dateStr = new Date().toLocaleString("vi-VN");
        ctx.font = "12px 'Segoe UI', system-ui, sans-serif";
        ctx.fillStyle = "#64748b";
        ctx.fillText(`Thời gian giám định: ${dateStr} • Thời gian phân tích: ${data.inference_time_ms} ms • Độ phân giải: ${data.image_size} • Thiết bị: ${data.device.toUpperCase()}`, 35, 765);
        ctx.fillText("Hệ thống giám định hình ảnh CO-SPY v2.0 • Viện Nghiên Cứu & Phát Triển Trí Tuệ Nhân Tạo", 35, 788);

        // Download trigger
        const link = document.createElement("a");
        link.download = `COSPY_Forensic_Report_${Date.now()}.png`;
        link.href = canvas.toDataURL("image/png");
        link.click();
    });

    // Helper for rounded rectangles on canvas
    function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        if (fill) ctx.fill();
        if (stroke) ctx.stroke();
    }
});