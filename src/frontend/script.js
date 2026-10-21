// QRadAssist Unified Frontend Workflow Controller
// Single-Upload, Consolidated Multi-Label Analysis & Multi-Class Grad-CAM Rendering

let currentFile = null;
let currentImageURL = null;

// Disease color palettes (RGB)
const DISEASE_COLORS = {
  Pneumonia: { r: 255, g: 69, b: 0, name: "Red / Orange" },       // Orange-Red
  Atelectasis: { r: 0, g: 191, b: 255, name: "Blue / Cyan" },     // Deep Sky Blue
  "Pleural Effusion": { r: 50, g: 205, b: 50, name: "Green / Lime" }, // Lime Green
};

// Map disease name to container IDs
const DISEASE_CARD_IDS = {
  Pneumonia: {
    wrap: "wrap-pneumonia",
    badge: "badge-prob-pneumonia",
  },
  Atelectasis: {
    wrap: "wrap-atelectasis",
    badge: "badge-prob-atelectasis",
  },
  "Pleural Effusion": {
    wrap: "wrap-effusion",
    badge: "badge-prob-effusion",
  },
};

// Initialize event listeners when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("xray-file");
  const dropZone = document.getElementById("drop-zone");
  const analyzeBtn = document.getElementById("analyze-btn");
  const changeImgBtn = document.getElementById("change-image-btn");

  // File selection via input
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelected(file);
  });

  // Drag and drop handlers
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelected(file);
  });

  // Change image button
  changeImgBtn?.addEventListener("click", () => {
    fileInput.value = "";
    fileInput.click();
  });

  // Primary Analyze button
  analyzeBtn.addEventListener("click", runUnifiedAnalysis);
});

// Handle newly chosen file
function handleFileSelected(file) {
  if (!file.type.startsWith("image/")) {
    showError("Selected file is not a supported image format. Please select a PNG or JPEG radiograph.");
    return;
  }

  currentFile = file;
  hideError();

  // Create preview object URL
  if (currentImageURL) {
    URL.revokeObjectURL(currentImageURL);
  }
  currentImageURL = URL.createObjectURL(file);

  // Update preview UI
  const previewImg = document.getElementById("image-preview");
  const uploadPrompt = document.getElementById("upload-prompt");
  const previewWrapper = document.getElementById("preview-wrapper");
  const fileNameSpan = document.getElementById("file-name");
  const analyzeBtn = document.getElementById("analyze-btn");

  previewImg.src = currentImageURL;
  fileNameSpan.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

  uploadPrompt.style.display = "none";
  previewWrapper.style.display = "flex";
  analyzeBtn.disabled = false;
}

// Error presentation helpers
function showError(msg) {
  const banner = document.getElementById("error-banner");
  banner.textContent = `❌ ${msg}`;
  banner.style.display = "block";
}

function hideError() {
  const banner = document.getElementById("error-banner");
  banner.style.display = "none";
  banner.textContent = "";
}

// Ensure probability is strictly in [0, 1] range (single-sigmoid guarantee)
function normalizeProbability(val) {
  if (typeof val !== "number") return 0.0;
  // If outside [0, 1], treat as uncalibrated logit and apply sigmoid once
  if (val < 0.0 || val > 1.0) {
    return 1.0 / (1.0 + Math.exp(-val));
  }
  return val;
}

// Execute complete unified analysis
async function runUnifiedAnalysis() {
  if (!currentFile) {
    showError("Please select a chest X-ray image before running analysis.");
    return;
  }

  const analyzeBtn = document.getElementById("analyze-btn");
  const loader = document.getElementById("analyze-loader");
  const reportContainer = document.getElementById("report-container");

  // Set loading state
  analyzeBtn.disabled = true;
  loader.style.display = "flex";
  hideError();

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const resp = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.detail || `Server returned error status ${resp.status}`);
    }

    // Populate and show the unified report
    renderConsolidatedReport(data);

    reportContainer.style.display = "block";
    reportContainer.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    console.error("Analysis pipeline error:", err);
    showError(`Analysis failed: ${err.message || err}. Please verify the server is running.`);
  } finally {
    analyzeBtn.disabled = false;
    loader.style.display = "none";
  }
}

// Render the consolidated report with all components
function renderConsolidatedReport(data) {
  // 1. Report Timestamp
  const tsEl = document.getElementById("report-timestamp");
  tsEl.textContent = `Generated on: ${new Date().toLocaleString()} • Assessment Mode: Multi-Label Frontal CXR`;

  // 2. Set source image in report
  const sourceImg = document.getElementById("report-source-img");
  sourceImg.src = currentImageURL;

  // 3. Render Probabilities with styled progress bars
  const probs = data.probabilities || {};
  const probListContainer = document.getElementById("probabilities-list");
  probListContainer.innerHTML = "";

  const diseases = ["Pneumonia", "Atelectasis", "Pleural Effusion"];

  diseases.forEach((disease) => {
    const rawVal = probs[disease];
    const prob = normalizeProbability(rawVal);
    const pct = (prob * 100).toFixed(1);

    const item = document.createElement("div");
    item.className = "prob-item";

    const colorMeta = DISEASE_COLORS[disease] || { r: 255, g: 126, b: 95 };
    const themeColor = `rgb(${colorMeta.r}, ${colorMeta.g}, ${colorMeta.b})`;

    item.innerHTML = `
      <div class="prob-header">
        <span class="prob-name">${disease}</span>
        <span class="prob-pct" style="color: ${themeColor}">${pct}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${pct}%; background: ${themeColor};"></div>
      </div>
      <div class="prob-footer">
        <span class="prob-subtext">${prob >= 0.5 ? "Positive Finding (≥ 50%)" : "Negative / Sub-threshold (< 50%)"}</span>
        <span class="prob-subtext">Predicted Prob: ${prob.toFixed(4)}</span>
      </div>
    `;
    probListContainer.appendChild(item);

    // Update heatmap card badge
    const badgeId = DISEASE_CARD_IDS[disease]?.badge;
    if (badgeId) {
      const badgeEl = document.getElementById(badgeId);
      if (badgeEl) {
        badgeEl.textContent = `${pct}% probability`;
      }
    }
  });

  // 4. Render AI-Drafted Findings
  const findingsList = document.getElementById("findings-list");
  findingsList.innerHTML = "";
  const findings = data.report?.findings || [];

  if (findings.length === 0) {
    const li = document.createElement("li");
    li.className = "finding-item finding-negative";
    li.textContent = "No acute cardiopulmonary findings exceeding the standard clinical reporting threshold (≥ 0.50).";
    findingsList.appendChild(li);
  } else {
    findings.forEach((f) => {
      const li = document.createElement("li");
      li.className = "finding-item finding-positive";
      li.textContent = f;
      findingsList.appendChild(li);
    });
  }

  // 5. Render Impression
  const impressionBox = document.getElementById("impression-text");
  impressionBox.textContent = data.report?.impression || "No acute findings identified.";

  // 6. Render the 3 Disease-Specific Grad-CAM Overlays
  renderAllGradCAMOverlays(data.heatmaps || {});
}

// Render Grad-CAM overlays for all three diseases
function renderAllGradCAMOverlays(heatmaps) {
  const diseases = ["Pneumonia", "Atelectasis", "Pleural Effusion"];

  diseases.forEach((disease) => {
    const ids = DISEASE_CARD_IDS[disease];
    if (!ids) return;

    const wrapper = document.getElementById(ids.wrap);
    if (!wrapper) return;

    wrapper.innerHTML = ""; // Clear existing

    const hmData = heatmaps[disease];

    // Handle partial failure gracefully
    if (!hmData || hmData.status === "error" || !hmData.heatmap) {
      const errBox = document.createElement("div");
      errBox.className = "heatmap-error";
      errBox.textContent = hmData?.error || `Grad-CAM visualization unavailable for ${disease}.`;
      wrapper.appendChild(errBox);
      return;
    }

    // Render original image with transparent colored heatmap overlay
    const img = document.createElement("img");
    img.src = currentImageURL;
    img.className = "gradcam-base-img";

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width || 224;
      canvas.height = img.height || 224;
      canvas.className = "gradcam-overlay-canvas";

      const ctx = canvas.getContext("2d");
      const hm = hmData.heatmap;
      const rows = hm.length;
      const cols = hm[0].length;

      const scaleX = canvas.width / cols;
      const scaleY = canvas.height / rows;

      const color = DISEASE_COLORS[disease] || { r: 255, g: 69, b: 0 };

      // Paint activation cells with custom disease color and intensity-scaled alpha
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const intensity = hm[y][x]; // Float in [0.0, 1.0]
          if (intensity > 0.05) {
            // Apply quadratic alpha scaling for crisp visual contrast against grayscale X-ray
            const alpha = Math.min(0.75, Math.pow(intensity, 1.2) * 0.7);
            ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha})`;
            ctx.fillRect(x * scaleX, y * scaleY, scaleX + 0.5, scaleY + 0.5);
          }
        }
      }

      wrapper.appendChild(img);
      wrapper.appendChild(canvas);
    };
  });
}
