/* ================================================================
   DermScan AI — Multi-Label Facial Skin Condition Detection
   Frontend JavaScript (Multi-Label Enhanced)
   ================================================================ */

'use strict';

// ── Condition metadata (icons + accent colours) ────────────────────
const CONDITION_META = {
  'Acne':                    { icon: '🔴', colour: '#E74C3C', gradient: 'linear-gradient(135deg, #E74C3C, #C0392B)' },
  'Rosacea':                 { icon: '🟠', colour: '#E67E22', gradient: 'linear-gradient(135deg, #E67E22, #D35400)' },
  'Melasma':                 { icon: '🟣', colour: '#8E44AD', gradient: 'linear-gradient(135deg, #8E44AD, #7D3C98)' },
  'Seborrheic Dermatitis':   { icon: '🟢', colour: '#2ECC71', gradient: 'linear-gradient(135deg, #2ECC71, #27AE60)' },
  'Normal / Clear Skin':     { icon: '✨', colour: '#10B981', gradient: 'linear-gradient(135deg, #10B981, #059669)' },
  'Clear Skin':              { icon: '✨', colour: '#10B981', gradient: 'linear-gradient(135deg, #10B981, #059669)' },
  'Normal Facial Skin':      { icon: '✨', colour: '#10B981', gradient: 'linear-gradient(135deg, #10B981, #059669)' },
  'Unknown Facial Condition':{ icon: '⚠️', colour: '#F59E0B', gradient: 'linear-gradient(135deg, #F59E0B, #D97706)' },
  'Unknown Image':           { icon: '⚠️', colour: '#F59E0B', gradient: 'linear-gradient(135deg, #F59E0B, #D97706)' },
  // Clear skin (no disease) key
  'This person doesnt have any disease': {
    icon: '✨', colour: '#10B981', gradient: 'linear-gradient(135deg, #10B981, #059669)'
  },
  // Unknown / other disease key
  'This is not acne rosacea melasma and seborrheic disease this is some another disease': {
    icon: '⚠️', colour: '#F59E0B', gradient: 'linear-gradient(135deg, #F59E0B, #D97706)'
  },
};

const DEFAULT_META = { icon: '🩺', colour: '#6366f1', gradient: 'linear-gradient(135deg, #6366f1, #a855f7)' };

// ── Element references ─────────────────────────────────────────────
const dropZone         = document.getElementById('drop-zone');
const fileInput        = document.getElementById('file-input');
const previewWrapper   = document.getElementById('preview-wrapper');
const previewImg       = document.getElementById('preview-img');
const previewFilename  = document.getElementById('preview-filename');
const btnClear         = document.getElementById('btn-clear');
const btnAnalyse       = document.getElementById('btn-analyse');
const errorBanner      = document.getElementById('error-banner');
const errorText        = document.getElementById('error-text');

const uploadCard       = document.getElementById('upload-card');
const uploadSection    = document.querySelector('.upload-section');
const resultSection    = document.getElementById('result-section');
const multiAlert       = document.getElementById('multi-alert');
const multiAlertDetail = document.getElementById('multi-alert-detail');
const clearSkinAlert   = document.getElementById('clear-skin-alert');
const clearSkinDetail  = document.getElementById('clear-skin-detail');
const conditionsGrid   = document.getElementById('conditions-grid');
const probaBars        = document.getElementById('proba-bars');
const disclaimerText   = document.getElementById('disclaimer-text');
const btnRetry         = document.getElementById('btn-retry');

// Reference UI elements
const faceAlertCard       = document.getElementById('face-alert-card');
const faceAlertTitle      = document.getElementById('face-alert-title');
const faceAlertLine1      = document.getElementById('face-alert-line1');
const faceAlertLine2      = document.getElementById('face-alert-line2');
const btnFaceBackHome     = document.getElementById('btn-face-back-home');
const btnFaceTryAgain     = document.getElementById('btn-face-try-again'); // may be null (button removed)
// ── State ──────────────────────────────────────────────────────────
let selectedFile = null;

// Reset all cards and alerts
function resetAllCards() {
  if (faceAlertCard) faceAlertCard.hidden = true;
  if (resultSection) resultSection.hidden = true;
  if (multiAlert) multiAlert.hidden = true;
  if (clearSkinAlert) clearSkinAlert.hidden = true;
  hideError();
}

// Wire up Reset/Home buttons
[btnFaceTryAgain, btnFaceBackHome].forEach(btn => {
  if (btn) {
    btn.addEventListener('click', () => {
      resetAllCards();
      resetUpload();
      if (uploadSection) uploadSection.style.display = '';
      if (dropZone) dropZone.style.display = '';
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }
});

// ── Drop Zone — click to open file picker ─────────────────────────
dropZone.addEventListener('click', (e) => {
  // Only trigger if the click was NOT directly on the file input itself
  if (e.target !== fileInput) {
    fileInput.click();
  }
});

// ── Drop Zone — drag events ────────────────────────────────────────
['dragenter', 'dragover'].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
});
['dragleave', 'drop'].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
  });
});
dropZone.addEventListener('drop', e => {
  const files = e.dataTransfer.files;
  if (files && files[0]) handleFileSelected(files[0]);
});

// ── Drop Zone — keyboard activation ────────────────────────────────
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files[0]) {
    handleFileSelected(fileInput.files[0]);
  }
});

// ── File selection ─────────────────────────────────────────────────
function handleFileSelected(file) {
  if (!file) return;

  const validMimes = [
    'image/jpeg', 'image/png', 'image/bmp', 'image/webp',
    'image/jpg', 'image/pjpeg', 'image/jfif', 'image/x-png'
  ];
  const ext = (file.name || '').split('.').pop().toLowerCase();
  const validExts = ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'jfif'];

  const type = (file.type || '').toLowerCase();
  const isValid = validMimes.includes(type) || validExts.includes(ext);

  if (!isValid) {
    showError('Unsupported file format. Please upload JPG, JPEG, PNG, WebP, or BMP.');
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    showError('File too large. Maximum size is 15 MB.');
    return;
  }

  hideError();
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.alt = `Preview of ${file.name}`;
    previewWrapper.classList.add('visible');
    previewWrapper.style.display = 'flex';
    dropZone.style.display = 'none';
    if (previewFilename) {
      previewFilename.textContent = file.name;
    }
    // Safely reset file input value once image is loaded into memory
    fileInput.value = '';
  };
  reader.onerror = err => {
    console.error('FileReader error:', err);
    showError('Failed to read image file.');
  };
  reader.readAsDataURL(file);

  btnAnalyse.disabled = false;
}

// ── Clear ──────────────────────────────────────────────────────────
btnClear.addEventListener('click', () => {
  resetUpload();
});

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src  = '';
  previewWrapper.classList.remove('visible');
  previewWrapper.style.display = 'none';
  dropZone.style.display = '';
  btnAnalyse.style.display = '';
  btnAnalyse.classList.remove('loading');
  btnAnalyse.disabled = true;
  hideError();
}

// ── Analyse ────────────────────────────────────────────────────────
btnAnalyse.addEventListener('click', async () => {
  if (!selectedFile) return;
  await runPrediction(selectedFile);
});

async function runPrediction(file) {
  if (!file) return;

  // Clear any existing prediction output and error state
  if (resultSection) resultSection.hidden = true;
  if (multiAlert) multiAlert.hidden = true;
  if (clearSkinAlert) clearSkinAlert.hidden = true;
  const otherDiseaseAlert = document.getElementById('other-disease-alert');
  if (otherDiseaseAlert) otherDiseaseAlert.hidden = true;
  if (faceAlertCard) faceAlertCard.hidden = true;
  if (conditionsGrid) conditionsGrid.innerHTML = '';
  if (probaBars) probaBars.innerHTML = '';
  hideError();

  // Set button to loading state
  if (btnAnalyse) {
    btnAnalyse.classList.add('loading');
    btnAnalyse.disabled = true;
  }

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (btnAnalyse) {
      btnAnalyse.classList.remove('loading');
      btnAnalyse.disabled = false;
    }

    // Handle Invalid Images and Face Pre-validation Rejections
    if (!response.ok || !data.success) {
      const errCode = data.error_code || '';
      const isInvalidImage = [
        'INVALID_IMAGE',
        'NO_FACE_INFECTION_DETECTED',
        'MULTIPLE_FACES_DETECTED',
        'NO_FACE_DETECTED',
        'IMAGE_TOO_BLURRY',
        'IMAGE_TOO_DARK',
        'IMAGE_TOO_BRIGHT',
      ].includes(errCode) || data.title === 'Invalid Image' || data.is_rejected;

      if (isInvalidImage) {
        if (faceAlertTitle) faceAlertTitle.textContent = data.title || 'Invalid Image';
        if (faceAlertLine1) faceAlertLine1.textContent = data.line1 || data.error || data.message || 'This image is invalid for facial skin infection analysis.';
        if (faceAlertLine2) faceAlertLine2.textContent = data.line2 || 'Please upload a clear image of a single face showing a facial skin condition.';
        if (faceAlertCard) {
          faceAlertCard.hidden = false;
          faceAlertCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        return;
      }

      // Other generic errors (e.g. format, size)
      if (previewWrapper) previewWrapper.style.display = '';
      if (btnAnalyse) {
        btnAnalyse.style.display = '';
        btnAnalyse.classList.remove('loading');
        btnAnalyse.disabled = false;
      }
      const errMsg = data.error || data.message || `Server error (${response.status}). Please try again.`;
      showError(errMsg);
      return;
    }


    // Standard Supported Disease Classification Result
    renderResult(data);

  } catch (err) {
    console.error('Analyse error:', err);
    if (previewWrapper) previewWrapper.style.display = '';
    if (btnAnalyse) {
      btnAnalyse.style.display = '';
      btnAnalyse.classList.remove('loading');
      btnAnalyse.disabled = false;
    }
    showError('Network error. Make sure the server is running and try again.');
  }
}


// ════════════════════════════════════════════════════════════════════
//  RENDER RESULT — Multi-Label Enhanced
// ════════════════════════════════════════════════════════════════════
// ════════════════════════════════════════════════════════════════════
//  RENDER RESULT — Multi-Label & Clear Skin Enhanced
// ════════════════════════════════════════════════════════════════════
function renderResult(data) {
  const {
    is_rejected,
    is_clear_skin,
    is_other_disease,
    is_multi_condition,
    detected_conditions,
    probabilities,
    decision_metrics,
    affected_area,
    disclaimer,
    message,
    detailed_reason,
  } = data;

  // Grab alerts and probability card
  const otherDiseaseAlert = document.getElementById('other-disease-alert');
  const probaCard = probaBars ? probaBars.closest('.card') : null;
  const affectedCard = document.getElementById('result-affected-wrap');
  const disclaimerBox = document.querySelector('.disclaimer-box');

  // -- Alert banners: rejection / clear skin / other disease / multi-condition
  if (is_rejected || is_other_disease) {
    if (clearSkinAlert) clearSkinAlert.hidden = true;
    if (multiAlert) multiAlert.hidden = true;
    if (otherDiseaseAlert) {
      const msgSpan = otherDiseaseAlert.querySelector('span');
      if (msgSpan) {
        let cleanMsg = message || 'This skin condition is outside the 4 supported categories.';
        cleanMsg = cleanMsg.replace(/^Unknown Facial Condition Detected\s*[—–-]\s*/i, '');
        msgSpan.innerHTML = `${cleanMsg}<br><small style="opacity:0.85;margin-top:6px;display:block;">${detailed_reason || 'This is not Acne, Melasma, Rosacea, or Seborrheic Dermatitis. Please consult a dermatologist for a professional clinical diagnosis.'}</small>`;
      }
      otherDiseaseAlert.hidden = false;
      otherDiseaseAlert.classList.add('animate-in');
    }
    // Suppress all cards / explanations below the alert for unknown conditions
    conditionsGrid.innerHTML = '';
    if (affectedCard) affectedCard.style.display = 'none';
    if (probaCard) probaCard.style.display = 'none';
    if (disclaimerBox) disclaimerBox.style.display = 'none';
  } else if (is_clear_skin) {
    if (clearSkinAlert) { clearSkinAlert.hidden = false; clearSkinAlert.classList.add('animate-in'); }
    if (multiAlert) multiAlert.hidden = true;
    if (otherDiseaseAlert) otherDiseaseAlert.hidden = true;
    // Suppress all cards / explanations below the alert for normal skin
    conditionsGrid.innerHTML = '';
    if (affectedCard) affectedCard.style.display = 'none';
    if (probaCard) probaCard.style.display = 'none';
    if (disclaimerBox) disclaimerBox.style.display = 'none';
  } else {
    if (clearSkinAlert) clearSkinAlert.hidden = true;
    if (otherDiseaseAlert) otherDiseaseAlert.hidden = true;
    if (multiAlert) multiAlert.hidden = true;
    if (affectedCard) affectedCard.style.display = '';
    if (probaCard) probaCard.style.display = '';
    if (disclaimerBox) disclaimerBox.style.display = '';
  }

  // -- Render condition cards (Single Disease Condition) ----------------------
  conditionsGrid.innerHTML = '';
  if (!is_rejected && !is_other_disease && !is_clear_skin) {
    const conditionsToRender = (detected_conditions || []).slice(0, 1);

    conditionsToRender.forEach((cond, index) => {
    const meta   = CONDITION_META[cond.class] || DEFAULT_META;
    const colour = cond.condition_info.colour || meta.colour;
    const info   = cond.condition_info;
    const rx     = cond.prescription || {};
    const isClear = is_clear_skin ||
      cond.class === 'Normal Facial Skin' ||
      cond.class === 'This person doesnt have any disease' ||
      cond.class.includes('Clear') ||
      cond.class.includes('Normal');
    const isOther = is_other_disease ||
      cond.class === 'Unknown Facial Condition' ||
      cond.class.includes('another disease') ||
      cond.class.includes('Unknown') ||
      cond.class.includes('Unrecognised');

    const card = document.createElement('div');
    card.className = `card condition-card animate-card ${isClear ? 'condition-card-clear' : ''} ${isOther ? 'condition-card-other' : ''}`;
    card.style.setProperty('--card-accent', colour);
    card.style.animationDelay = `${index * 0.15}s`;

    const ordinalLabel = isClear
      ? '✨ Health Status'
      : isOther
        ? '⚠️ Unknown Facial Condition'
        : 'Detected Disease';

    const rxTitle      = isClear ? 'Daily Skincare & Protection'
                       : isOther ? 'Recommended Action'
                       : 'Treatment Guidance';
    const topicalTitle = isClear ? '🧴 Daily Barrier Care & Sunscreen'
                       : isOther ? '🚫 What NOT to Do'
                       : '🧴 Topical Treatments';
    const oralTitle    = isClear ? '🥗 Hydration & Lifestyle'
                       : isOther ? '🏥 Seek Professional Help'
                       : '💊 Oral Medications';
    const careTitle    = isClear ? '🌿 Daily Skincare Routine' : '🌿 Daily Skincare Routine';

    card.innerHTML = `
      <div class="condition-card-header" style="border-color: ${colour}">
        <div class="condition-badge" style="background: ${meta.gradient}">
          <span class="condition-badge-icon">${meta.icon}</span>
          <span class="condition-badge-conf">${isOther ? 'Unknown' : cond.confidence + '%'}</span>
        </div>
        <div class="condition-header-text">
          <span class="condition-ordinal">${ordinalLabel}</span>
          <h2 class="condition-name" style="color: ${colour}">${cond.class}</h2>
        </div>
      </div>

      <!-- Confidence Bar -->
      <div class="cond-confidence-wrap">
        <div class="cond-confidence-track">
          <div class="cond-confidence-fill" style="background: ${meta.gradient}" data-width="${cond.confidence}"></div>
        </div>
      </div>

      <!-- Description -->
      <div class="cond-section">
        <h3 class="cond-section-label">${isClear ? 'About Your Skin Assessment' : 'About This Condition'}</h3>
        <p class="cond-description">${info.description || ''}</p>
      </div>

      <!-- Symptoms / Features -->
      <div class="cond-section">
        <h3 class="cond-section-label">${isClear ? 'Observed Skin Features' : 'Common Symptoms'}</h3>
        <ul class="cond-symptoms-list">
          ${(info.common_symptoms || []).map(s => `<li>${s}</li>`).join('')}
        </ul>
      </div>

      <!-- Prescription / Care Guidance -->
      <div class="cond-prescription-box">
        <div class="rx-header">
          <span class="rx-icon">${isClear ? '✨' : '💊'}</span>
          <h3 class="rx-title">${rxTitle}</h3>
        </div>

        <div class="rx-grid">
          <div class="rx-card">
            <div class="rx-card-title">${topicalTitle}</div>
            <ul class="rx-list">
              ${(rx.topical_treatments || []).map(t => `<li>${t}</li>`).join('')}
            </ul>
          </div>
          <div class="rx-card">
            <div class="rx-card-title">${oralTitle}</div>
            <ul class="rx-list">
              ${(rx.oral_medications || []).map(t => `<li>${t}</li>`).join('')}
            </ul>
          </div>
          <div class="rx-card rx-card-full">
            <div class="rx-card-title">${careTitle}</div>
            <ul class="rx-list">
              ${(rx.care_routine || []).map(t => `<li>${t}</li>`).join('')}
            </ul>
          </div>
        </div>
      </div>
    `;

    conditionsGrid.appendChild(card);
    });
  }

  // -- Affected Area --------------------------------------------------------
  const affected = affected_area || { percentage: 0.0, severity: 'Clear / Healthy', severity_class: 'severity-clear' };
  const affectedNumber = document.getElementById('affected-number');
  const severityPill   = document.getElementById('severity-pill');
  const affectedFill   = document.getElementById('affected-bar-fill');

  if (affectedNumber) affectedNumber.textContent = `${affected.percentage}%`;
  if (severityPill) {
    if (is_clear_skin || (affected.severity && affected.severity.toLowerCase().includes('clear'))) {
      severityPill.textContent = 'Clear / Healthy Skin';
      severityPill.className = 'severity-pill severity-clear';
    } else {
      severityPill.textContent = `${affected.severity} Severity`;
      severityPill.className = `severity-pill severity-${affected.severity.toLowerCase()}`;
    }
  }

  // -- Probability Bars -----------------------------------------------------
  probaBars.innerHTML = '';
  const sortedEntries = Object.entries(probabilities || {})
    .sort((a, b) => b[1] - a[1]);

  // Determine which classes are "detected" (above threshold)
  const detectedClassNames = new Set(detected_conditions.map(c => c.class));

  sortedEntries.forEach(([cls, prob]) => {
    const clsMeta   = CONDITION_META[cls] || DEFAULT_META;
    const clsColour = clsMeta.colour;
    const isDetected = detectedClassNames.has(cls);

    const row = document.createElement('div');
    row.className = `proba-row ${isDetected ? 'proba-detected' : ''}`;
    row.innerHTML = `
      <div class="proba-label-row">
        <span class="proba-name" style="${isDetected ? `color:${clsColour};font-weight:700` : ''}">${cls}</span>
        <span class="proba-value">${prob.toFixed(1)}%</span>
      </div>
      <div class="proba-track">
        <div class="proba-fill" style="background:${isDetected ? clsMeta.gradient : clsColour};opacity:${isDetected ? 1 : 0.3}" data-width="${prob}"></div>
      </div>
      ${isDetected ? '<span class="proba-detected-badge">Detected</span>' : ''}
    `;
    probaBars.appendChild(row);
  });

  // -- Disclaimer -----------------------------------------------------------
  disclaimerText.textContent = (!is_rejected && !is_other_disease && !is_clear_skin) ? disclaimer : '';

  // -- Show result section, hide upload section ------------------------------
  uploadSection.style.display = 'none';
  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // -- Animate bars after paint ---------------------------------------------
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      // Affected area bar
      if (affectedFill) {
        affectedFill.style.width = `${affected.percentage}%`;
      }

      // Condition confidence fills
      const condFills = conditionsGrid.querySelectorAll('.cond-confidence-fill');
      condFills.forEach((fill, i) => {
        setTimeout(() => {
          fill.style.width = `${fill.dataset.width}%`;
        }, i * 100 + 200);
      });

      // Proba fills (staggered)
      const fills = probaBars.querySelectorAll('.proba-fill');
      fills.forEach((fill, i) => {
        setTimeout(() => {
          fill.style.width = `${fill.dataset.width}%`;
        }, i * 80 + 300);
      });
    });
  });

  btnAnalyse.classList.remove('loading');
}


// ── Retry ──────────────────────────────────────────────────────────
btnRetry.addEventListener('click', () => {
  resultSection.hidden = true;
  multiAlert.hidden = true;
  if (clearSkinAlert) clearSkinAlert.hidden = true;
  const otherDiseaseAlert = document.getElementById('other-disease-alert');
  if (otherDiseaseAlert) otherDiseaseAlert.hidden = true;

  const affectedCard = document.getElementById('result-affected-wrap');
  const probaCard = probaBars ? probaBars.closest('.card') : null;
  const disclaimerBox = document.querySelector('.disclaimer-box');
  if (affectedCard) affectedCard.style.display = '';
  if (probaCard) probaCard.style.display = '';
  if (disclaimerBox) disclaimerBox.style.display = '';

  uploadSection.style.display = '';
  resetUpload();

  // Reset result UI
  conditionsGrid.innerHTML = '';
  probaBars.innerHTML      = '';
  disclaimerText.textContent = '';

  const affectedFill = document.getElementById('affected-bar-fill');
  if (affectedFill) affectedFill.style.width = '0%';

  uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
});


// ── Error Helpers ──────────────────────────────────────────────────
function showError(msg) {
  errorText.textContent = msg;
  errorBanner.hidden    = false;
  errorBanner.classList.add('animate-in');
}
function hideError() {
  errorBanner.hidden = true;
  errorBanner.classList.remove('animate-in');
  errorText.textContent = '';
}
