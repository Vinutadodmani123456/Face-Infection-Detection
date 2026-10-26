import React, { useState, useRef } from 'react';
import './SkinConditionDetector.css';

/**
 * DermScan AI — React Skin Condition Detection Component
 * 
 * Features:
 * - Single facial photo upload with preview & drag-and-drop
 * - Calibrated Open-Set Rejection handling for unsupported diseases (Eczema, etc.)
 * - Normal / Clear Skin status handling
 * - Multi-label classification for the 4 supported diseases (Acne, Melasma, Rosacea, Seborrheic Dermatitis)
 * - Automatic clearing of previous predictions upon new uploads or rejections
 * - Clean medical disclaimer and dermatologist advisory
 */

export default function SkinConditionDetector({ apiBaseUrl = 'http://localhost:8000' }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [rejection, setRejection] = useState(null);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  // Clear previous output state
  const clearResults = () => {
    setPrediction(null);
    setRejection(null);
    setError(null);
  };

  const handleFileChange = (file) => {
    clearResults();

    if (!file) return;

    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
      setError('Unsupported file format. Please upload JPG, JPEG, PNG, WebP, or BMP.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('Image file is too large. Maximum supported size is 10 MB.');
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    clearResults();
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    clearResults();
    setLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      // Handle HTTP 400 / 422 pre-validation failures
      if (!response.ok || !data.success) {
        const errCode = data.error_code || 'VALIDATION_FAILED';
        if (errCode === 'MULTIPLE_FACES_DETECTED') {
          setRejection({
            type: 'face_alert',
            title: 'Multiple Faces Detected',
            line1: 'The image contains more than one face.',
            line2: 'Please upload an image with only one face.',
          });
        } else if (errCode === 'NO_FACE_DETECTED') {
          setRejection({
            type: 'face_alert',
            title: 'No Face Detected',
            line1: 'No human face was detected in this image.',
            line2: 'Please upload an image with only one face.',
          });
        } else {
          setError(data.error || data.message || 'Unable to classify this image.');
        }
        setLoading(false);
        return;
      }

      // Handle In-Model Rejection (Unsupported skin conditions outside supported categories)
      if (data.is_rejected) {
        setRejection({
          type: 'outside_category',
          badge: 'Unable to Classify',
          heading: 'This skin condition may be outside the supported categories.',
          text: "We couldn't detect a known skin disease from our model. Please try another image or consult a dermatologist.",
        });
        setLoading(false);
        return;
      }

      // Handle Accepted Prediction (In-Distribution or Normal Skin)
      setPrediction(data);

    } catch (err) {
      setError('Network communication failed. Please ensure the backend service is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dermscan-container">
      {/* Header */}
      <header className="dermscan-header">
        <div className="dermscan-badge">MCA Project Prototype</div>
        <h1 className="dermscan-title">
          Facial Skin Condition <span className="highlight-text">Detector</span>
        </h1>
        <p className="dermscan-subtitle">
          AI classification with open-set rejection. Supports 4 conditions with calibrated rejection for unsupported conditions.
        </p>

        {/* Supported Disease Tags */}
        <div className="supported-tags">
          <span className="tag-label">Supported Categories:</span>
          <span className="pill pill-acne">Acne</span>
          <span className="pill pill-melasma">Melasma</span>
          <span className="pill pill-rosacea">Rosacea</span>
          <span className="pill pill-seb">Seborrheic Dermatitis</span>
        </div>
      </header>

      {/* Upload Zone & Analyzing / Alert States */}
      <div className="upload-card">
        {loading ? (
          /* Reference Image 2: Analyzing Loader */
          <div className="analyzing-state-wrap">
            <div className="blue-spinner-ring"></div>
            <p className="analyzing-text">Analyzing Your Image...</p>
          </div>
        ) : rejection && rejection.type === 'face_alert' ? (
          /* Reference Images 1 & 4: Multiple Faces or No Face */
          <div className="face-alert-card">
            <div className="face-alert-icon-wrap">
              <div className="red-circle-exclamation">
                <span>!</span>
              </div>
            </div>
            <h2 className="face-alert-title">{rejection.title}</h2>
            <div className="face-alert-body">
              <p>{rejection.line1}</p>
              <p>{rejection.line2}</p>
            </div>
            <div className="face-alert-actions">
              <button type="button" className="btn-back-home" onClick={handleReset}>Back to Home</button>
            </div>
          </div>
        ) : rejection && rejection.type === 'outside_category' ? (
          /* Reference Image 3: Outside Supported Categories */
          <div className="outside-category-card">
            <div className="outside-card-thumb-wrap">
              <img src={previewUrl} alt="Uploaded skin condition" className="outside-card-img" />
            </div>
            <div className="outside-card-content">
              <span className="unable-pill-badge">{rejection.badge}</span>
              <h2 className="outside-card-heading">{rejection.heading}</h2>
              <p className="outside-card-text">{rejection.text}</p>
              <div className="outside-card-actions">
                <button type="button" className="btn-back-home" onClick={handleReset}>Back to Home</button>
              </div>
            </div>
          </div>
        ) : (
          /* Default Upload / Preview View */
          <>
            {!previewUrl ? (
              <div
                className="dropzone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="dropzone-icon">📷</div>
                <p className="dropzone-title">Click to upload or drag & drop facial image</p>
                <p className="dropzone-hint">Supports high-resolution JPG, PNG, WEBP, BMP (Max 10 MB)</p>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
                  accept="image/jpeg,image/png,image/webp,image/bmp"
                  style={{ display: 'none' }}
                />
              </div>
            ) : (
              <div className="preview-container">
                <img src={previewUrl} alt="Facial skin preview" className="preview-image" />
                <div className="preview-actions">
                  <span className="preview-filename">{selectedFile?.name}</span>
                  <button onClick={handleReset} className="btn-secondary">
                    Remove Image
                  </button>
                </div>
              </div>
            )}

            {/* Action Button */}
            {selectedFile && (
              <div className="action-row">
                <button
                  onClick={handleAnalyze}
                  className="btn-primary"
                >
                  Analyze Skin Image
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Network / Client Error Banner */}
      {error && (
        <div className="alert-card alert-error">
          <div className="alert-icon">❌</div>
          <div className="alert-body">
            <strong>System Error</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* ACCEPTED PREDICTION OUTPUT */}
      {prediction && !prediction.is_rejected && (
        <div className="results-container">
          {/* Normal Skin Banner */}
          {prediction.is_clear_skin && (
            <div className="alert-card alert-success">
              <div className="alert-icon">✨</div>
              <div className="alert-body">
                <h3>Normal / Clear Skin Detected</h3>
                <p>The facial image shows healthy skin with no clinical signs of Acne, Melasma, Rosacea, or Seborrheic Dermatitis.</p>
              </div>
            </div>
          )}

          {/* Condition Cards */}
          <div className="conditions-grid">
            {prediction.detected_conditions.map((item, idx) => (
              <div key={idx} className="condition-card" style={{ borderColor: item.condition_info?.colour || '#6366f1' }}>
                <div className="card-header">
                  <div>
                    <span className="card-ordinal">Detected Condition {prediction.detected_conditions.length > 1 ? `#${idx + 1}` : ''}</span>
                    <h2 className="card-title" style={{ color: item.condition_info?.colour || '#6366f1' }}>
                      {item.class}
                    </h2>
                  </div>
                  <div className="confidence-badge" style={{ backgroundColor: item.condition_info?.colour || '#6366f1' }}>
                    {item.confidence}% Match
                  </div>
                </div>

                <p className="card-description">{item.condition_info?.description}</p>

                {item.condition_info?.common_symptoms && (
                  <div className="symptoms-section">
                    <h4>Clinical Symptoms:</h4>
                    <ul>
                      {item.condition_info.common_symptoms.map((sym, sIdx) => (
                        <li key={sIdx}>{sym}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {item.prescription && (
                  <div className="guidance-section">
                    <h4>Skincare Guidance & Treatments:</h4>
                    {item.prescription.topical_treatments && (
                      <div className="rx-group">
                        <strong>Topical Care:</strong>
                        <ul>
                          {item.prescription.topical_treatments.map((t, tIdx) => (
                            <li key={tIdx}>{t}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {item.prescription.care_routine && (
                      <div className="rx-group">
                        <strong>Daily Routine:</strong>
                        <ul>
                          {item.prescription.care_routine.map((r, rIdx) => (
                            <li key={rIdx}>{r}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Medical Disclaimer */}
          <div className="disclaimer-card">
            <strong>⚠️ Medical Disclaimer</strong>
            <p>{prediction.disclaimer}</p>
          </div>
        </div>
      )}
    </div>
  );
}
