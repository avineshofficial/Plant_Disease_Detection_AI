/* ═══════════════════════════════════════════════════════════
   PlantGuard AI — Frontend Logic
   Upload, Webcam capture, Prediction, Result rendering
   ═══════════════════════════════════════════════════════════ */

$(document).ready(function () {
    let webcamStream = null;
    let capturedImageData = null;

    // ── Mode Switching ──────────────────────────────────────
    window.switchMode = function (mode) {
        if (mode === 'upload') {
            $('#upload-section').show();
            $('#webcam-section').hide();
            $('#btn-upload-mode').addClass('active');
            $('#btn-webcam-mode').removeClass('active');
            stopCamera();
        } else {
            $('#upload-section').hide();
            $('#webcam-section').show();
            $('#btn-webcam-mode').addClass('active');
            $('#btn-upload-mode').removeClass('active');
        }
        hideResults();
    };

    // ── Upload Mode ─────────────────────────────────────────

    // Click drop zone to trigger file input
    $('#drop-zone').on('click', function (e) {
        if (e.target.closest('#btn-remove-image') || e.target.closest('.btn-upload')) return;
        if ($('#preview-container').is(':visible')) return;
        $('#imageUpload').click();
    });

    // Drag & drop handlers
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
        ['dragenter', 'dragover'].forEach(evt => {
            dropZone.addEventListener(evt, function (e) {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });
        });
        ['dragleave', 'drop'].forEach(evt => {
            dropZone.addEventListener(evt, function (e) {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
            });
        });
        dropZone.addEventListener('drop', function (e) {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('imageUpload').files = files;
                showImagePreview(files[0]);
            }
        });
    }

    // File input change
    $('#imageUpload').on('change', function () {
        if (this.files && this.files[0]) {
            showImagePreview(this.files[0]);
        }
    });

    function showImagePreview(file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            $('#imagePreview').attr('src', e.target.result);
            $('#drop-zone-content').hide();
            $('#preview-container').show();
            $('#predict-section').show();
        };
        reader.readAsDataURL(file);
        hideResults();
    }

    // Remove image
    $('#btn-remove-image').on('click', function (e) {
        e.stopPropagation();
        $('#imageUpload').val('');
        $('#preview-container').hide();
        $('#drop-zone-content').show();
        $('#predict-section').hide();
        hideResults();
    });

    // Predict from upload
    $('#btn-predict').on('click', function () {
        const formData = new FormData($('#upload-file')[0]);
        showLoader();
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: formData,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                hideLoader();
                renderResults(data);
            },
            error: function (xhr) {
                hideLoader();
                let msg = 'An error occurred during prediction.';
                try { msg = JSON.parse(xhr.responseText).error || msg; } catch (e) { /* ignore */ }
                alert(msg);
            }
        });
    });

    // ── Webcam Mode ─────────────────────────────────────────

    $('#btn-start-camera').on('click', startCamera);
    $('#btn-stop-camera').on('click', stopCamera);
    $('#btn-capture').on('click', captureFrame);
    $('#btn-retake').on('click', retakeWebcam);
    $('#btn-predict-webcam').on('click', predictWebcam);

    function startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Your browser does not support webcam access. Please use Chrome, Firefox, or Safari.');
            return;
        }
        navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } }
        })
        .then(function (stream) {
            webcamStream = stream;
            const video = document.getElementById('webcam-video');
            video.srcObject = stream;
            video.play();
            $('#video-overlay').hide();
            $('#btn-start-camera').hide();
            $('#btn-capture').show();
            $('#btn-stop-camera').show();
        })
        .catch(function (err) {
            console.error('Camera error:', err);
            alert('Unable to access camera. Please allow camera permissions and try again.');
        });
    }

    function stopCamera() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        const video = document.getElementById('webcam-video');
        if (video) video.srcObject = null;

        $('#video-overlay').show();
        $('#btn-start-camera').show();
        $('#btn-capture').hide();
        $('#btn-stop-camera').hide();
        $('#btn-predict-webcam').hide();
        $('#captured-preview').hide();
        $('#video-wrapper').show();
        capturedImageData = null;
    }

    function captureFrame() {
        const video = document.getElementById('webcam-video');
        const canvas = document.getElementById('webcam-canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        // Mirror the capture to match the mirrored video display
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0);
        ctx.setTransform(1, 0, 0, 1, 0, 0);

        capturedImageData = canvas.toDataURL('image/jpeg', 0.9);
        $('#capturedImage').attr('src', capturedImageData);
        $('#video-wrapper').hide();
        $('#captured-preview').show();
        $('#btn-capture').hide();
        $('#btn-predict-webcam').show();
    }

    function retakeWebcam() {
        capturedImageData = null;
        $('#captured-preview').hide();
        $('#video-wrapper').show();
        $('#btn-capture').show();
        $('#btn-predict-webcam').hide();
        hideResults();
    }

    function predictWebcam() {
        if (!capturedImageData) return;
        showLoader();
        $.ajax({
            type: 'POST',
            url: '/predict_webcam',
            contentType: 'application/json',
            data: JSON.stringify({ image: capturedImageData }),
            success: function (data) {
                hideLoader();
                renderResults(data);
            },
            error: function (xhr) {
                hideLoader();
                let msg = 'An error occurred during prediction.';
                try { msg = JSON.parse(xhr.responseText).error || msg; } catch (e) { /* ignore */ }
                alert(msg);
            }
        });
    }

    // ── Loader ──────────────────────────────────────────────

    function showLoader() {
        $('#loader-overlay').fadeIn(200);
    }

    function hideLoader() {
        $('#loader-overlay').fadeOut(200);
    }

    // ── Results Rendering ───────────────────────────────────

    function hideResults() {
        $('#results-section').hide();
    }

    function renderResults(data) {
        // Scroll smoothly to results
        const resultsSection = $('#results-section');
        resultsSection.show();

        // Diagnosis card styling
        const card = $('#diagnosis-card');
        card.removeClass('healthy diseased');
        card.addClass(data.info.status);

        // Icon & label
        $('#diagnosis-icon').text(data.info.icon);
        $('#diagnosis-label').text(data.label === 'Healthy' ? '✅ Healthy Plant' :
                                  data.label === 'Powdery' ? '⚠️ Powdery Mildew Detected' :
                                                              '🔴 Rust Disease Detected');
        $('#diagnosis-desc').text(data.info.description);

        // Confidence bars
        const barsHtml = Object.entries(data.probabilities).map(function ([label, pct]) {
            const barClass = label === 'Healthy' ? 'bar-healthy' :
                             label === 'Powdery' ? 'bar-powdery' : 'bar-rust';
            return `
                <div class="confidence-bar-item">
                    <div class="confidence-bar-label">
                        <span>${label}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-value ${barClass}" style="width: 0%;" data-width="${pct}"></div>
                    </div>
                </div>`;
        }).join('');
        $('#confidence-bars').html(barsHtml);

        // Animate bars
        setTimeout(function () {
            $('.confidence-bar-value').each(function () {
                $(this).css('width', $(this).data('width') + '%');
            });
        }, 100);

        // Severity
        if (data.info.severity && data.info.severity !== 'None') {
            const sevClass = data.info.severity === 'High' ? 'severity-high' : 'severity-moderate';
            $('#severity-badge').attr('class', 'severity-badge ' + sevClass)
                .html('<i class="bi bi-exclamation-triangle-fill"></i> Severity: ' + data.info.severity);
            $('#severity-section').show();
        } else {
            $('#severity-section').hide();
        }

        // Recommendations
        const recsHtml = data.info.recommendations.map(function (rec) {
            const itemsHtml = rec.items.map(item => `<li>${item}</li>`).join('');
            return `
                <div class="recommendation-card">
                    <div class="rec-header">
                        <span class="rec-icon">${rec.icon}</span>
                        <span>${rec.type}</span>
                    </div>
                    <div class="rec-body">
                        <ul>${itemsHtml}</ul>
                    </div>
                </div>`;
        }).join('');
        $('#recommendations-container').html(recsHtml);

        // Scroll to results
        setTimeout(function () {
            $('html, body').animate({
                scrollTop: resultsSection.offset().top - 80
            }, 500);
        }, 200);
    }
});

