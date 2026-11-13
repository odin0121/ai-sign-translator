/**
 * AI Sign Language Translator - Next-Gen Live Vision, Audio & XAI Controller
 * Production Real-Time Translation & Telemetry Stream Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const video = document.getElementById('client-webcam');
    const canvas = document.getElementById('output-canvas');
    const ctx = canvas.getContext('2d');
    const mjpegImg = document.getElementById('mjpeg-stream');
    const toggleCamBtn = document.getElementById('toggle-camera-btn');
    const toggleModeBtn = document.getElementById('toggle-mode-btn');
    const startBtnQuick = document.getElementById('start-btn-quick');
    const idlePrompt = document.getElementById('idle-prompt');
    const cheatsheetBtn = document.getElementById('open-cheatsheet-btn');
    const cheatsheetModal = document.getElementById('cheatsheet-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    
    // HUD & Prediction displays
    const hudLatencyPill = document.getElementById('hud-latency-pill');
    const hudLatency = document.getElementById('hud-latency');
    const hudFps = document.getElementById('hud-fps');
    const handStatusChip = document.getElementById('hand-status-chip');
    const activeTokenEl = document.getElementById('active-token');
    const confidenceBadge = document.getElementById('confidence-badge');
    const confidenceBar = document.getElementById('confidence-bar');
    const candidatesContainer = document.getElementById('candidates-container');
    const xaiJointsList = document.getElementById('xai-joints-list');
    
    // Sentence, Audio & TTS
    const sentenceText = document.getElementById('sentence-text');
    const copySentenceBtn = document.getElementById('copy-sentence-btn');
    const addSpaceBtn = document.getElementById('add-space-btn');
    const backspaceBtn = document.getElementById('backspace-btn');
    const clearSentenceBtn = document.getElementById('clear-sentence-btn');
    const speakSentenceBtn = document.getElementById('speak-sentence-btn');
    const targetLangSelect = document.getElementById('target-lang-select');
    const translateSentenceBtn = document.getElementById('translate-sentence-btn');
    const translationPreviewBadge = document.getElementById('translation-preview-badge');
    const autoSpeakChk = document.getElementById('auto-speak-chk');
    const audioChimeChk = document.getElementById('audio-chime-chk');
    const ttsWaveAnim = document.getElementById('tts-wave-anim');
    const modelProfileSelect = document.getElementById('model-profile-select');
    const modelProfileStatus = document.getElementById('model-profile-status');
    const vocabCloud = document.getElementById('vocab-chips-cloud');
    const vocabTitle = document.getElementById('vocab-title');
    let vocabChips = document.querySelectorAll('.vocab-chip');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');

    // -------------------------------------------------------------------------
    // Theme Manager (Dark / Bright Mode)
    // -------------------------------------------------------------------------
    function initTheme() {
        const savedTheme = localStorage.getItem('app-theme') || 'dark';
        applyTheme(savedTheme);

        if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', () => {
                const isCurrentlyDark = document.body.classList.contains('dark-theme');
                const newTheme = isCurrentlyDark ? 'light' : 'dark';
                applyTheme(newTheme);
                localStorage.setItem('app-theme', newTheme);
            });
        }
    }

    function applyTheme(theme) {
        const iconSun = themeToggleBtn ? themeToggleBtn.querySelector('.icon-sun') : null;
        const iconMoon = themeToggleBtn ? themeToggleBtn.querySelector('.icon-moon') : null;
        const themeLabel = themeToggleBtn ? themeToggleBtn.querySelector('.theme-label') : null;

        if (theme === 'light') {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
            if (iconSun) iconSun.style.display = 'none';
            if (iconMoon) iconMoon.style.display = 'inline-block';
            if (themeLabel) themeLabel.innerText = 'Dark';
        } else {
            document.body.classList.remove('light-theme');
            document.body.classList.add('dark-theme');
            if (iconSun) iconSun.style.display = 'inline-block';
            if (iconMoon) iconMoon.style.display = 'none';
            if (themeLabel) themeLabel.innerText = 'Bright';
        }
    }

    initTheme();

    // Runtime state
    let isStreaming = false;
    let streamMode = 'browser'; // 'browser' | 'server'
    let mediaStream = null;
    let processInterval = null;
    let isProcessingFrame = false;
    let isProfileLoading = false;
    let profileRequestId = 0;
    const urlParams = new URLSearchParams(window.location.search);
    const requestedProfile = urlParams.get('model_profile');
    const validProfiles = ['wlasl100', 'isl_emergency', 'asl_alphabet'];
    if (modelProfileSelect && validProfiles.includes(requestedProfile)) {
        modelProfileSelect.value = requestedProfile;
    }
    let selectedModelProfile = modelProfileSelect ? modelProfileSelect.value : 'wlasl100';
    const requestedPracticeToken = urlParams.get('practice_sign')?.trim().toUpperCase() || null;
    let practiceTargetToken = null;
    const practiceNotice = document.getElementById('practice-session-notice');
    const practiceStatus = document.getElementById('practice-session-status');
    
    // Temporal debouncing for continuous sentence formulation
    let lastStableToken = null;
    let tokenStreak = 0;
    const STREAK_THRESHOLD = 3; // Consecutive frames to commit token
    let lastCommittedToken = null;
    let lastSpokenToken = null;
    let handMissStreak = 0;
    const HAND_MISS_GRACE = 4;
    
    // FPS tracking
    let frameTimes = [];

    // Web Audio Synthesizer Context for harmonic chimes
    let audioCtx = null;

    function playSuccessChime() {
        if (!audioChimeChk || !audioChimeChk.checked) return;
        try {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const now = audioCtx.currentTime;
            
            const osc1 = audioCtx.createOscillator();
            const osc2 = audioCtx.createOscillator();
            const gain = audioCtx.createGain();

            osc1.type = 'sine';
            osc2.type = 'triangle';
            osc1.frequency.setValueAtTime(523.25, now); // C5
            osc2.frequency.setValueAtTime(659.25, now + 0.08); // E5

            gain.gain.setValueAtTime(0.08, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

            osc1.connect(gain);
            osc2.connect(gain);
            gain.connect(audioCtx.destination);

            osc1.start(now);
            osc1.stop(now + 0.15);
            osc2.start(now + 0.08);
            osc2.stop(now + 0.35);
        } catch (e) {
            // Audio context not allowed until user gesture
        }
    }

    function getCsrfToken() {
        const input = document.querySelector('[name=csrfmiddlewaretoken]');
        return input ? input.value : '';
    }

    async function loadModelProfile(modelProfile) {
        const requestId = ++profileRequestId;
        isProfileLoading = true;
        practiceTargetToken = null;
        if (practiceNotice) practiceNotice.hidden = true;
        lastStableToken = null;
        tokenStreak = 0;
        lastCommittedToken = null;
        handMissStreak = 0;
        activeTokenEl.innerText = 'LOADING MODEL';
        activeTokenEl.style.color = '#64748b';
        candidatesContainer.innerHTML = '<span class="cand-placeholder">Loading selected sign vocabulary...</span>';
        try {
            const response = await fetch(`/api/vocabulary/?model_profile=${encodeURIComponent(modelProfile)}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (requestId !== profileRequestId) return;

            if (requestedPracticeToken && modelProfile === (requestedProfile || 'wlasl100')
                && data.vocabulary.some(item => item.token === requestedPracticeToken)) {
                practiceTargetToken = requestedPracticeToken;
                if (practiceNotice) practiceNotice.hidden = false;
                if (practiceStatus) {
                    practiceStatus.textContent = `Practice mode: show ${practiceTargetToken} in front of the camera. Predictions will not be added to your message.`;
                }
            }

            vocabCloud.replaceChildren();
            for (const item of data.vocabulary) {
                const chip = document.createElement('button');
                chip.className = 'vocab-chip';
                chip.type = 'button';
                chip.dataset.token = item.token;
                chip.textContent = item.token;
                vocabCloud.appendChild(chip);
            }
            vocabChips = vocabCloud.querySelectorAll('.vocab-chip');
            vocabTitle.textContent = `SUPPORTED VOCABULARY (${data.count} CLASSES)`;
            modelProfileStatus.textContent = data.model_ready
                ? `${data.test_accuracy_percent}% held-out accuracy`
                : 'Model not trained';
            if (!data.model_ready) {
                activeTokenEl.innerText = 'MODEL NOT TRAINED';
                candidatesContainer.innerHTML = '<span class="cand-placeholder">Train this profile before translating.</span>';
            } else {
                activeTokenEl.innerText = 'WAITING';
                candidatesContainer.innerHTML = '<span class="cand-placeholder">Position hand clearly in camera view...</span>';
            }
            confidenceBadge.innerText = '0% Conf';
            confidenceBar.style.width = '0%';
            highlightVocabChip(null);
        } catch (error) {
            if (requestId !== profileRequestId) return;
            console.error('Could not load selected sign model:', error);
            modelProfileStatus.textContent = 'Model unavailable';
            activeTokenEl.innerText = 'MODEL UNAVAILABLE';
        } finally {
            if (requestId === profileRequestId) isProfileLoading = false;
        }
    }

    if (modelProfileSelect) {
        modelProfileSelect.addEventListener('change', () => {
            selectedModelProfile = modelProfileSelect.value;
            loadModelProfile(selectedModelProfile);
        });
    }

    // -------------------------------------------------------------------------
    // Camera Control & Stream Management
    // -------------------------------------------------------------------------
    async function startCamera() {
        if (isStreaming) return;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            let msg = "Camera access is unavailable in this browser context.\n\n";
            if (window.location.hostname === '0.0.0.0') {
                msg += "IMPORTANT: Modern browsers block webcams on 0.0.0.0. Please open http://127.0.0.1:" + (window.location.port || '8000') + "/ or http://localhost:" + (window.location.port || '8000') + "/ in your address bar.";
            } else {
                msg += "Please open http://127.0.0.1:" + (window.location.port || '8000') + "/ and ensure camera permissions are granted.";
            }
            alert(msg);
            return;
        }

        try {
            if (toggleCamBtn) {
                toggleCamBtn.innerText = 'Connecting...';
            }

            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                audio: false
            });

            video.srcObject = mediaStream;
            video.muted = true;
            video.setAttribute('playsinline', '');

            await new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play().then(resolve).catch(resolve);
                };
                setTimeout(resolve, 800);
            });

            try {
                await video.play();
            } catch (e) {
                console.warn("video.play:", e);
            }

            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            canvas.style.display = 'block';
            if (mjpegImg) mjpegImg.style.display = 'none';

            isStreaming = true;
            if (idlePrompt) idlePrompt.style.display = 'none';
            if (toggleCamBtn) {
                toggleCamBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="6" y="4" width="4" height="16"></rect>
                        <rect x="14" y="4" width="4" height="16"></rect>
                    </svg> <span>Stop Camera</span>
                `;
                toggleCamBtn.classList.add('danger');
            }

            if (handStatusChip) {
                handStatusChip.innerText = 'Camera Active';
                handStatusChip.style.color = '#38bdf8';
            }

            // Launch prediction loop (~20 FPS)
            if (processInterval) clearInterval(processInterval);
            processInterval = setInterval(sendFrameForPrediction, 50);

        } catch (err) {
            console.error('Camera access error:', err);
            let userMsg = "Camera access error:\n\n";
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                userMsg += "Camera permission was denied. Please click the camera/lock icon in your browser URL address bar and allow camera access, then retry.";
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                userMsg += "No camera found on this device. Please connect a webcam.";
            } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
                userMsg += "Camera is in use by another application (e.g., Zoom, FaceTime, Teams). Please close other camera apps and retry.";
            } else {
                userMsg += (err.message || String(err));
            }
            alert(userMsg);
            stopCamera();
        }
    }

    function stopCamera() {
        if (processInterval) {
            clearInterval(processInterval);
            processInterval = null;
        }

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => {
                try { track.stop(); } catch (e) {}
            });
            mediaStream = null;
        }

        if (video) {
            video.srcObject = null;
        }

        if (streamMode === 'server' && mjpegImg) {
            mjpegImg.src = '';
            mjpegImg.style.display = 'none';
        }

        isStreaming = false;
        isProcessingFrame = false;

        if (idlePrompt) idlePrompt.style.display = 'flex';
        if (toggleCamBtn) {
            toggleCamBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M23 7l-7 5 7 5V7z"></path>
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                </svg> <span>Start Camera</span>
            `;
            toggleCamBtn.classList.remove('danger');
        }
        if (hudLatency) hudLatency.querySelector('.chip-val').innerText = '-- ms';
        if (hudFps) hudFps.querySelector('.chip-val').innerText = '--';
        if (handStatusChip) {
            handStatusChip.innerText = 'Offline';
            handStatusChip.style.color = '#94a3b8';
        }
    }

    function switchMode(newMode) {
        const wasStreaming = isStreaming;
        if (wasStreaming) stopCamera();
        streamMode = newMode;
        if (toggleModeBtn) {
            toggleModeBtn.innerText = streamMode === 'browser' ? 'Webcam Mode' : 'Server Stream';
        }
        if (wasStreaming) startCamera();
    }

    // -------------------------------------------------------------------------
    // Frame Inference & XAI Visualization
    // -------------------------------------------------------------------------
    async function sendFrameForPrediction() {
        if (!isStreaming || isProcessingFrame || isProfileLoading || video.readyState !== 4) return;
        isProcessingFrame = true;

        const t0 = performance.now();
        const frameModelProfile = selectedModelProfile;

        // Capture frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameDataUrl = canvas.toDataURL('image/jpeg', 0.68);

        try {
            const response = await fetch('/api/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                },
                body: JSON.stringify({
                    image: frameDataUrl,
                    enable_tts: autoSpeakChk.checked,
                    model_profile: frameModelProfile
                })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (frameModelProfile !== selectedModelProfile) return;

            // Render annotated thermal skeleton on canvas
            if (data.annotated_image) {
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = data.annotated_image;
            }

            const elapsed = data.latency_ms || (performance.now() - t0);
            updateTelemetry(elapsed, data.has_hand, data.hand_count);
            updatePredictionUI(data);

        } catch (err) {
            console.warn('Inference error:', err);
        } finally {
            isProcessingFrame = false;
        }
    }

    function updateTelemetry(latencyMs, hasHand, handCount = 0) {
        if (hudLatency) hudLatency.querySelector('.chip-val').innerText = `${latencyMs.toFixed(1)} ms`;
        if (hudLatencyPill) {
            hudLatencyPill.innerText = `Latency: ${latencyMs.toFixed(1)} ms`;
            hudLatencyPill.style.color = latencyMs <= 50.0 ? '#34d399' : '#fbbf24';
        }

        if (handStatusChip) {
            if (hasHand) {
                handStatusChip.innerText = selectedModelProfile === 'wlasl100'
                    ? `Word tracking (${handCount}/2 hands)`
                    : `Hand Tracking (21 Joints)`;
                handStatusChip.style.color = '#34d399';
            } else {
                handStatusChip.innerText = 'No Hand Detected';
                handStatusChip.style.color = '#94a3b8';
            }
        }

        // FPS calculation
        const now = performance.now();
        frameTimes.push(now);
        frameTimes = frameTimes.filter(t => now - t <= 1000);
        const fps = frameTimes.length;
        if (hudFps) hudFps.querySelector('.chip-val').innerText = `${fps}`;
    }

    function updatePredictionUI(data) {
        if (!data || !data.has_hand) {
            handMissStreak++;
            activeTokenEl.innerText = 'WAITING';
            activeTokenEl.style.color = '#64748b';
            delete activeTokenEl.dataset.hasPrevious;
            confidenceBadge.innerText = '0% Conf';
            confidenceBadge.style.color = '#64748b';
            confidenceBar.style.width = '0%';
            candidatesContainer.innerHTML = '<span class="cand-placeholder">Position hand clearly in camera view...</span>';
            xaiJointsList.innerHTML = '<div class="joint-item empty"><span>Awaiting kinematic hand detection...</span></div>';
            highlightVocabChip(null);
            if (handMissStreak > HAND_MISS_GRACE) {
                lastStableToken = null;
                tokenStreak = 0;
                lastCommittedToken = null;
            }
            if (practiceTargetToken) {
                if (practiceStatus) practiceStatus.textContent = `Show ${practiceTargetToken} clearly in the camera frame to practice.`;
            }
            return;
        }

        handMissStreak = 0;
        const token = data.prediction || 'UNKNOWN';
        const confPercent = Math.round((data.confidence || 0) * 100);

        activeTokenEl.dataset.hasPrevious = 'true';
        activeTokenEl.innerText = token;
        activeTokenEl.style.color = confPercent >= 65 ? '#38bdf8' : '#f1f5f9';
        confidenceBadge.innerText = `${confPercent}% Confidence`;
        confidenceBar.style.width = `${Math.min(100, Math.max(10, confPercent))}%`;

        if (confPercent >= 65) {
            confidenceBadge.style.color = '#34d399';
            confidenceBadge.style.borderColor = 'rgba(52, 211, 153, 0.4)';
        } else if (confPercent >= 40) {
            confidenceBadge.style.color = '#38bdf8';
            confidenceBadge.style.borderColor = 'rgba(56, 189, 248, 0.4)';
        } else {
            confidenceBadge.style.color = '#fbbf24';
            confidenceBadge.style.borderColor = 'rgba(251, 191, 36, 0.4)';
        }

        highlightVocabChip(token);

        // Candidates display
        if (data.candidates && data.candidates.length > 0) {
            candidatesContainer.innerHTML = data.candidates.map(([candToken, prob]) => {
                const p = Math.round(prob * 100);
                return `<span class="cand-chip"><strong>${candToken}</strong>: ${p}%</span>`;
            }).join('');
        }

        // Top XAI Saliency Hotspots
        if (data.top_joints && data.top_joints.length > 0) {
            xaiJointsList.innerHTML = data.top_joints.map(joint => {
                const pct = Math.round(joint.score * 100);
                return `
                    <div class="joint-item">
                        <span class="joint-name">${joint.joint_name}</span>
                        <div style="display: flex; align-items: center;">
                            <span class="joint-score">${(joint.score).toFixed(2)}</span>
                            <div class="joint-score-bar">
                                <div class="joint-score-fill" style="width: ${pct}%;"></div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Sentence formulation with debounce & chime (50% threshold for real-world reliability)
        const minimumConfidence = Number.isFinite(data.min_confidence)
            ? data.min_confidence
            : 0.5;
        if (practiceTargetToken) {
            if (token === practiceTargetToken && (data.confidence || 0) >= minimumConfidence) {
                if (practiceStatus) practiceStatus.textContent = `Recognized ${practiceTargetToken}. Compare your hand shape with the example.`;
            } else if ((data.confidence || 0) >= minimumConfidence) {
                if (practiceStatus) practiceStatus.textContent = `The model currently reads ${token}. Review the example for ${practiceTargetToken} and try again.`;
            } else {
                if (practiceStatus) practiceStatus.textContent = `Keep practicing ${practiceTargetToken}; the current reading is uncertain.`;
            }
            lastStableToken = null;
            tokenStreak = 0;
            lastCommittedToken = null;
            return;
        }
        if ((data.confidence || 0) >= minimumConfidence) {
            if (token === lastStableToken) {
                tokenStreak++;
                if (tokenStreak >= STREAK_THRESHOLD && token !== lastCommittedToken) {
                    appendTokenToSentence(token);
                    lastCommittedToken = token;
                    playSuccessChime();

                    if (autoSpeakChk.checked && token !== lastSpokenToken) {
                        speakText(token);
                        lastSpokenToken = token;
                    }
                }
            } else {
                lastStableToken = token;
                tokenStreak = 1;
            }
        } else {
            tokenStreak = 0;
            lastCommittedToken = null;
        }
    }

    function highlightVocabChip(token) {
        vocabChips.forEach(chip => {
            if (token && chip.dataset.token === token) {
                chip.classList.add('highlight');
            } else {
                chip.classList.remove('highlight');
            }
        });
    }

    function appendTokenToSentence(token) {
        const current = sentenceText.innerText.trim();
        const separator = selectedModelProfile === 'asl_alphabet' ? '' : ' ';
        sentenceText.innerText = (current ? current + separator : '') + token;
    }

    // -------------------------------------------------------------------------
    // Text-to-Speech (TTS) Actions
    // -------------------------------------------------------------------------
    async function speakText(text) {
        text = text || sentenceText.innerText.trim();
        if (!text) return;

        // Show wave animation
        if (ttsWaveAnim) ttsWaveAnim.style.display = 'flex';

        try {
            await fetch('/api/speak/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                },
                body: JSON.stringify({ text: text })
            });
        } catch (e) {
            // Client-side Web Speech fallback
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0;
                window.speechSynthesis.speak(utterance);
            }
        } finally {
            setTimeout(() => {
                if (ttsWaveAnim) ttsWaveAnim.style.display = 'none';
            }, 1200);
        }
    }

    // -------------------------------------------------------------------------
    // Event Listeners
    // -------------------------------------------------------------------------
    if (toggleCamBtn) {
        toggleCamBtn.addEventListener('click', () => {
            if (isStreaming) stopCamera();
            else startCamera();
        });
    }

    if (startBtnQuick) {
        startBtnQuick.addEventListener('click', () => {
            if (!isStreaming) startCamera();
        });
    }

    if (toggleModeBtn) {
        toggleModeBtn.addEventListener('click', () => {
            const nextMode = streamMode === 'browser' ? 'server' : 'browser';
            switchMode(nextMode);
        });
    }

    if (addSpaceBtn) {
        addSpaceBtn.addEventListener('click', () => {
            sentenceText.innerText += ' ';
        });
    }

    if (backspaceBtn) {
        backspaceBtn.addEventListener('click', () => {
            sentenceText.innerText = sentenceText.innerText.slice(0, -1);
        });
    }

    if (clearSentenceBtn) {
        clearSentenceBtn.addEventListener('click', () => {
            sentenceText.innerText = '';
            lastStableToken = null;
            tokenStreak = 0;
            lastCommittedToken = null;
            if (translationPreviewBadge) {
                translationPreviewBadge.innerText = '';
                translationPreviewBadge.style.display = 'none';
            }
        });
    }

    if (speakSentenceBtn) {
        speakSentenceBtn.addEventListener('click', () => {
            speakText(sentenceText.innerText.trim());
        });
    }

    // Sign-to-Text Grammar Polishing & Multilingual Translation
    async function translateAndPolishSentence() {
        const rawText = sentenceText.innerText.trim();
        if (!rawText) return;

        const targetLang = targetLangSelect ? targetLangSelect.value : 'English';

        if (translateSentenceBtn) {
            translateSentenceBtn.innerText = 'Translating...';
            translateSentenceBtn.disabled = true;
        }

        try {
            const resp = await fetch('/api/translate/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                },
                body: JSON.stringify({
                    text: rawText,
                    target_language: targetLang
                })
            });

            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            if (data.status === 'success') {
                if (targetLang === 'English') {
                    sentenceText.innerText = data.polished_sentence || rawText;
                    if (translationPreviewBadge) {
                        translationPreviewBadge.innerText = 'Polished ASL';
                        translationPreviewBadge.style.display = 'inline-block';
                    }
                    speakText(data.polished_sentence || rawText);
                } else {
                    const translated = data.translated_sentence || data.polished_sentence;
                    if (translationPreviewBadge) {
                        translationPreviewBadge.innerText = `${targetLang}: ${translated}`;
                        translationPreviewBadge.style.display = 'inline-block';
                    }
                    speakText(translated);
                }
            }
        } catch (e) {
            console.error('Translation error:', e);
        } finally {
            if (translateSentenceBtn) {
                translateSentenceBtn.innerText = '✨ Translate';
                translateSentenceBtn.disabled = false;
            }
        }
    }

    if (translateSentenceBtn) {
        translateSentenceBtn.addEventListener('click', translateAndPolishSentence);
    }

    // Copy to clipboard
    if (copySentenceBtn) {
        copySentenceBtn.addEventListener('click', () => {
            const text = sentenceText.innerText.trim();
            if (!text) return;
            navigator.clipboard.writeText(text).then(() => {
                copySentenceBtn.style.color = '#34d399';
                setTimeout(() => { copySentenceBtn.style.color = ''; }, 1000);
            });
        });
    }

    // Modal Cheatsheet
    if (cheatsheetBtn && cheatsheetModal) {
        cheatsheetBtn.addEventListener('click', () => {
            cheatsheetModal.style.display = 'flex';
        });
    }

    if (closeModalBtn && cheatsheetModal) {
        closeModalBtn.addEventListener('click', () => {
            cheatsheetModal.style.display = 'none';
        });
    }

    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && cheatsheetModal) {
            cheatsheetModal.style.display = 'none';
        }
    });

    // Vocab chip interactive click
    if (vocabCloud) {
        vocabCloud.addEventListener('click', event => {
            const chip = event.target.closest('.vocab-chip');
            if (!chip) return;
            const token = chip.dataset.token;
            appendTokenToSentence(token);
            playSuccessChime();
            speakText(token);
        });
    }

    loadModelProfile(selectedModelProfile);
});
