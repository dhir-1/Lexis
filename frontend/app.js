/**
 * Lexis — Multimodal Translation Frontend Engine
 * Surrealism & Wispr Flow Interactive Client
 */

// ── 50 Active Conversational ASL Signs ──────────────────────────────────────
const CORE_50_SIGNS = [
  "again", "baby", "bathroom", "book", "brother", "car", "drink", "drive",
  "eat", "family", "father", "fine", "friend", "goodbye", "happy", "hello",
  "help", "home", "house", "how", "i love you", "like", "man", "more",
  "mother", "my", "name", "no", "play", "please", "sad", "school",
  "sister", "sorry", "stop", "student", "teacher", "thank you", "time", "want",
  "water", "what", "when", "where", "who", "why", "woman", "work",
  "yes", "you"
];

// Language Flag Emojis
const LANG_FLAGS = {
  "hi": "🇮🇳", "hindi": "🇮🇳",
  "en": "🇬🇧", "english": "🇬🇧",
  "es": "🇪🇸", "spanish": "🇪🇸",
  "fr": "🇫🇷", "french": "🇫🇷",
  "de": "🇩🇪", "german": "🇩🇪",
  "ja": "🇯🇵", "japanese": "🇯🇵",
  "zh": "🇨🇳", "chinese": "🇨🇳",
  "ar": "🇸🇦", "arabic": "🇸🇦",
  "ru": "🇷🇺", "russian": "🇷🇺",
  "gu": "🇮🇳", "gujarati": "🇮🇳",
  "mr": "🇮🇳", "marathi": "🇮🇳",
  "pa": "🇮🇳", "punjabi": "🇮🇳",
};

// State
let currentTab = "vision";
let visionWs = null;
let isCameraActive = true;
let isRecordingAudio = false;
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let animationFrameId = null;

// ── Document Ready ───────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initVisionWebSocket();
  initSidebarChips();
  initAudioRecorder();
  initVideoDropzone();
  initDictionarySearch();
  initHistoryView();
  fetchSystemStatus();
});

// ── 1. Wispr Flow Navigation Tabs ───────────────────────────────────────────

function initTabs() {
  const tabs = document.querySelectorAll(".dock-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-tab");
      switchTab(target);
    });
  });
}

function switchTab(tabId) {
  currentTab = tabId;
  document.querySelectorAll(".dock-tab").forEach(t => {
    t.classList.toggle("active", t.getAttribute("data-tab") === tabId);
  });

  document.querySelectorAll(".view-panel").forEach(p => {
    p.classList.remove("active");
  });

  const activePanel = document.getElementById(`view${capitalize(tabId)}`);
  if (activePanel) {
    activePanel.classList.add("active");
  }

  if (tabId === "history") {
    fetchHistory();
  }
}

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// ── 2. Live Continuous ASL Vision Stream & WebSocket ────────────────────────

function initVisionWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/vision`;

  try {
    visionWs = new WebSocket(wsUrl);

    visionWs.onopen = () => {
      console.log("[VISION WS]: Connected to live subtitle telemetry.");
      document.getElementById("statusText").textContent = "30 FPS • RTMPose Online";
    };

    visionWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleVisionTelemetry(data);
      } catch (err) {
        console.warn("[VISION WS ERROR]:", err);
      }
    };

    visionWs.onclose = () => {
      document.getElementById("statusText").textContent = "Reconnecting stream...";
      setTimeout(initVisionWebSocket, 2000);
    };
  } catch (exc) {
    console.error("[WS EXCEPTION]:", exc);
  }

  // Camera Toggle
  const btnToggle = document.getElementById("btnToggleCamera");
  const mjpegFeed = document.getElementById("mjpegFeed");
  btnToggle.addEventListener("click", () => {
    isCameraActive = !isCameraActive;
    if (isCameraActive) {
      mjpegFeed.src = "/api/vision/stream";
      document.getElementById("camIcon").textContent = "⏸️";
    } else {
      mjpegFeed.src = "";
      document.getElementById("camIcon").textContent = "▶️";
    }
  });
}

function handleVisionTelemetry(data) {
  const subtitlePill = document.getElementById("subtitlePill");
  const subtitleText = document.getElementById("liveSubtitleText");
  const badgeSource = document.getElementById("subtitleTypeBadge");
  const badgeGemini = document.getElementById("geminiBadge");

  // Finalized vs Live Accumulating
  if (data.completed_sentence) {
    subtitleText.textContent = `"${data.completed_sentence}"`;
    badgeSource.textContent = "FINALIZED";
    badgeSource.style.color = "var(--accent-emerald)";
    badgeGemini.style.display = "inline-flex";
    highlightActiveWord(null);
  } else if (data.sentence_so_far) {
    subtitleText.textContent = data.sentence_so_far;
    badgeSource.textContent = "LIVE SIGNING";
    badgeSource.style.color = "var(--accent-cyan)";
    badgeGemini.style.display = "none";
  }

  // Active word highlight
  if (data.gesture && data.gesture !== "...") {
    highlightActiveWord(data.gesture.toLowerCase());
  }

  // Render Top Predictions Bars
  if (data.top_predictions && data.top_predictions.length > 0) {
    renderPredictions(data.top_predictions);
  }
}

function renderPredictions(predictions) {
  const container = document.getElementById("predictionsList");
  container.innerHTML = predictions.map(p => {
    const pct = Math.round(p.confidence * 100);
    return `
      <div class="prediction-row">
        <div class="prediction-info">
          <span>${escapeHtml(p.gesture.toUpperCase())}</span>
          <span style="color: var(--accent-cyan);">${pct}%</span>
        </div>
        <div class="prediction-bar-track">
          <div class="prediction-bar-fill" style="width: ${pct}%;"></div>
        </div>
      </div>
    `;
  }).join("");
}

function initSidebarChips() {
  const container = document.getElementById("quickVocabChips");
  container.innerHTML = CORE_50_SIGNS.map(sign => `
    <span class="vocab-chip" data-sign="${sign}" id="chip-${sign.replace(/\s+/g, '_')}">${sign}</span>
  `).join("");
}

function highlightActiveWord(signName) {
  document.querySelectorAll(".vocab-chip.highlight").forEach(c => c.classList.remove("highlight"));
  if (!signName) return;

  const cleanId = `chip-${signName.replace(/\s+/g, '_')}`;
  const chip = document.getElementById(cleanId);
  if (chip) {
    chip.classList.add("highlight");
    chip.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// ── 3. Multilingual Voice Flow (Audio Translator) ───────────────────────────

function initAudioRecorder() {
  const btnMic = document.getElementById("btnMicRecord");
  const micLabel = document.getElementById("micStateLabel");

  btnMic.addEventListener("click", async () => {
    if (!isRecordingAudio) {
      await startAudioRecording();
    } else {
      stopAudioRecording();
    }
  });
}

async function startAudioRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    analyser.fftSize = 256;

    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      await sendAudioForTranslation(audioBlob);
      stream.getTracks().forEach(track => track.stop());
      if (audioContext) audioContext.close();
      cancelAnimationFrame(animationFrameId);
      clearWaveform();
    };

    mediaRecorder.start();
    isRecordingAudio = true;
    document.getElementById("btnMicRecord").classList.add("recording");
    document.getElementById("micStateLabel").textContent = "Listening... Tap to Translate";
    drawWaveform();
  } catch (err) {
    alert("Microphone permission denied or unavailable: " + err.message);
  }
}

function stopAudioRecording() {
  if (mediaRecorder && isRecordingAudio) {
    mediaRecorder.stop();
    isRecordingAudio = false;
    document.getElementById("btnMicRecord").classList.remove("recording");
    document.getElementById("micStateLabel").textContent = "Translating with Groq...";
  }
}

async function sendAudioForTranslation(blob) {
  const formData = new FormData();
  formData.append("file", blob, "speech.wav");

  try {
    const t0 = performance.now();
    const res = await fetch("/api/audio/translate", {
      method: "POST",
      body: formData,
    });
    const result = await res.json();
    const dt = Math.round(performance.now() - t0);

    if (result.success) {
      const flag = LANG_FLAGS[result.detected_language.toLowerCase()] || "🌐";
      document.getElementById("detectedFlag").textContent = flag;
      document.getElementById("detectedLangTitle").textContent = `${result.detected_language.toUpperCase()} (Detected)`;
      document.getElementById("textOriginalSpeech").textContent = `"${result.original_speech}"`;
      document.getElementById("textTranslatedSpeech").textContent = `"${result.english_translation}"`;
      document.getElementById("audioLatencyPill").textContent = `Groq Latency: ${result.direct_latency_ms || dt} ms ⚡`;
    } else {
      document.getElementById("textTranslatedSpeech").textContent = "Translation failed or no speech detected.";
    }
  } catch (err) {
    console.error("[AUDIO TRANSLATE ERROR]:", err);
    document.getElementById("textTranslatedSpeech").textContent = "Server error processing audio.";
  } finally {
    document.getElementById("micStateLabel").textContent = "Tap to Speak";
  }
}

function drawWaveform() {
  const canvas = document.getElementById("waveformCanvas");
  if (!canvas || !analyser) return;

  const ctx = canvas.getContext("2d");
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function render() {
    animationFrameId = requestAnimationFrame(render);
    analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = "rgba(6, 7, 11, 0.4)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2.5;
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, "#8b5cf6");
    gradient.addColorStop(0.5, "#06b6d4");
    gradient.addColorStop(1, "#ec4899");
    ctx.strokeStyle = gradient;

    ctx.beginPath();
    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }
  render();
}

function clearWaveform() {
  const canvas = document.getElementById("waveformCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ── 4. Sign Studio ("Shazam for ASL" & Video File Analyzer) ────────────────

function initVideoDropzone() {
  const dropzone = document.getElementById("videoDropzone");
  const fileInput = document.getElementById("fileInputVideo");
  const btnBrowse = document.getElementById("btnBrowseVideo");

  btnBrowse.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", (e) => {
    if (e.target !== btnBrowse) fileInput.click();
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      handleVideoUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleVideoUpload(fileInput.files[0]);
    }
  });
}

async function handleVideoUpload(file) {
  const progressContainer = document.getElementById("analysisProgressContainer");
  const progressBar = document.getElementById("analysisProgressBar");
  const progressLabel = document.getElementById("analysisProgressLabel");
  const resultsBody = document.getElementById("studioResultsBody");
  const langTag = document.getElementById("studioLangTag");

  progressContainer.style.display = "block";
  progressBar.style.width = "20%";
  progressLabel.textContent = `Uploading '${file.name}'...`;
  langTag.textContent = "Analyzing Video...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    progressBar.style.width = "50%";
    progressLabel.textContent = "Running RTMPose Wholebody & 2,414-Class Bi-GRU Model...";

    const res = await fetch("/api/studio/analyze-video", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    progressBar.style.width = "100%";
    progressLabel.textContent = "Analysis Complete!";

    if (data.success && data.report) {
      renderStudioReport(data.report);
    } else {
      resultsBody.innerHTML = `<div class="empty-hint">Error: ${escapeHtml(data.error || "Failed to analyze video.")}</div>`;
    }
  } catch (err) {
    resultsBody.innerHTML = `<div class="empty-hint">Upload error: ${escapeHtml(err.message)}</div>`;
  } finally {
    setTimeout(() => {
      progressContainer.style.display = "none";
    }, 1500);
  }
}

function renderStudioReport(report) {
  const resultsBody = document.getElementById("studioResultsBody");
  const langTag = document.getElementById("studioLangTag");

  langTag.textContent = report.detected_language;

  const wlaslRows = (report.top_predictions_2400_class || []).map(item => `
    <div class="sign-match-pill">
      <span>${escapeHtml(item.sign)}</span>
      <span class="match-pct">${item.percentage}</span>
    </div>
  `).join("") || '<div class="empty-hint">No matches</div>';

  const userRows = (report.top_conversational_50_class || []).map(item => `
    <div class="sign-match-pill">
      <span>${escapeHtml(item.sign)}</span>
      <span class="match-pct">${item.percentage}</span>
    </div>
  `).join("") || '<div class="empty-hint">No matches</div>';

  resultsBody.innerHTML = `
    <div class="result-banner-box">
      <div class="result-lang-title">Language Identified</div>
      <div class="result-lang-name">${escapeHtml(report.detected_language)}</div>
      <div style="font-size: 0.80rem; color: var(--text-dim); margin-top: 4px;">
        Confidence: <strong>${Math.round(report.language_confidence * 100)}%</strong> • Hand Presence: <strong>${Math.round(report.hand_presence_ratio * 100)}%</strong> • ${report.duration_seconds}s (${report.total_frames} frames)
      </div>
    </div>

    <div class="studio-grid-deck">
      <div class="mini-deck-card">
        <div class="deck-card-title">2,414 WLASL Sign Matches</div>
        ${wlaslRows}
      </div>
      <div class="mini-deck-card">
        <div class="deck-card-title">Core 50 Conversational</div>
        ${userRows}
      </div>
    </div>

    <div class="translation-reveal-box">
      <div class="translation-reveal-label">✨ Gemini 2.5 Flash English Subtitle</div>
      <div class="translation-reveal-sentence">"${escapeHtml(report.final_english_translation)}"</div>
    </div>
  `;
}

// ── 5. ASL Sign Dictionary & Search ─────────────────────────────────────────

function initDictionarySearch() {
  const searchInput = document.getElementById("inputSignSearch");
  let debounceTimeout = null;

  // Initial load
  loadDictionary("");

  searchInput.addEventListener("input", (e) => {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
      loadDictionary(e.target.value.trim());
    }, 250);
  });
}

async function loadDictionary(query) {
  const grid = document.getElementById("dictionaryGrid");
  const countBadge = document.getElementById("searchCountBadge");

  try {
    const res = await fetch(`/api/studio/search?q=${encodeURIComponent(query || "a")}&limit=36`);
    const data = await res.json();

    if (data.results && data.results.length > 0) {
      countBadge.textContent = `${data.results.length} Matches Found`;
      grid.innerHTML = data.results.map(item => `
        <div class="dict-card ${item.is_core_50 ? "core-50" : ""}">
          <div class="dict-name">${escapeHtml(item.sign)}</div>
          <div class="dict-badge">${escapeHtml(item.dataset)}</div>
        </div>
      `).join("");
    } else {
      countBadge.textContent = "0 Matches";
      grid.innerHTML = `<div class="empty-hint large" style="grid-column: 1/-1;">No signs found matching '${escapeHtml(query)}'</div>`;
    }
  } catch (err) {
    console.error("[SEARCH ERROR]:", err);
  }
}

// ── 6. Neon Cloud PostgreSQL History ────────────────────────────────────────

function initHistoryView() {
  document.getElementById("btnRefreshHistory").addEventListener("click", fetchHistory);
}

async function fetchHistory() {
  const tbody = document.getElementById("historyTableBody");
  tbody.innerHTML = '<tr><td colspan="5" class="loading-td">Fetching logs from Neon DB...</td></tr>';

  try {
    const res = await fetch("/api/history?limit=25");
    const data = await res.json();

    if (data.history && data.history.length > 0) {
      tbody.innerHTML = data.history.map(item => {
        const domainClass = item.input_type === "vision_gesture" || item.input_type === "vision_sentence" 
          ? "tag-vision" 
          : item.input_type === "audio_speech" 
          ? "tag-audio" 
          : "tag-studio";

        const formattedTime = new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        return `
          <tr>
            <td style="font-family: var(--font-mono); color: var(--text-dim);">${formattedTime}</td>
            <td><span class="tag-domain ${domainClass}">${escapeHtml(item.input_type)}</span></td>
            <td style="color: var(--text-dim); max-width: 200px; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(item.raw_text || "—")}</td>
            <td style="font-weight: 600; color: #fff;">${escapeHtml(item.translated_text)}</td>
            <td style="font-family: var(--font-mono); color: var(--accent-cyan);">${item.confidence_score ? Math.round(item.confidence_score * 100) + "%" : "—"}</td>
          </tr>
        `;
      }).join("");
    } else {
      tbody.innerHTML = '<tr><td colspan="5" class="empty-hint">No translation history recorded yet in database.</td></tr>';
    }
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-hint">Database offline or error: ${escapeHtml(err.message)}</td></tr>`;
  }
}

async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    console.log("[SYSTEM STATUS]:", data);
  } catch (err) {
    console.warn("[STATUS CHECK]:", err);
  }
}

function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
