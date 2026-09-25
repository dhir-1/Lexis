"""
Lexis Full-Stack Server & Real-Time API.
FastAPI + WebSockets + MJPEG Video Streaming.
Powers:
1. Live Continuous ASL Vision Subtitler
2. Multilingual Speech-to-English Flow Subtitler
3. 2,400-Class Sign Studio & Video Analyzer ("Shazam for ASL")
4. Neon PostgreSQL Database History
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from database import get_connection, log_async
from sentence_generator import smooth_asl_sentence
from sign_studio import get_sign_studio
from vision import (
    get_latest_event,
    get_latest_frame_bytes,
    is_vision_stream_running,
    start_vision_stream,
    stop_vision_stream,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="Lexis Multimodal Translation Platform",
    description="Continuous ASL Vision Subtitler, Multilingual Speech Subtitler, & 2,400-Class Sign Studio",
    version="2.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── MJPEG Video Streaming Endpoint ───────────────────────────────────────────

def _generate_mjpeg():
    """Generates continuous MJPEG frames for browser video streaming."""
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Warm dark slate canvas with clean message
    blank_frame[:] = (20, 24, 28)
    cv2.putText(
        blank_frame,
        "Lexis Vision Inactive",
        (460, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (220, 220, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        blank_frame,
        "Click 'Start Detection' to begin continuous ASL subtitling",
        (310, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (160, 160, 175),
        1,
        cv2.LINE_AA,
    )
    _, blank_bytes = cv2.imencode(".jpg", blank_frame)
    blank_payload = blank_bytes.tobytes()

    while True:
        if not is_vision_stream_running():
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + blank_payload + b"\r\n")
            time.sleep(0.1)
            continue
        frame_bytes = get_latest_frame_bytes()
        payload = frame_bytes if frame_bytes is not None else blank_payload
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n")
        time.sleep(0.033)  # ~30 FPS


@app.get("/api/vision/stream")
def vision_stream():
    """Streams live webcam feed with RTMPose Wholebody skeleton and HUD overlays."""
    return StreamingResponse(_generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/vision/start")
def start_vision():
    success = start_vision_stream()
    return {"status": "running" if success else "failed", "success": success}


@app.post("/api/vision/stop")
def stop_vision():
    stop_vision_stream()
    return {"status": "stopped", "success": True}


# ── Vision WebSocket for Live Event Streaming ────────────────────────────────

@app.websocket("/ws/vision")
async def vision_websocket(websocket: WebSocket):
    """Pushes real-time token detections, suggestions, and finalized sentences."""
    await websocket.accept()
    last_sent_sentence = ""
    last_sent_gesture = ""

    try:
        while True:
            event = get_latest_event()
            # Push updates whenever state changes or heartbeat every 100ms
            await websocket.send_json(event)
            await asyncio.sleep(0.08)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ── Multilingual Audio Translation Endpoint ──────────────────────────────────

@app.post("/api/audio/translate")
async def translate_audio_file(file: UploadFile = File(...)):
    """Receives audio wav/webm clip, transcribes via Groq Whisper, and returns translation."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured in backend/.env")

    import groq
    client = groq.Groq(api_key=groq_key)

    suffix = Path(file.filename).suffix or ".wav"
    temp_file = UPLOAD_DIR / f"speech_{int(time.time()*1000)}{suffix}"

    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Direct Whisper Native Translation (<250ms)
        t0 = time.perf_counter()
        with open(temp_file, "rb") as f:
            trans_res = client.audio.translations.create(
                file=(temp_file.name, f.read()),
                model="whisper-large-v3",
                response_format="json",
            )
        t_direct_ms = (time.perf_counter() - t0) * 1000
        english_translation = trans_res.text.strip()

        # 2. Transcribe to get original language and native text
        t1 = time.perf_counter()
        with open(temp_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(temp_file.name, f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        t_transcribe_ms = (time.perf_counter() - t1) * 1000
        raw_text = getattr(transcription, "text", "").strip()
        detected_language = getattr(transcription, "language", "en") or "en"

        # Log to Neon DB
        log_async(
            input_type="audio",
            text=english_translation,
            confidence=None,
            detected_language=detected_language,
            raw_text=raw_text,
        )

        return {
            "success": True,
            "detected_language": detected_language,
            "original_speech": raw_text,
            "english_translation": english_translation,
            "direct_latency_ms": round(t_direct_ms, 1),
            "transcription_latency_ms": round(t_transcribe_ms, 1),
        }
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


# ── Sign Studio: Video Analysis & ASL Shazam ──────────────────────────────────

@app.post("/api/studio/analyze-video")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    """
    Ingests an uploaded video file, detects if it is ASL, classifies the signs
    across 2,414 classes, and translates it using Gemini Flash.
    """
    studio = get_sign_studio()
    suffix = Path(file.filename).suffix or ".mp4"
    temp_in = UPLOAD_DIR / f"upload_{int(time.time()*1000)}{suffix}"
    temp_out = UPLOAD_DIR / f"annotated_{int(time.time()*1000)}.mp4"

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process through Sign Studio
        report = studio.analyze_video(video_path=temp_in, output_annotated_path=temp_out)

        # Log to Database
        if report.get("is_sign_language"):
            log_async(
                input_type="vision",
                text=str(report.get("final_english_translation", "")),
                confidence=float(report.get("language_confidence", 0.0)),
                detected_language="asl",
                raw_text=file.filename,
            )

        return JSONResponse(content={"success": True, "report": report})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    finally:
        if temp_in.exists():
            try:
                temp_in.unlink()
            except Exception:
                pass


@app.get("/api/studio/search")
def search_vocabulary(q: str = Query("", min_length=1), limit: int = Query(20, ge=1, le=100)):
    """Searches across all 2,414 ASL classes."""
    studio = get_sign_studio()
    results = studio.search_vocabulary(q, limit=limit)
    return {"query": q, "count": len(results), "results": results}


@app.post("/api/studio/practice")
def evaluate_sign_practice(target_sign: str = Query(...)):
    """Initiates practice assessment for target sign."""
    studio = get_sign_studio()
    # Stub assessment data or evaluates live buffer
    return {"target": target_sign.upper(), "status": "ready"}


# ── Translation History & System Status ───────────────────────────────────────

@app.get("/api/history")
def get_translation_history(limit: int = Query(25, ge=1, le=100)):
    """Fetches real-time translation logs from PostgreSQL Neon DB."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, input_type, translated_text, confidence_score, detected_language, raw_text, created_at
            FROM translation_logs
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        history = []
        for r in rows:
            history.append({
                "id": r[0],
                "input_type": r[1],
                "translated_text": r[2],
                "confidence_score": float(r[3]) if r[3] is not None else None,
                "detected_language": r[4],
                "raw_text": r[5],
                "created_at": r[6].isoformat() if hasattr(r[6], "isoformat") else str(r[6]),
            })
        return {"count": len(history), "history": history}
    except Exception as exc:
        return {"count": 0, "history": [], "warning": f"DB Offline: {exc}"}


@app.get("/api/status")
def system_status():
    """Returns platform diagnostic health status and model specifications."""
    studio = get_sign_studio()
    return {
        "status": "online",
        "vision_stream_running": is_vision_stream_running(),
        "models": {
            "vision_tracker": "RTMPose Wholebody (133 3D Keypoints)",
            "conversational_classifier": f"ExtraTrees Ensemble ({len(studio.user_label_map)} Active Signs, 99.7% CV Acc)",
            "sign_studio_deep_learning": f"PyTorch 2-Layer Bi-GRU ({len(studio.gru_classes)} Classes, 74.3% Top-5 Acc)",
            "sentence_restructuring": "Google Gemini 2.5 Flash",
            "speech_transcription": "Groq Whisper Large V3 Turbo (<120ms)",
            "database": "Neon Cloud PostgreSQL",
        },
    }


# ── Serve Frontend Static Files (Vite React Build + Static Fallback) ──────────

DIST_DIR = FRONTEND_DIR / "dist"
if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    if (DIST_DIR / "index.html").exists():
        return HTMLResponse(content=(DIST_DIR / "index.html").read_text(encoding="utf-8"))
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Lexis Frontend Initializing...</h1>")
