from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd

from database import log_async

BASE_DIR = Path(__file__).resolve().parent
TEMP_AUDIO_PATH = BASE_DIR / "temp_audio.wav"
ENABLE_AUDIO = os.getenv("ENABLE_AUDIO", "1").strip().lower() not in {"0", "false", "no", "off"}
SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
RMS_SILENCE_THRESHOLD = 0.02

HALLUCINATION_FILTER = {
    "",
    ".",
    "bye.",
    "bye-bye.",
    "thank you.",
    "thanks for watching.",
    "the",
    "you",
    " ",
}

LANGUAGE_LABELS = {
    "ar": "arabic",
    "bn": "bengali",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "gu": "gujarati",
    "hi": "hindi",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "mr": "marathi",
    "ne": "nepali",
    "pa": "punjabi",
    "pt": "portuguese",
    "ru": "russian",
    "ta": "tamil",
    "te": "telugu",
    "ur": "urdu",
    "zh": "chinese",
}

# ── Groq client setup ────────────────────────────────────────────────────────

try:
    from groq import Groq
    _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    GROQ_AVAILABLE = True
    print("[AUDIO]: Groq client initialized.")
except Exception as exc:
    _groq_client = None
    GROQ_AVAILABLE = False
    print(f"[AUDIO]: Groq unavailable: {exc}. Will fall back to faster-whisper.")

# ── Fallback: faster-whisper ─────────────────────────────────────────────────

def _load_fallback_model():
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("[AUDIO]: Fallback faster-whisper model loaded.")
        return model
    except Exception as exc:
        print(f"[AUDIO]: faster-whisper also unavailable: {exc}")
        return None

# ── Helpers ──────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _is_hallucination(text: str) -> bool:
    cleaned = _clean_text(text).lower()
    if not cleaned:
        return True
    if len(cleaned) <= 1:
        return True
    if cleaned in HALLUCINATION_FILTER:
        return True
    return False


def _language_label(language: str | None) -> str:
    if not language:
        return "unknown"
    return LANGUAGE_LABELS.get(language.lower(), language.lower())


# ── Groq transcription ───────────────────────────────────────────────────────

def _transcribe_groq(wav_path: Path) -> tuple[str, str]:
    """
    Returns (transcript, detected_language).
    Uses verbose_json so language detection is free — no extra API call.
    """
    with open(wav_path, "rb") as f:
        result = _groq_client.audio.transcriptions.create(
            file=(wav_path.name, f.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            language=None,  # auto-detect
        )
    transcript = _clean_text(result.text)
    detected_language = getattr(result, "language", "en") or "en"
    return transcript, detected_language


# ── Groq translation ─────────────────────────────────────────────────────────

def _translate_groq(text: str, from_language: str) -> str:
    """
    Translates to English using LLaMA 8B instant.
    Only called when detected language is not English — saves tokens.
    """
    if from_language == "en":
        return text

    response = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Translate the user's message to English. Reply with the translation only. No explanations, no notes."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        max_tokens=200,
    )
    return _clean_text(response.choices[0].message.content)


# ── Fallback transcription ───────────────────────────────────────────────────

def _transcribe_fallback(model, wav_path: Path) -> tuple[str, str]:
    segments, info = model.transcribe(
        str(wav_path),
        task="transcribe",
        vad_filter=True,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )
    transcript = _clean_text(" ".join(seg.text for seg in segments))
    detected_language = getattr(info, "language", "en") or "en"
    return transcript, detected_language


def _translate_fallback(model, wav_path: Path, detected_language: str) -> str:
    if detected_language == "en":
        return None
    segments, _ = model.transcribe(
        str(wav_path),
        task="translate",
        language=detected_language,
        vad_filter=True,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )
    return _clean_text(" ".join(seg.text for seg in segments))


# ── Main loop ────────────────────────────────────────────────────────────────

def record_and_translate():
    if not ENABLE_AUDIO:
        print("[AUDIO]: Disabled via ENABLE_AUDIO=0.")
        return

    fallback_model = None
    if not GROQ_AVAILABLE:
        fallback_model = _load_fallback_model()
        if fallback_model is None:
            print("[AUDIO]: No transcription backend available. Audio disabled.")
            return

    while True:
        try:
            print(f"[AUDIO]: Recording {CHUNK_SECONDS}s chunk...")
            audio = sd.rec(
                int(CHUNK_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            # Skip silent chunks before any API call
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if rms < RMS_SILENCE_THRESHOLD:
                print("[AUDIO]: Silence detected, skipping.")
                continue

            wav.write(TEMP_AUDIO_PATH, SAMPLE_RATE, (audio * 32767).astype(np.int16))

            # ── Groq path ────────────────────────────────────────────────────
            if GROQ_AVAILABLE:
                raw_text, detected_language = _transcribe_groq(TEMP_AUDIO_PATH)

                if _is_hallucination(raw_text):
                    print("[AUDIO]: Hallucination detected, skipping.")
                    continue

                # English → no translation call at all
                if detected_language == "en":
                    translated_text = raw_text
                else:
                    translated_text = _translate_groq(raw_text, detected_language)

                if _is_hallucination(translated_text):
                    print("[AUDIO]: Hallucination in translation, skipping.")
                    continue

            # ── Fallback path ────────────────────────────────────────────────
            else:
                raw_text, detected_language = _transcribe_fallback(fallback_model, TEMP_AUDIO_PATH)

                if _is_hallucination(raw_text):
                    print("[AUDIO]: Hallucination detected, skipping.")
                    continue

                if detected_language == "en":
                    translated_text = raw_text
                else:
                    translated_text = _translate_fallback(
                        fallback_model, TEMP_AUDIO_PATH, detected_language
                    ) or raw_text

                if _is_hallucination(translated_text):
                    print("[AUDIO]: Hallucination in translation, skipping.")
                    continue

            source_label = _language_label(detected_language)
            print(f"[AUDIO] ({source_label}): {raw_text} -> [en]: {translated_text}")

            log_async(
                input_type="audio_speech",
                text=translated_text,
                confidence=None,
                detected_language=detected_language,
                raw_text=raw_text,
            )

        except Exception as exc:
            print(f"[AUDIO ERROR]: {exc}")
            time.sleep(1)

        finally:
            if TEMP_AUDIO_PATH.exists():
                try:
                    TEMP_AUDIO_PATH.unlink()
                except OSError:
                    pass


# ── Thread entrypoint ────────────────────────────────────────────────────────

def start_audio_thread():
    if not ENABLE_AUDIO:
        print("[AUDIO]: Disabled via ENABLE_AUDIO=0.")
        return False

    thread = threading.Thread(target=record_and_translate, daemon=True)
    thread.start()
    print("[AUDIO]: Background thread started.")
    return True