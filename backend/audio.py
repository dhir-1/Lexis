from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from dotenv import load_dotenv

from database import log_async

load_dotenv()

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
    "ar": "Arabic",
    "bn": "Bengali",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "gu": "Gujarati",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mr": "Marathi",
    "ne": "Nepali",
    "pa": "Punjabi",
    "pt": "Portuguese",
    "ru": "Russian",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "zh": "Chinese",
}

latest_audio_event: dict[str, object] = {
    "raw_text": "",
    "translated_text": "",
    "detected_language": "en",
    "language_label": "English",
    "timestamp": 0.0,
}
_audio_lock = threading.Lock()


def _set_audio_state(raw: str, translated: str, lang: str) -> None:
    global latest_audio_event
    with _audio_lock:
        latest_audio_event = {
            "raw_text": raw,
            "translated_text": translated,
            "detected_language": lang,
            "language_label": LANGUAGE_LABELS.get(lang.lower(), lang.capitalize()),
            "timestamp": time.monotonic(),
        }


def get_latest_audio_event() -> dict[str, object]:
    with _audio_lock:
        return dict(latest_audio_event)


# ── Groq client setup ────────────────────────────────────────────────────────

groq_key = os.getenv("GROQ_API_KEY", "").strip()
try:
    from groq import Groq
    _groq_client = Groq(api_key=groq_key)
    GROQ_AVAILABLE = bool(groq_key)
    if GROQ_AVAILABLE:
        print("[AUDIO]: Groq client ready (Whisper-large-v3-turbo + LLaMA-3.1-8B-Instant).")
    else:
        print("[AUDIO]: Groq API key missing. Will fall back to local transcription.")
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
    if not cleaned or len(cleaned) <= 1 or cleaned in HALLUCINATION_FILTER:
        return True
    return False


def _language_label(language: str | None) -> str:
    if not language:
        return "Unknown"
    return LANGUAGE_LABELS.get(language.lower(), language.capitalize())


# ── Groq transcription & LLM Translation ────────────────────────────────────

def _transcribe_groq(wav_path: Path) -> tuple[str, str]:
    """Transcribes audio and auto-detects language using Whisper Large V3 Turbo."""
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


def _translate_groq(text: str, from_language: str) -> str:
    """Translates non-English speech into English subtitles using ultra-fast LLaMA-3.1-8B-Instant."""
    if from_language.lower() == "en":
        return text

    source_name = _language_label(from_language)
    response = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": f"You are a real-time subtitle translator. Translate the following {source_name} spoken speech directly into fluent English subtitles. Output only the English translation with no quotes or extra commentary.",
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        max_tokens=150,
        temperature=0.1,
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


# ── Main Audio Worker Loop ───────────────────────────────────────────────────

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
            audio = sd.rec(
                int(CHUNK_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            # Silence rejection: avoid unnecessary API calls
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if rms < RMS_SILENCE_THRESHOLD:
                continue

            wav.write(TEMP_AUDIO_PATH, SAMPLE_RATE, (audio * 32767).astype(np.int16))

            # ── Groq Accelerated Path ────────────────────────────────────────
            if GROQ_AVAILABLE:
                raw_text, detected_language = _transcribe_groq(TEMP_AUDIO_PATH)

                if _is_hallucination(raw_text):
                    continue

                if detected_language.lower() == "en":
                    translated_text = raw_text
                else:
                    translated_text = _translate_groq(raw_text, detected_language)

                if _is_hallucination(translated_text):
                    continue

            # ── Fallback Local Path ──────────────────────────────────────────
            else:
                raw_text, detected_language = _transcribe_fallback(fallback_model, TEMP_AUDIO_PATH)

                if _is_hallucination(raw_text):
                    continue

                if detected_language.lower() == "en":
                    translated_text = raw_text
                else:
                    translated_text = _translate_fallback(
                        fallback_model, TEMP_AUDIO_PATH, detected_language
                    ) or raw_text

                if _is_hallucination(translated_text):
                    continue

            source_label = _language_label(detected_language)
            _set_audio_state(raw_text, translated_text, detected_language)

            if detected_language.lower() == "en":
                print(f"[AUDIO SPEECH (English)]: \"{translated_text}\"")
            else:
                print(f"[AUDIO TRANSLATION ({source_label} -> English)]: \"{raw_text}\" -> \"{translated_text}\"")

            log_async(
                input_type="audio_speech",
                text=translated_text,
                confidence=None,
                detected_language=detected_language,
                raw_text=raw_text,
            )

        except Exception as exc:
            print(f"[AUDIO NOTICE]: {exc}")
            time.sleep(1)

        finally:
            if TEMP_AUDIO_PATH.exists():
                try:
                    TEMP_AUDIO_PATH.unlink()
                except OSError:
                    pass


# ── Thread Entrypoint ────────────────────────────────────────────────────────

def start_audio_thread() -> bool:
    if not ENABLE_AUDIO:
        print("[AUDIO]: Disabled via ENABLE_AUDIO=0.")
        return False

    thread = threading.Thread(target=record_and_translate, daemon=True, name="AudioTranslator")
    thread.start()
    print("[AUDIO]: Multi-Language Speech-to-English Subtitle Engine started.")
    return True