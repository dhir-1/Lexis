"""
Lexis Ultra-Fast Multilingual Audio Subtitler
Features:
- Instant Recording (0s countdown delay)
- Auto Voice Activity Detection (stops immediately when you stop speaking)
- Direct Groq Whisper Translation (<250ms)
- Original Language Transcription + Gemini Translation comparison
- Strict Hallucination Shield (filters "The", silence noise, etc.)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TEMP_WAV = BASE_DIR / "temp_test_audio.wav"
SAMPLE_RATE = 16000
CHUNK_FRAMES = 1024  # 64ms chunks for instant VAD response

# Strict silence and hallucination filter
SILENCE_RMS_THRESHOLD = 0.012
POST_SPEECH_SILENCE_SECONDS = 0.7
MAX_RECORD_SECONDS = 6.0

HALLUCINATIONS = {
    "",
    ".",
    "the",
    "the.",
    "you",
    "you.",
    "thank you.",
    "thank you",
    "thanks for watching.",
    "bye.",
    "bye-bye.",
    "subscribe.",
    "so",
    "yeah",
}

groq_key = os.getenv("GROQ_API_KEY", "").strip()
if not groq_key:
    print("[ERROR]: GROQ_API_KEY missing from backend/.env")
    sys.exit(1)

import groq
groq_client = groq.Groq(api_key=groq_key)

from sentence_generator import translate_speech_gemini


def record_speech_with_vad() -> np.ndarray | None:
    """
    Records speech with real-time Voice Activity Detection.
    Starts immediately and stops 0.7s after user finishes speaking.
    """
    print("\n  🎤 [LISTENING] Speak now in Hindi (stops automatically when you pause)...")

    recorded_chunks: list[np.ndarray] = []
    speech_started = False
    speech_start_time = 0.0
    last_sound_time = time.monotonic()
    record_start = time.monotonic()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_FRAMES) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_FRAMES)
            recorded_chunks.append(chunk)

            rms = float(np.sqrt(np.mean(np.square(chunk))))
            now = time.monotonic()

            if rms >= SILENCE_RMS_THRESHOLD:
                if not speech_started:
                    speech_started = True
                    speech_start_time = now
                    print("  🔴 [VOICE DETECTED] Recording speech...", end="\r", flush=True)
                last_sound_time = now
            else:
                # If speech was ongoing, check if silence duration exceeded
                if speech_started and (now - last_sound_time) >= POST_SPEECH_SILENCE_SECONDS:
                    print(f"  ⏹️ [SPEECH ENDED] Captured {(now - speech_start_time):.1f}s of speech.       ")
                    break

            # Hard timeout guard
            if (now - record_start) >= MAX_RECORD_SECONDS:
                if speech_started:
                    print("  ⏹️ [MAX DURATION] Processing captured speech...                 ")
                else:
                    print("  ⚠️ [TIMEOUT] No speech detected.                              ")
                break

    if not recorded_chunks:
        return None

    full_audio = np.concatenate(recorded_chunks, axis=0).flatten()
    total_rms = float(np.sqrt(np.mean(np.square(full_audio))))

    if not speech_started or total_rms < SILENCE_RMS_THRESHOLD:
        print("  ⚠️ Audio was silent or too quiet. Please speak closer to the mic.")
        return None

    return full_audio


def main():
    print("=" * 70)
    print("   🎙️  LEXIS ULTRA-FAST MULTILINGUAL AUDIO TRANSLATOR")
    print("=" * 70)
    print("• Instant-start: No countdown delays")
    print("• Auto-stop VAD: Finishes immediately when you stop speaking")
    print("• Direct Groq Whisper: Instant native English translation")
    print("-" * 70)

    run_once = "--once" in sys.argv

    while True:
        if not run_once:
            try:
                input("\n👉 Press [Enter] and speak in Hindi (or Ctrl+C to quit)... ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                break

        audio_data = record_speech_with_vad()
        if audio_data is None:
            if run_once:
                break
            continue

        # Save audio to temp file
        int_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        wav.write(TEMP_WAV, SAMPLE_RATE, int_data)

        print("\n⚡ Processing with Groq LPUs...")

        # ── 1. Direct Whisper Translation (<200ms) ───────────────────────────
        t0 = time.perf_counter()
        with open(TEMP_WAV, "rb") as f:
            direct_result = groq_client.audio.translations.create(
                file=(TEMP_WAV.name, f.read()),
                model="whisper-large-v3",
                response_format="json",
            )
        t_direct_ms = (time.perf_counter() - t0) * 1000
        direct_english = direct_result.text.strip()

        # ── 2. Whisper Original Transcription (for language & Hindi script) ─
        t1 = time.perf_counter()
        with open(TEMP_WAV, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(TEMP_WAV.name, f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        t_transcribe_ms = (time.perf_counter() - t1) * 1000
        raw_speech = getattr(transcription, "text", "").strip()
        detected_lang = getattr(transcription, "language", "en") or "en"

        # Hallucination filter
        if raw_speech.lower().strip() in HALLUCINATIONS or direct_english.lower().strip() in HALLUCINATIONS:
            print("  ⚠️ Filtered out background silence artifact.")
            if run_once:
                break
            continue

        print("\n" + "─" * 70)
        print(f"  🇮🇳 Original Speech ({detected_lang.upper()}) : \"{raw_speech}\"")
        print(f"  🇬🇧 English Subtitle (Direct) : \"{direct_english}\"")
        print(f"  ⏱️ Direct Whisper Latency    : {t_direct_ms:.1f} ms  ⚡ (ULTRA-FAST)")
        print(f"  ⏱️ Full Transcription Time   : {t_transcribe_ms:.1f} ms")
        print("─" * 70)

        if run_once:
            break


if __name__ == "__main__":
    main()
