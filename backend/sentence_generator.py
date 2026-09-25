"""
Lexis ASL Gloss to Natural English Sentence Smoother.
Powered by Google Gemini 2.5 Flash (via google-generativeai).

Converts raw ASL gloss sequences into natural, fluent English sentences
while preserving meaning and adding proper punctuation and grammar.
Includes resilient offline heuristics and in-memory caching to stay within
Google AI Studio free-tier quotas.
"""

from __future__ import annotations

import os
import threading
import time
import warnings
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

# Filter deprecation/future warnings from google.generativeai
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_CANDIDATES = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]

# Single-token fast path mappings (0ms latency, zero API calls)
FAST_PATH_MAP: dict[str, str] = {
    "again": "Again, please.",
    "baby": "A baby.",
    "bathroom": "The bathroom.",
    "book": "A book.",
    "brother": "My brother.",
    "car": "A car.",
    "drink": "I would like a drink.",
    "drive": "Driving.",
    "eat": "Eating food.",
    "family": "My family.",
    "father": "My father.",
    "fine": "I am doing fine.",
    "friend": "A friend.",
    "goodbye": "Goodbye!",
    "happy": "I am happy.",
    "hello": "Hello!",
    "help": "Help me, please!",
    "home": "At home.",
    "house": "A house.",
    "how": "How?",
    "i love you": "I love you.",
    "like": "I like that.",
    "man": "A man.",
    "more": "More, please.",
    "mother": "My mother.",
    "my": "My.",
    "name": "Name.",
    "no": "No.",
    "play": "Playing.",
    "please": "Please.",
    "sad": "I am sad.",
    "school": "At school.",
    "sister": "My sister.",
    "sorry": "I am sorry.",
    "stop": "Stop.",
    "student": "A student.",
    "teacher": "A teacher.",
    "thank you": "Thank you!",
    "time": "What time is it?",
    "want": "I want that.",
    "water": "Some water, please.",
    "what": "What?",
    "when": "When?",
    "where": "Where?",
    "who": "Who?",
    "why": "Why?",
    "woman": "A woman.",
    "work": "Working.",
    "yes": "Yes.",
    "you": "You.",
}

# Multi-word heuristic patterns for instantaneous local translation
COMMON_PHRASE_HEURISTICS: dict[str, str] = {
    "i love you mother": "I love you, Mother.",
    "i love you father": "I love you, Father.",
    "i love you brother": "I love you, Brother.",
    "i love you sister": "I love you, Sister.",
    "i love you friend": "I love you, my friend.",
    "where bathroom": "Where is the bathroom?",
    "where bathroom please": "Where is the bathroom, please?",
    "want water": "I would like some water, please.",
    "want more water": "I want more water, please.",
    "want water please": "I want water, please.",
    "want more food": "I want more food, please.",
    "what time": "What time is it?",
    "how you": "How are you doing?",
    "who you": "Who are you?",
    "why you sad": "Why are you sad?",
    "why you happy": "Why are you happy?",
    "drive car home": "Driving the car home.",
    "drive car school": "Driving the car to school.",
    "student work school": "The student is working at school.",
    "teacher help student": "The teacher is helping the student.",
    "mother father family": "Mother and father make a family.",
    "fine thank you": "I am fine, thank you.",
    "help please": "Help, please!",
}

# In-memory cache for recent smoothed sentences (prevents redundant API calls)
_SMOOTH_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()

# Rate limit cooldown tracker (seconds until next API attempt allowed)
_COOLDOWN_UNTIL: float = 0.0

# Initialized Gemini Model
_model = None
_active_model_name: str = ""
_client_initialized = False


def _get_model():
    """Lazily initialize and return an active Gemini Flash GenerativeModel instance."""
    global _model, _client_initialized, _active_model_name

    if _client_initialized:
        return _model

    api_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    if not api_key:
        print("[GEMINI FLASH]: GEMINI_API_KEY not found in .env. Using heuristic smoothing.")
        _client_initialized = True
        _model = None
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        # Try active models in priority order
        for cand in MODEL_CANDIDATES:
            try:
                candidate_model = genai.GenerativeModel(
                    model_name=cand,
                    system_instruction=(
                        "You are an expert American Sign Language (ASL) to spoken English interpreter. "
                        "Your task is to convert raw ASL gloss tokens or chained words into a single, natural, "
                        "grammatically fluent conversational English sentence.\n\n"
                        "Rules:\n"
                        "1. Add necessary grammatical helper verbs (is, am, are, was), articles (a, an, the), "
                        "and prepositions.\n"
                        "2. Preserve the exact intended meaning of the signer without adding extraneous information.\n"
                        "3. Ensure proper capitalization and ending punctuation (period, question mark, or exclamation mark).\n"
                        "4. Output ONLY the finalized English sentence. Do NOT include quotes, explanations, markdown, or commentary."
                    ),
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 150,
                    },
                )
                _model = candidate_model
                _active_model_name = cand
                print(f"[GEMINI FLASH]: Ready with model '{cand}'.")
                break
            except Exception:
                continue

    except Exception as exc:
        print(f"[GEMINI FLASH]: Initialization failed: {exc}. Falling back to heuristic smoothing.")
        _model = None

    _client_initialized = True
    return _model


def _heuristic_smooth(tokens: list[str]) -> str:
    """Fast, offline rule-based fallback when Gemini API is offline or quota is exceeded."""
    if not tokens:
        return ""

    raw = " ".join(tokens).strip()
    key = raw.lower()

    if key in FAST_PATH_MAP:
        return FAST_PATH_MAP[key]

    if key in COMMON_PHRASE_HEURISTICS:
        return COMMON_PHRASE_HEURISTICS[key]

    # Name introduction pattern: "hello my name [X]"
    if key.startswith("hello my name"):
        name_parts = key.replace("hello my name", "").strip()
        if name_parts:
            # Format words like "ryuk" -> "Ryuk"
            capitalized_name = " ".join(w.capitalize() for w in name_parts.split())
            return f"Hello, my name is {capitalized_name}."
        return "Hello, my name is..."

    # General punctuation & capitalization rules
    words = [w.strip() for w in key.split() if w.strip()]
    if not words:
        return ""

    joined = " ".join(words)
    capitalized = joined[0].upper() + joined[1:] if len(joined) > 1 else joined.upper()

    if not capitalized.endswith((".", "?", "!")):
        if any(words[0].lower() == q for q in ["what", "when", "where", "who", "why", "how"]):
            capitalized += "?"
        elif words[0].lower() in ["help", "stop"]:
            capitalized += "!"
        else:
            capitalized += "."

    return capitalized


def smooth_asl_sentence(tokens: list[str]) -> str:
    """
    Smooths raw ASL tokens into a fluent English sentence.
    
    1. Checks single-token fast path mappings (0ms).
    2. Checks common phrase heuristics (0ms).
    3. Checks in-memory cache (0ms).
    4. Calls Google Gemini Flash if outside rate-limit cooldown.
    5. Falls back seamlessly to rule-based heuristics.
    """
    global _COOLDOWN_UNTIL

    if not tokens:
        return ""

    raw_key = " ".join(tokens).strip().lower()
    if not raw_key:
        return ""

    # 1. Fast-path lookup for single words
    if len(tokens) == 1 and raw_key in FAST_PATH_MAP:
        return FAST_PATH_MAP[raw_key]

    # 2. Check known common phrase heuristics
    if raw_key in COMMON_PHRASE_HEURISTICS:
        return COMMON_PHRASE_HEURISTICS[raw_key]

    # 3. Check in-memory cache
    with _CACHE_LOCK:
        if raw_key in _SMOOTH_CACHE:
            return _SMOOTH_CACHE[raw_key]

    # 4. Attempt Gemini Flash generation if not cooling down
    now = time.monotonic()
    if now >= _COOLDOWN_UNTIL:
        model = _get_model()
        if model is not None:
            try:
                prompt = (
                    f"Translate these American Sign Language (ASL) gloss tokens into a complete, natural English sentence. "
                    f"Preserve all names, questions, and concepts:\nASL: {raw_key}\nEnglish:"
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    cleaned = response.text.strip().strip('"').strip("'")
                    if cleaned:
                        with _CACHE_LOCK:
                            _SMOOTH_CACHE[raw_key] = cleaned
                        return cleaned
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                    _COOLDOWN_UNTIL = now + 45.0  # Cool down for 45 seconds on free-tier limit
                    print("[GEMINI FLASH]: Rate limit reached (5 RPM free tier). Seamlessly using heuristic smoothing.")
                else:
                    print(f"[GEMINI FLASH]: Notice ({exc_str[:60]}...). Using heuristic fallback.")

    # 5. Fallback to heuristic rules
    fallback = _heuristic_smooth(tokens)
    with _CACHE_LOCK:
        _SMOOTH_CACHE[raw_key] = fallback
    return fallback


def smooth_asl_sentence_async(
    tokens: list[str],
    callback: Callable[[str, str], None],
) -> threading.Thread:
    """
    Asynchronously smooths an ASL sentence using Gemini Flash in a background daemon thread.
    
    Args:
        tokens: The list of raw ASL tokens.
        callback: Function invoked upon completion with signature `callback(raw_text, smoothed_text)`.
    """
    raw_text = " ".join(tokens).strip()

    def _worker():
        smoothed = smooth_asl_sentence(tokens)
        try:
            callback(raw_text, smoothed)
        except Exception as exc:
            print(f"[GEMINI FLASH]: Async callback error: {exc}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def translate_speech_gemini(text: str, from_language: str) -> str:
    """
    Translates non-English speech transcripts into fluent English using Gemini Flash.
    """
    global _COOLDOWN_UNTIL

    if not text or from_language.lower() == "en":
        return text

    now = time.monotonic()
    if now < _COOLDOWN_UNTIL:
        return text

    model = _get_model()
    if model is None:
        return text

    try:
        prompt = (
            f"You are a real-time subtitle translator. "
            f"Translate the following speech spoken in {from_language} directly into fluent English subtitles. "
            f"Output ONLY the English translation without quotes or commentary.\n\nSpeech: {text}"
        )
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip().strip('"').strip("'")
    except Exception as exc:
        exc_str = str(exc)
        if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
            _COOLDOWN_UNTIL = now + 45.0
            print("[GEMINI FLASH]: Audio translate rate limit. Skipping LLM translation during cooldown.")
        else:
            print(f"[GEMINI FLASH]: Audio translation notice: {exc_str[:60]}")

    return text
