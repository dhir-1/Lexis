from __future__ import annotations

import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp

from database import log_async
from gesture_features import normalize_landmarks

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv("VISION_MODEL_PATH", "").strip()
    or (BASE_DIR / "models" / "gesture_classifier.pkl")
)
CAMERA_INDEX = int(os.getenv("VISION_CAMERA_INDEX", "0"))
MIN_CONFIDENCE = 0.70
STABILITY_WINDOW = 12
STABLE_VOTES = 9
LOG_COOLDOWN_FRAMES = 20
SENTENCE_RESET_SECONDS = 4.0
FLASH_SECONDS = 0.65
SUBTITLE_BAR_HEIGHT = 84
SUBTITLE_ALPHA = 0.65

# Letter-by-letter timing constants
LETTER_CONFIRM_SECONDS = 1.5   # hold same letter this long to confirm
WORD_BOUNDARY_SECONDS = 2.0    # pause this long to add a space

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

latest_frame_bytes: bytes | None = None
latest_event: dict[str, object] = {
    "gesture": "...",
    "confidence": 0.0,
    "sentence_so_far": "",
    "event_type": "idle",
}

_vision_stream: VisionStream | None = None
_state_lock = threading.Lock()


def _set_shared_state(frame_bytes: bytes | None = None, event: dict[str, object] | None = None) -> None:
    global latest_frame_bytes, latest_event
    with _state_lock:
        if frame_bytes is not None:
            latest_frame_bytes = frame_bytes
        if event is not None:
            latest_event = dict(event)


def get_latest_frame_bytes() -> bytes | None:
    with _state_lock:
        return latest_frame_bytes


def get_latest_event() -> dict[str, object]:
    with _state_lock:
        return dict(latest_event)


class VisionStream:
    """Own the camera, gesture classifier, and subtitle-style overlay."""

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        self.clf = None
        self.hands = None
        self.cap = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        # Debounce buffers
        self.prediction_window = deque(maxlen=STABILITY_WINDOW)
        self.confidence_window = deque(maxlen=STABILITY_WINDOW)
        self.cooldown_frames = 0
        self.idle_frames = 0
        self.last_logged_prediction: str | None = None

        # Sentence builder — word mode (old model: Hello, Goodbye etc)
        self.current_sentence: list[str] = []
        self.current_sentence_confidences: list[float] = []
        self.last_gesture_time: float | None = None
        self.flash_text = ""
        self.flash_until = 0.0

        # Letter mode state (new ASL model: A-Z)
        self.current_word: list[str] = []          # letters being spelled
        self.current_word_display: str = ""         # "H-E-L-L-O"
        self.last_letter_time: float | None = None
        self.last_confirmed_letter: str | None = None
        self.letter_hold_start: float | None = None
        self.pending_letter: str | None = None

    def start(self) -> bool:
        if self.thread and self.thread.is_alive():
            return True

        self.prediction_window.clear()
        self.confidence_window.clear()
        self.cooldown_frames = 0
        self.idle_frames = 0
        self.last_logged_prediction = None
        self.current_sentence.clear()
        self.current_sentence_confidences.clear()
        self.last_gesture_time = None
        self.flash_text = ""
        self.flash_until = 0.0
        self.current_word.clear()
        self.current_word_display = ""
        self.last_letter_time = None
        self.last_confirmed_letter = None
        self.letter_hold_start = None
        self.pending_letter = None

        if not MODEL_PATH.exists():
            print(f"[VISION]: Missing model file: {MODEL_PATH}")
            return False

        try:
            self.clf = joblib.load(MODEL_PATH)
        except Exception as exc:
            print(f"[VISION]: Could not load classifier: {exc}")
            return False

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self._cleanup_capture()
            return False

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="VisionStream")
        self.thread.start()
        print("[VISION]: Background stream started.")
        return True

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.thread = None

    def _cleanup_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
            self.hands = None

    # ── Mode detection ────────────────────────────────────────────────────────

    def _is_letter_mode(self, label: str) -> bool:
        """Single character = letter mode (A-Z). Full word = word mode (Hello etc)."""
        return len(label) == 1

    # ── Sentence text helpers ─────────────────────────────────────────────────

    def _current_sentence_text(self) -> str:
        """Full sentence built from committed words."""
        return " ".join(self.current_sentence).strip()

    def _spelling_display(self) -> str:
        """Current word being spelled as H-E-L-L-O."""
        if not self.current_word:
            return ""
        return "-".join(self.current_word)

    def _full_display_text(self) -> str:
        """
        Combines the full sentence + currently spelling word for subtitle bar.
        Example: 'HELLO WORLD  |  Spelling: H-E-L'
        """
        sentence = self._current_sentence_text()
        spelling = self._spelling_display()
        if sentence and spelling:
            return f"{sentence}  |  {spelling}"
        if spelling:
            return f"Spelling: {spelling}"
        return sentence

    # ── Draw helpers ──────────────────────────────────────────────────────────

    def _draw_status_label(self, frame, text: str, confidence: float) -> None:
        if text == "...":
            label = "..."
            color = (180, 180, 180)
        else:
            label = f"{text} ({confidence:.0%})"
            color = (0, 255, 0)
        cv2.putText(frame, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def _draw_subtitle_bar(self, frame, text: str, flash: bool = False) -> None:
        if not text:
            return
        height, width = frame.shape[:2]
        bar_height = min(SUBTITLE_BAR_HEIGHT, max(64, int(height * 0.14)))
        bar_color = (18, 18, 18) if not flash else (20, 90, 20)
        text_color = (255, 255, 255) if not flash else (0, 255, 0)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - bar_height), (width, height), bar_color, -1)
        cv2.addWeighted(overlay, SUBTITLE_ALPHA, frame, 1 - SUBTITLE_ALPHA, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.82
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        max_width = width - 48
        if text_size[0] > max_width:
            font_scale = max(0.55, font_scale * max_width / max(text_size[0], 1))
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        x = 24
        y = height - max(18, (bar_height - text_size[1]) // 2)
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def _draw_letter_progress(self, frame, pending_letter: str, hold_start: float, now: float) -> None:
        """Draw a hold progress bar above the subtitle when a letter is being held."""
        if not pending_letter or hold_start is None:
            return
        progress = min((now - hold_start) / LETTER_CONFIRM_SECONDS, 1.0)
        height, width = frame.shape[:2]
        bar_y = height - SUBTITLE_BAR_HEIGHT - 12
        bar_w = int(width * progress)
        cv2.rectangle(frame, (0, bar_y), (bar_w, bar_y + 6), (99, 102, 241), -1)

    # ── Stable prediction ─────────────────────────────────────────────────────

    def _resolve_stable_prediction(self) -> tuple[str | None, float]:
        votes = [label for label in self.prediction_window if label is not None]
        if not votes:
            return None, 0.0
        stable_prediction, vote_count = Counter(votes).most_common(1)[0]
        if vote_count < STABLE_VOTES:
            return None, 0.0
        stable_confidences = [
            c for l, c in zip(self.prediction_window, self.confidence_window)
            if l == stable_prediction
        ]
        if not stable_confidences:
            return None, 0.0
        return stable_prediction, sum(stable_confidences) / len(stable_confidences)

    # ── Commit logic ──────────────────────────────────────────────────────────

    def _should_commit(self, prediction: str, confidence: float) -> bool:
        return (
            confidence >= MIN_CONFIDENCE
            and self.cooldown_frames == 0
            and prediction != self.last_logged_prediction
        )

    def _commit_letter(self, letter: str, confidence: float, now: float) -> None:
        """Confirm a single letter into the current word being spelled."""
        self.current_word.append(letter)
        self.last_confirmed_letter = letter
        self.last_letter_time = now
        self.letter_hold_start = None
        self.pending_letter = None
        self.cooldown_frames = LOG_COOLDOWN_FRAMES
        self.idle_frames = 0
        print(f"[VISION]: Letter confirmed → {letter} ({confidence:.0%}) | Word: {self._spelling_display()}")

    def _commit_word(self, now: float) -> None:
        """Finish the current word and push it into the sentence."""
        if not self.current_word:
            return
        word = "".join(self.current_word)
        self.current_sentence.append(word)
        self.current_sentence_confidences.append(MIN_CONFIDENCE)
        self.last_gesture_time = now
        print(f"[VISION]: Word committed → {word}")
        log_async(
            input_type="vision_gesture",
            text=word,
            confidence=MIN_CONFIDENCE,
        )
        self.current_word.clear()
        self.last_confirmed_letter = None
        self.last_letter_time = None

    def _commit_gesture(self, prediction: str, confidence: float, now: float) -> None:
        """Word mode commit — for full-word labels like Hello, Goodbye."""
        self.current_sentence.append(prediction)
        self.current_sentence_confidences.append(confidence)
        self.last_logged_prediction = prediction
        self.last_gesture_time = now
        self.cooldown_frames = LOG_COOLDOWN_FRAMES
        self.idle_frames = 0
        print(f"[VISION]: {prediction} ({confidence:.0%})")
        log_async(
            input_type="vision_gesture",
            text=prediction,
            confidence=confidence,
        )

    # ── Flush sentence ────────────────────────────────────────────────────────

    def _flush_sentence(self, now: float) -> tuple[str, float]:
        # Commit any in-progress word first
        if self.current_word:
            self._commit_word(now)

        sentence_text = self._current_sentence_text()
        sentence_confidence = 0.0
        if self.current_sentence_confidences:
            sentence_confidence = sum(self.current_sentence_confidences) / len(self.current_sentence_confidences)

        print(f"[SENTENCE]: {sentence_text}")
        log_async(
            input_type="vision_gesture",
            text=sentence_text,
            confidence=sentence_confidence,
        )

        self.flash_text = sentence_text
        self.flash_until = now + FLASH_SECONDS
        self.current_sentence.clear()
        self.current_sentence_confidences.clear()
        self.prediction_window.clear()
        self.confidence_window.clear()
        self.last_logged_prediction = None
        self.last_gesture_time = None
        self.idle_frames = 0
        self.current_word.clear()
        self.last_confirmed_letter = None
        self.last_letter_time = None
        self.pending_letter = None
        self.letter_hold_start = None

        return sentence_text, sentence_confidence

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                if self.cooldown_frames > 0:
                    self.cooldown_frames -= 1

                ret, frame = self.cap.read() if self.cap is not None else (False, None)
                if not ret or frame is None:
                    print("Error: Failed to read frame.")
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame) if self.hands is not None else None

                gesture_display = "..."
                gesture_confidence = 0.0
                event_type = "idle"
                completed_sentence = None
                now = time.monotonic()

                if results and results.multi_hand_landmarks:
                    self.idle_frames = 0
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    for landmark in hand_landmarks.landmark:
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

                    landmarks_flat = normalize_landmarks(hand_landmarks.landmark)
                    prediction_proba = self.clf.predict_proba([landmarks_flat])[0]
                    confidence = float(prediction_proba.max())
                    prediction = str(self.clf.classes_[prediction_proba.argmax()])

                    if confidence < MIN_CONFIDENCE:
                        self.prediction_window.append(None)
                        self.confidence_window.append(confidence)
                        self.idle_frames += 1
                        gesture_display = "..."
                        gesture_confidence = confidence
                    else:
                        self.prediction_window.append(prediction)
                        self.confidence_window.append(confidence)
                        gesture_display = prediction
                        gesture_confidence = confidence
                        event_type = "gesture"

                        stable_prediction, stable_confidence = self._resolve_stable_prediction()
                        if stable_prediction is not None:
                            gesture_display = stable_prediction
                            gesture_confidence = stable_confidence

                            # ── Letter mode (A-Z single chars) ────────────────
                            if self._is_letter_mode(stable_prediction):
                                if stable_prediction != self.pending_letter:
                                    # New letter — check word boundary pause first
                                    if (
                                        self.current_word
                                        and self.last_letter_time is not None
                                        and now - self.last_letter_time >= WORD_BOUNDARY_SECONDS
                                    ):
                                        self._commit_word(now)

                                    self.pending_letter = stable_prediction
                                    self.letter_hold_start = now
                                else:
                                    # Same letter held — check if held long enough
                                    if (
                                        self.letter_hold_start is not None
                                        and now - self.letter_hold_start >= LETTER_CONFIRM_SECONDS
                                        and stable_prediction != self.last_confirmed_letter
                                        and self.cooldown_frames == 0
                                    ):
                                        self._commit_letter(stable_prediction, stable_confidence, now)

                            # ── Word mode (Hello, Goodbye etc) ────────────────
                            else:
                                if self._should_commit(stable_prediction, stable_confidence):
                                    self._commit_gesture(stable_prediction, stable_confidence, now)

                else:
                    self.prediction_window.append(None)
                    self.confidence_window.append(0.0)
                    self.idle_frames += 1

                    # Hand lifted — word boundary check
                    if (
                        self.current_word
                        and self.last_letter_time is not None
                        and now - self.last_letter_time >= WORD_BOUNDARY_SECONDS
                    ):
                        self._commit_word(now)

                    # Reset pending letter tracking when hand is gone
                    self.pending_letter = None
                    self.letter_hold_start = None

                if self.idle_frames >= STABILITY_WINDOW:
                    self.prediction_window.clear()
                    self.confidence_window.clear()
                    self.idle_frames = 0

                # Sentence reset timer
                if (
                    self.current_sentence
                    and self.last_gesture_time is not None
                    and now - self.last_gesture_time >= SENTENCE_RESET_SECONDS
                ):
                    completed_sentence, sentence_confidence = self._flush_sentence(now)
                    gesture_display = "..."
                    gesture_confidence = sentence_confidence
                    event_type = "sentence_complete"

                # Draw hold progress bar for current pending letter
                if self.pending_letter and self.letter_hold_start:
                    self._draw_letter_progress(frame, self.pending_letter, self.letter_hold_start, now)

                subtitle_text = self._full_display_text()
                subtitle_flash = False
                if self.flash_text and now < self.flash_until:
                    subtitle_text = self.flash_text
                    subtitle_flash = True
                elif self.flash_text and now >= self.flash_until:
                    self.flash_text = ""

                self._draw_status_label(frame, gesture_display, gesture_confidence)
                if subtitle_text:
                    self._draw_subtitle_bar(frame, subtitle_text, flash=subtitle_flash)

                sentence_for_event = completed_sentence if completed_sentence is not None else self._current_sentence_text()
                event = {
                    "gesture": gesture_display,
                    "confidence": round(float(gesture_confidence), 3),
                    "sentence_so_far": sentence_for_event,
                    "event_type": event_type,
                }
                if completed_sentence is not None:
                    event["completed_sentence"] = completed_sentence

                success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if success:
                    _set_shared_state(encoded.tobytes(), event)

        finally:
            self._cleanup_capture()


def start_vision_stream() -> bool:
    global _vision_stream
    if _vision_stream is None:
        _vision_stream = VisionStream()
    return _vision_stream.start()


def stop_vision_stream() -> None:
    if _vision_stream is not None:
        _vision_stream.stop()


def is_vision_stream_running() -> bool:
    return _vision_stream is not None and _vision_stream.thread is not None and _vision_stream.thread.is_alive()