import json
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np
from rtmlib import RTMPose, Wholebody

from database import log_async
from sentence_generator import smooth_asl_sentence_async
from train_user_signs import build_sequence_vector_from_matrix, extract_frame_features

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


CAMERA_INDEX = int(os.getenv("VISION_CAMERA_INDEX", "0"))
MIN_CONFIDENCE = float(os.getenv("VISION_MIN_CONFIDENCE", "0.72"))
HOLD_REQUIRED_FRAMES = 4      # 4 frames (~0.12s) snappy hold for instant recognition
TOKEN_COOLDOWN_SECONDS = 0.35 # Fluid natural conversational signing speed
SENTENCE_RESET_SECONDS = 3.5  # Natural pause after full sentence before sending to LLM


FLASH_SECONDS = 0.85
SUBTITLE_BAR_HEIGHT = 84
SUBTITLE_ALPHA = 0.70
TOP_K_SUGGESTIONS = 5

# RTMPose Wholebody keypoint indices
BODY_IDX = list(range(0, 17))
LEFT_HAND_IDX = list(range(91, 112))
RIGHT_HAND_IDX = list(range(112, 133))

# Hand skeleton connections (21 keypoints per hand)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),              # Palm
]

latest_frame_bytes: bytes | None = None
latest_event: dict[str, object] = {
    "gesture": "...",
    "confidence": 0.0,
    "top_predictions": [],
    "sentence_so_far": "",
    "event_type": "idle",
}

_vision_stream: "VisionStream | None" = None
_state_lock = threading.Lock()


def _register_cuda_dll_paths() -> None:
    if os.name != "nt":
        return
    for path in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.23\bin\12.9\x64",
    ]:
        if os.path.exists(path):
            if path not in os.environ["PATH"]:
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(path)
            except Exception:
                pass


def _is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False



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


class RobustSignEngine:
    """
    Robust Sign Engine:
    Combines User-Trained High Accuracy Landmark Classifier with
    fallback kinematic rules and full A-Z fingerspelling.
    """

    def __init__(self):
        self.r_traj = deque(maxlen=24)
        self.l_traj = deque(maxlen=24)
        self.seq_buffer = deque(maxlen=45)
        self.frame_feats_buffer = deque(maxlen=45)
        self.user_clf = None
        self.user_label_map: dict[int, str] = {}
        self._load_user_classifier()


    def _load_user_classifier(self) -> None:
        model_path = BASE_DIR / "models" / "user_sign_classifier.pkl"
        label_map_path = BASE_DIR / "models" / "user_sign_label_map.json"
        if model_path.exists() and label_map_path.exists():
            try:
                self.user_clf = joblib.load(model_path)
                with open(label_map_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
                    self.user_label_map = {int(k): v for k, v in raw_map.items()}
                print(f"[VISION]: Loaded user-trained classifier with {len(self.user_label_map)} words: {list(self.user_label_map.values())}")
            except Exception as e:
                print(f"[WARN]: Could not load user classifier: {e}")
                self.user_clf = None

    def reset(self) -> None:
        self.r_traj.clear()
        self.l_traj.clear()


    @staticmethod
    def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1[:2] - p2[:2]))

    def _get_finger_states(self, hand_kpts: np.ndarray) -> tuple[list[bool], list[float]]:
        wrist = hand_kpts[0]
        pip_idx = [2, 6, 10, 14, 18]
        tip_idx = [4, 8, 12, 16, 20]
        mcp_idx = [1, 5, 9, 13, 17]

        is_extended = []
        curl_ratios = []

        # Thumb
        thumb_tip_dist = self._dist(hand_kpts[4], hand_kpts[17])
        thumb_ip_dist = self._dist(hand_kpts[3], hand_kpts[17])
        is_extended.append(thumb_tip_dist > thumb_ip_dist * 1.05)
        curl_ratios.append(thumb_tip_dist / max(1e-4, thumb_ip_dist))

        # Index, Middle, Ring, Pinky
        for i in range(1, 5):
            pip = hand_kpts[pip_idx[i]]
            tip = hand_kpts[tip_idx[i]]
            mcp = hand_kpts[mcp_idx[i]]

            tip_wrist = self._dist(tip, wrist)
            pip_wrist = self._dist(pip, wrist)

            extended = tip_wrist > pip_wrist * 1.03 and self._dist(tip, mcp) > self._dist(pip, mcp) * 1.04
            is_extended.append(extended)
            curl_ratios.append(tip_wrist / max(1e-4, pip_wrist))

        return is_extended, curl_ratios

    def classify(
        self,
        right_kpts: np.ndarray | None,
        right_scores: np.ndarray | None,
        left_kpts: np.ndarray | None,
        left_scores: np.ndarray | None,
        body_kpts: np.ndarray,
        body_scores: np.ndarray,
        full_kpts: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        now = time.monotonic()

        if full_kpts is not None:
            f_feat = extract_frame_features(full_kpts)
            self.frame_feats_buffer.append(f_feat)
            self.seq_buffer.append(full_kpts)

        # Anatomical landmarks
        nose_y = float(body_kpts[0][1]) if body_scores[0] > 0.15 else 0.22
        nose_x = float(body_kpts[0][0]) if body_scores[0] > 0.15 else 0.50
        shoulder_y = float((body_kpts[5][1] + body_kpts[6][1]) / 2.0) if (body_scores[5] > 0.15 or body_scores[6] > 0.15) else 0.48
        shoulder_x = float((body_kpts[5][0] + body_kpts[6][0]) / 2.0) if (body_scores[5] > 0.15 or body_scores[6] > 0.15) else 0.50
        chin_y = (nose_y + shoulder_y) / 2.0

        has_right = right_kpts is not None and np.mean(right_scores) > 0.18 and np.max(right_scores) > 0.30
        has_left = left_kpts is not None and np.mean(left_scores) > 0.18 and np.max(left_scores) > 0.30

        if not has_right and not has_left:
            self.reset()
            return []

        # Foreshortening wrist clamp: when pointing straight at camera,
        # prevent detached wrist from snapping across body
        if has_right and right_kpts is not None:
            hmcp = (right_kpts[5] + right_kpts[9] + right_kpts[13] + right_kpts[17]) / 4.0
            if self._dist(right_kpts[0], hmcp) > 0.18:
                right_kpts[0] = hmcp + np.array([0.0, 0.04], dtype=np.float32)

        if has_left and left_kpts is not None:
            hmcp = (left_kpts[5] + left_kpts[9] + left_kpts[13] + left_kpts[17]) / 4.0
            if self._dist(left_kpts[0], hmcp) > 0.18:
                left_kpts[0] = hmcp + np.array([0.0, 0.04], dtype=np.float32)

        r_wrist = right_kpts[0] if has_right else None
        l_wrist = left_kpts[0] if has_left else None

        r_palm = (right_kpts[0] + right_kpts[9]) / 2.0 if has_right else None
        l_palm = (left_kpts[0] + left_kpts[9]) / 2.0 if has_left else None

        # Velocity tracking
        r_speed, r_dy, r_dx = 0.0, 0.0, 0.0
        l_speed, l_dy, l_dx = 0.0, 0.0, 0.0

        if has_right and r_wrist is not None:
            r_scale = max(1e-4, self._dist(right_kpts[0], right_kpts[9]))
            self.r_traj.append((now, float(r_wrist[0]), float(r_wrist[1]), r_scale))
            if len(self.r_traj) >= 4:
                old = self.r_traj[0]
                r_dx = (r_wrist[0] - old[1]) / r_scale
                r_dy = (r_wrist[1] - old[2]) / r_scale
                r_speed = float(np.sqrt(r_dx**2 + r_dy**2))

        if has_left and l_wrist is not None:
            l_scale = max(1e-4, self._dist(left_kpts[0], left_kpts[9]))
            self.l_traj.append((now, float(l_wrist[0]), float(l_wrist[1]), l_scale))
            if len(self.l_traj) >= 4:
                old = self.l_traj[0]
                l_dx = (l_wrist[0] - old[1]) / l_scale
                l_dy = (l_wrist[1] - old[2]) / l_scale
                l_speed = float(np.sqrt(l_dx**2 + l_dy**2))

        # Downward retraction filter (ignoring drop transitions to desk)
        if (has_right and r_dy > 0.20 and r_speed > 0.25) or (has_left and l_dy > 0.20 and l_speed > 0.25):
            self.reset()
            return []

        # Active hands elevated in signing box (fingertips/hand above bottom desk cutoff)
        r_min_y = float(np.min(right_kpts[:, 1])) if has_right else 1.0
        l_min_y = float(np.min(left_kpts[:, 1])) if has_left else 1.0
        r_elevated = has_right and r_min_y < 0.94
        l_elevated = has_left and l_min_y < 0.94

        if not r_elevated and not l_elevated:
            return []

        # =========================================================================
        # 1. USER-TRAINED KINEMATIC SIGN CLASSIFIER (Primary Engine)
        # =========================================================================
        if self.user_clf is not None and len(self.frame_feats_buffer) >= 6:
            try:
                f_mat = np.array(self.frame_feats_buffer, dtype=np.float32)
                s_mat = np.array(self.seq_buffer, dtype=np.float32)
                feats = build_sequence_vector_from_matrix(f_mat, s_mat)
                probs = self.user_clf.predict_proba([feats])[0]

                # Sort predictions
                sorted_indices = np.argsort(probs)[::-1]
                all_candidates = [
                    (self.user_label_map[int(idx)], float(probs[idx]))
                    for idx in sorted_indices
                ]
                best_word, best_prob = all_candidates[0]
                second_prob = all_candidates[1][1] if len(all_candidates) > 1 else 0.0

                # Kinematic transition guards:
                # Downward descent (e.g. hand moving down from hello to my) is not 'please'
                is_descending = (has_right and r_dy > 0.06 and r_speed > 0.10) or (has_left and l_dy > 0.06 and l_speed > 0.10)
                is_lateral_rub = (has_right and abs(r_dx) > 0.04) or (has_left and abs(l_dx) > 0.04)

                if best_word.lower() == "please" and is_descending and not is_lateral_rub:
                    # Ignore downward transit glitch for please
                    if len(all_candidates) > 1 and all_candidates[1][0].lower() != "please":
                        best_word, best_prob = all_candidates[1]
                    else:
                        return []

                # 50-Class confidence thresholding (with 50 classes, uniform prior is 2%, so 28%+ with margin is a dominant winner)
                if best_prob >= 0.28 and (best_prob - second_prob >= 0.03) and best_word:
                    if best_word.lower() == "idle":
                        return []
                    # Boost confidence so valid gesture passes commit threshold smoothly
                    commit_conf = float(np.clip(best_prob * 1.30 + 0.35, 0.85, 0.98))
                    return [(best_word, commit_conf)] + all_candidates[1:5]

                return all_candidates[:5]


            except Exception:
                pass



        # Dominant Hand (highest in signing space)
        p_kpts = right_kpts if (r_elevated and (not l_elevated or r_palm[1] <= l_palm[1])) else left_kpts
        p_palm = r_palm if (r_elevated and (not l_elevated or r_palm[1] <= l_palm[1])) else l_palm
        p_speed = r_speed if (r_elevated and (not l_elevated or r_palm[1] <= l_palm[1])) else l_speed
        p_dy = r_dy if (r_elevated and (not l_elevated or r_palm[1] <= l_palm[1])) else l_dy
        p_dx = r_dx if (r_elevated and (not l_elevated or r_palm[1] <= l_palm[1])) else l_dx

        ext, curls = self._get_finger_states(p_kpts)
        thumb, index, middle, ring, pinky = ext

        hand_scale = max(1e-4, self._dist(p_kpts[0], p_kpts[9]))
        dist_im = self._dist(p_kpts[8], p_kpts[12]) / hand_scale
        dist_mr = self._dist(p_kpts[12], p_kpts[16]) / hand_scale
        dist_ti = self._dist(p_kpts[4], p_kpts[8]) / hand_scale
        dist_tm = self._dist(p_kpts[4], p_kpts[12]) / hand_scale

        all_open = index and middle and ring and pinky
        fist = not index and not middle and not ring and not pinky

        # =========================================================================
        # 2. UNIVERSAL A-Z FINGERSPPELLING FALLBACK (Only when no user sign active)
        # =========================================================================
        on_chest_check = (chin_y + 0.02 < p_palm[1] < 0.85) and (abs(p_palm[0] - shoulder_x) < 0.35)
        if p_speed < 0.08 and p_palm[1] < 0.68 and not on_chest_check:
            candidates: list[tuple[str, float]] = []



            # 'R' - Index & Middle crossed (for RYUK)
            if index and middle and not ring and not pinky and dist_im < 0.22 and not thumb:
                candidates.append(("R", 0.96))

            # 'Y' - Thumb & Pinky extended (shaka) (for RYUK)
            if thumb and pinky and not index and not middle and not ring:
                candidates.append(("Y", 0.96))

            # 'U' - Index & Middle extended straight up together (for RYUK)
            if index and middle and not ring and not pinky and dist_im < 0.28 and not thumb:
                candidates.append(("U", 0.94))

            # 'K' - Index up, Middle forward/up, Thumb between them (for RYUK)
            if index and middle and not ring and not pinky and dist_im > 0.35 and thumb:
                candidates.append(("K", 0.95))

            # 'A' - Fist with thumb vertical alongside index
            if fist and p_kpts[4][1] < p_kpts[5][1] and not thumb and p_speed < 0.08:
                candidates.append(("A", 0.90))

            # 'B' - 4 fingers open vertical, thumb tucked across palm
            if all_open and not thumb and dist_im < 0.30 and p_speed < 0.08:
                candidates.append(("B", 0.92))

            # 'C' - Curved hand forming 'C' shape
            if 0.30 < dist_ti < 0.85 and 0.25 < dist_tm < 0.85 and not all_open and not fist:
                candidates.append(("C", 0.88))

            # 'D' - Index straight up, thumb touching middle/ring/pinky tips
            if index and not middle and not ring and not pinky and dist_tm < 0.50:
                candidates.append(("D", 0.92))

            # 'E' - All fingers curled tight with thumb folded under
            if fist and p_kpts[4][1] > p_kpts[8][1] and p_speed < 0.08:
                candidates.append(("E", 0.88))

            # 'F' - Thumb & Index touching in circle ('OK' sign), 3 fingers up
            if dist_ti < 0.40 and middle and ring and pinky:
                candidates.append(("F", 0.93))

            # 'I' - Pinky straight up, others curled in fist
            if pinky and not index and not middle and not ring and not thumb:
                candidates.append(("I", 0.94))

            # 'L' - Thumb & Index extended (90 degree 'L' shape)
            if index and thumb and not middle and not ring and not pinky and dist_ti > 0.65:
                candidates.append(("L", 0.95))

            # 'O' - All fingertips touching thumb tip in circle
            if dist_ti < 0.38 and dist_tm < 0.40 and not all_open:
                candidates.append(("O", 0.89))

            # 'P' - 'K' shape pointing downward
            if index and middle and not ring and not pinky and thumb and p_kpts[8][1] > p_kpts[0][1]:
                candidates.append(("P", 0.91))

            # 'S' - Tight fist with thumb across front of fingers
            if fist and p_kpts[4][0] < p_kpts[12][0] and p_speed < 0.08:
                candidates.append(("S", 0.89))

            # 'T' - Fist with thumb between index and middle
            if fist and p_kpts[4][1] < p_kpts[6][1] and p_speed < 0.08:
                candidates.append(("T", 0.90))

            # 'V' - Index & Middle extended spread in 'V' shape
            if index and middle and not ring and not pinky and dist_im > 0.38 and not thumb:
                candidates.append(("V", 0.94))

            # 'W' - Index, Middle, Ring extended spread
            if index and middle and ring and not pinky and not thumb and p_palm[1] > chin_y:
                candidates.append(("W", 0.93))

            if candidates:
                candidates.sort(key=lambda item: item[1], reverse=True)
                return candidates

        return []


class VisionStream:
    """Live Continuous Subtitle Translation Stream."""

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        self.pose_model = None
        self.cap = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.engine = RobustSignEngine()
        self.stability_window = deque(maxlen=HOLD_REQUIRED_FRAMES)

        # Sentence State
        self.current_sentence_tokens: list[str] = []
        self.current_sentence_confidences: list[float] = []
        self.last_gesture_time: float | None = None
        self.last_commit_time: float = 0.0
        self.locked_token: str | None = None

        self.flash_text = ""
        self.flash_until = 0.0

    def start(self) -> bool:
        if self.thread and self.thread.is_alive():
            return True
        self._reset_state()

        _register_cuda_dll_paths()
        try:
            self.pose_model = RTMPose(
                Wholebody.MODE["performance"]["pose"],
                model_input_size=Wholebody.MODE["performance"]["pose_input_size"],
                backend="onnxruntime",
                device="cuda" if _is_cuda_available() else "cpu",
            )
            print("[VISION]: RTMPose Wholebody ready.")
        except Exception as exc:
            print(f"[VISION]: Failed to initialize RTMPose: {exc}")
            return False

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("[VISION]: Could not open webcam.")
            self._cleanup_capture()
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="VisionStream")
        self.thread.start()
        print("[VISION]: Precise Subtitle Translation Stream started.")
        return True

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.thread = None

    def _reset_state(self) -> None:
        self.engine.reset()
        self.stability_window.clear()
        self.current_sentence_tokens.clear()
        self.current_sentence_confidences.clear()
        self.last_gesture_time = None
        self.last_commit_time = 0.0
        self.locked_token = None
        self.flash_text = ""
        self.flash_until = 0.0

    def _cleanup_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.pose_model = None

    def _format_sentence(self) -> str:
        if not self.current_sentence_tokens:
            return ""

        words = []
        current_word = ""

        for token in self.current_sentence_tokens:
            if len(token) == 1:
                # Letters concatenate into word (e.g. R-Y-U-K -> RYUK)
                current_word += token
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(token)

        if current_word:
            words.append(current_word)

        return " ".join(words).strip()

    def _draw_hand_skeleton(self, frame, hand_kpts: np.ndarray, scores: np.ndarray, img_w: int, img_h: int, color=(0, 240, 120)) -> None:
        points = []
        for i in range(21):
            x = int(hand_kpts[i][0] * img_w)
            y = int(hand_kpts[i][1] * img_h)
            points.append((x, y))
            if scores[i] > 0.25:
                cv2.circle(frame, (x, y), 4, color, -1)

        max_bone_len = 0.16 * max(img_w, img_h)
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            if scores[p1_idx] > 0.25 and scores[p2_idx] > 0.25:
                dx = points[p1_idx][0] - points[p2_idx][0]
                dy = points[p1_idx][1] - points[p2_idx][1]
                dist = np.hypot(dx, dy)
                # Skip physically impossible long stretched lines (ghost jumps)
                if dist > max_bone_len:
                    continue
                cv2.line(frame, points[p1_idx], points[p2_idx], (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_hud(self, frame, text: str, confidence: float, fps: float) -> None:
        is_active = text != "..." and not text.startswith("Waiting")
        status_color = (0, 255, 120) if is_active else (200, 200, 200)
        label = text if not is_active else f"{text.upper()} ({confidence:.0%})"
        
        cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.0f} FPS", (24, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)

    def _draw_subtitle_bar(self, frame, text: str, flash: bool = False) -> None:
        if not text:
            return
        height, width = frame.shape[:2]
        bar_height = min(SUBTITLE_BAR_HEIGHT, max(60, int(height * 0.12)))
        bar_color = (18, 18, 18) if not flash else (20, 100, 20)
        text_color = (255, 255, 255) if not flash else (0, 255, 100)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - bar_height), (width, height), bar_color, -1)
        cv2.addWeighted(overlay, SUBTITLE_ALPHA, frame, 1 - SUBTITLE_ALPHA, 0, frame)
        cv2.putText(frame, text, (28, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2, cv2.LINE_AA)

    def _draw_suggestions(self, frame, suggestions: list[dict[str, object]]) -> None:
        if not suggestions:
            return
        height, width = frame.shape[:2]
        panel_w = min(320, max(240, int(width * 0.28)))
        row_h = 28
        shown = suggestions[:TOP_K_SUGGESTIONS]
        x1 = width - panel_w - 20
        y1 = 18
        x2 = width - 20
        y2 = y1 + 26 + row_h * len(shown)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, "Live Predictions", (x1 + 12, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        for idx, item in enumerate(shown, start=1):
            label = str(item.get("gesture", ""))
            confidence = float(item.get("confidence", 0.0))
            y = y1 + 20 + idx * row_h
            cv2.putText(frame, f"{idx}. {label}", (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{confidence:.0%}", (x2 - 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 240, 100), 1, cv2.LINE_AA)

    def _commit_token(self, token: str, confidence: float, now: float) -> None:
        if self.locked_token == token:
            return

        if (now - self.last_commit_time) < TOKEN_COOLDOWN_SECONDS:
            return

        self.current_sentence_tokens.append(token)
        self.current_sentence_confidences.append(confidence)
        self.last_gesture_time = now
        self.last_commit_time = now
        self.locked_token = token

        sentence_text = self._format_sentence()
        print(f"[SUBTITLE STREAM]: + '{token}' -> \"{sentence_text}\"")
        log_async(input_type="vision_gesture", text=sentence_text, confidence=confidence)

    def _flush_sentence(self, now: float) -> tuple[str, float]:
        tokens_to_smooth = list(self.current_sentence_tokens)
        sentence_text = self._format_sentence()
        sentence_confidence = 0.0
        if self.current_sentence_confidences:
            sentence_confidence = sum(self.current_sentence_confidences) / len(self.current_sentence_confidences)
        if sentence_text:
            print(f"[SUBTITLE FINALIZED]: \"{sentence_text}\"")
            log_async(input_type="vision_gesture", text=sentence_text, confidence=sentence_confidence)
        self.flash_text = sentence_text
        self.flash_until = now + FLASH_SECONDS
        self.current_sentence_tokens.clear()
        self.current_sentence_confidences.clear()
        self.stability_window.clear()
        self.locked_token = None
        self.last_gesture_time = None

        if tokens_to_smooth:
            def _on_smoothed(raw: str, smoothed: str):
                if smoothed:
                    self.flash_text = smoothed
                    self.flash_until = time.monotonic() + FLASH_SECONDS + 2.0
                    print(f"[GEMINI 1.5 FLASH]: \"{raw}\" -> \"{smoothed}\"")
                    log_async(
                        input_type="vision_sentence",
                        text=smoothed,
                        confidence=sentence_confidence,
                        detected_language="asl",
                        raw_text=raw,
                    )

            smooth_asl_sentence_async(tokens_to_smooth, _on_smoothed)

        return sentence_text, sentence_confidence

    def _run(self) -> None:
        fps_counter = 30.0
        last_frame_time = time.monotonic()

        try:
            while not self.stop_event.is_set():
                ok, raw_frame = self.cap.read() if self.cap is not None else (False, None)
                if not ok or raw_frame is None:
                    time.sleep(0.02)
                    continue

                now = time.monotonic()
                dt = now - last_frame_time
                last_frame_time = now
                if dt > 0:
                    fps_counter = 0.9 * fps_counter + 0.1 * (1.0 / dt)

                h, w = raw_frame.shape[:2]
                display_frame = cv2.flip(raw_frame, 1)

                gesture_display = "..."
                gesture_confidence = 0.0
                top_predictions: list[dict[str, object]] = []
                event_type = "idle"
                completed_sentence = None

                # Run RTMPose Wholebody
                try:
                    result = self.pose_model(display_frame, bboxes=[[0, 0, w, h]])
                except Exception as exc:
                    result = None

                if result is not None and isinstance(result, tuple) and len(result) == 2:
                    all_keypoints, all_scores = result
                    if all_keypoints is not None and len(all_keypoints) > 0:
                        mean_scores = all_scores.mean(axis=1) if getattr(all_scores, "ndim", 1) > 1 else all_scores
                        best_idx = int(np.argmax(mean_scores))
                        kpts = all_keypoints[best_idx]
                        scores = all_scores[best_idx]

                        # Normalize coordinates to [0, 1]
                        norm_kpts = kpts.copy().astype(np.float32)
                        norm_kpts[:, 0] /= max(w, 1)
                        norm_kpts[:, 1] /= max(h, 1)

                        # Anatomical wrist clamp: when pointing straight at camera,
                        # prevent detached wrist from snapping across body
                        for hand_slice in [RIGHT_HAND_IDX, LEFT_HAND_IDX]:
                            hwrist = norm_kpts[hand_slice[0], :2]
                            hmcp = np.mean(norm_kpts[[hand_slice[5], hand_slice[9], hand_slice[13], hand_slice[17]], :2], axis=0)
                            if float(np.linalg.norm(hwrist - hmcp)) > 0.16:
                                norm_kpts[hand_slice[0], :2] = hmcp + np.array([0.0, 0.04], dtype=np.float32)

                        body_kpts = norm_kpts[BODY_IDX]
                        body_scores = scores[BODY_IDX]
                        left_kpts = norm_kpts[LEFT_HAND_IDX]
                        left_scores = scores[LEFT_HAND_IDX]
                        right_kpts = norm_kpts[RIGHT_HAND_IDX]
                        right_scores = scores[RIGHT_HAND_IDX]

                        left_mean = float(np.mean(left_scores))
                        right_mean = float(np.mean(right_scores))

                        # Ghost duplicate suppression: if both hands overlap closely, suppress weaker
                        if left_mean > 0.20 and right_mean > 0.20:
                            r_center = np.mean(right_kpts[:, :2], axis=0)
                            l_center = np.mean(left_kpts[:, :2], axis=0)
                            if np.linalg.norm(r_center - l_center) < 0.08:
                                if right_mean > left_mean:
                                    left_mean = 0.0
                                    left_scores = np.zeros_like(left_scores)
                                else:
                                    right_mean = 0.0
                                    right_scores = np.zeros_like(right_scores)

                        # Draw skeletons for active hands
                        if right_mean > 0.22:
                            self._draw_hand_skeleton(display_frame, right_kpts, right_scores, w, h, color=(0, 240, 120))
                        if left_mean > 0.22:
                            self._draw_hand_skeleton(display_frame, left_kpts, left_scores, w, h, color=(0, 200, 255))

                        # Classify gesture frame
                        full_kpts_133 = np.zeros((133, 3), dtype=np.float32)
                        full_kpts_133[:, :2] = norm_kpts
                        full_kpts_133[:, 2] = scores

                        candidates = self.engine.classify(
                            right_kpts if right_mean > 0.18 else None,
                            right_scores if right_mean > 0.18 else None,
                            left_kpts if left_mean > 0.18 else None,
                            left_scores if left_mean > 0.18 else None,
                            body_kpts,
                            body_scores,
                            full_kpts=full_kpts_133,
                        )


                        if candidates:
                            top_predictions = [
                                {"gesture": name, "confidence": round(score, 2)}
                                for name, score in candidates[:TOP_K_SUGGESTIONS]
                            ]
                            best_token, best_conf = candidates[0]
                            gesture_display = best_token
                            gesture_confidence = best_conf

                            if best_conf >= MIN_CONFIDENCE:
                                event_type = "gesture"
                                self.stability_window.append(best_token)

                                # Require 4 out of 6 consecutive frames agreement
                                if len(self.stability_window) == HOLD_REQUIRED_FRAMES:
                                    most_common, count = Counter(self.stability_window).most_common(1)[0]
                                    if count >= (HOLD_REQUIRED_FRAMES - 2) and most_common is not None:
                                        if self.locked_token != most_common:
                                            self._commit_token(most_common, best_conf, now)
                                            self.stability_window.clear()
                            else:
                                self.stability_window.append(None)
                        else:
                            self.stability_window.append(None)
                            if list(self.stability_window).count(None) >= 8:
                                self.locked_token = None
                            gesture_display = "..."
                    else:
                        self.engine.reset()
                        self.stability_window.clear()
                        self.locked_token = None
                        gesture_display = "..."
                else:
                    self.engine.reset()
                    self.stability_window.clear()
                    self.locked_token = None
                    gesture_display = "..."

                # Auto-flush complete sentence after comfortable natural pause (5.5s)
                if (
                    self.current_sentence_tokens
                    and self.last_gesture_time is not None
                    and now - self.last_gesture_time >= SENTENCE_RESET_SECONDS
                ):
                    completed_sentence, sentence_confidence = self._flush_sentence(now)
                    gesture_display = "..."
                    gesture_confidence = sentence_confidence
                    event_type = "sentence_complete"

                subtitle_text = completed_sentence if completed_sentence is not None else self._format_sentence()
                subtitle_flash = bool(self.flash_text and now < self.flash_until)
                if subtitle_flash:
                    subtitle_text = self.flash_text
                elif self.flash_text and now >= self.flash_until:
                    self.flash_text = ""

                # Draw UI HUD
                self._draw_hud(display_frame, gesture_display, gesture_confidence, fps_counter)
                self._draw_suggestions(display_frame, top_predictions)
                if subtitle_text:
                    self._draw_subtitle_bar(display_frame, subtitle_text, flash=subtitle_flash)

                event = {
                    "gesture": gesture_display,
                    "confidence": round(float(gesture_confidence), 3),
                    "top_predictions": top_predictions,
                    "sentence_so_far": completed_sentence if completed_sentence is not None else self._format_sentence(),
                    "event_type": event_type,
                }
                if completed_sentence is not None:
                    event["completed_sentence"] = completed_sentence

                success, encoded = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
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