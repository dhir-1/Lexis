"""
Lexis Video Sign Language Analyzer & Studio.
Analyzes uploaded video clips:
- Identifies sign language (American Sign Language - ASL)
- Extracts RTMPose Wholebody 133 anatomical landmarks
- Evaluates 1,000-class WLASL Bi-GRU model + 50-class conversational ExtraTrees ensemble
- Restructures detected signs into fluent English via Gemini 2.5 Flash
- Generates an annotated output video with skeleton overlays and subtitle banners
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from rtmlib import RTMPose, Wholebody

from sentence_generator import smooth_asl_sentence
from train_user_signs import extract_sequence_features

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
GRU_WEIGHTS_PATH = MODELS_DIR / "gesture_gru.pt"
GRU_CLASSES_PATH = MODELS_DIR / "gru_classes.json"
USER_CLF_PATH = MODELS_DIR / "user_sign_classifier.pkl"
USER_LABEL_MAP_PATH = MODELS_DIR / "user_sign_label_map.json"

# Keypoint indices
BODY_IDX = list(range(0, 17))
LEFT_HAND_IDX = list(range(91, 112))
RIGHT_HAND_IDX = list(range(112, 133))

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),              # Palm
]


# ── PyTorch 1,000-Class Bidirectional GRU Model ───────────────────────────────

class GestureGRU(nn.Module):
    def __init__(self, input_size: int = 118, hidden_size: int = 256, num_layers: int = 2, num_classes: int = 1000, dropout: float = 0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        proj = self.input_proj(x_flat).view(B, T, -1)
        out, _ = self.gru(proj)
        logits = self.fc(out[:, -1])
        return logits


class VideoSignAnalyzer:
    """End-to-end video file processor and sign language classifier."""

    def __init__(self):
        self.pose_model = None
        self.gru_model = None
        self.gru_classes: list[str] = []
        self.user_clf = None
        self.user_label_map: dict[str, str] = {}
        self._load_models()

    def _load_models(self):
        # 1. Load RTMPose Wholebody
        try:
            self.pose_model = RTMPose(
                Wholebody.MODE["performance"]["pose"],
                model_input_size=Wholebody.MODE["performance"]["pose_input_size"],
                backend="onnxruntime",
                device="cpu",
            )
            print("[ANALYZER]: RTMPose Wholebody ready.")
        except Exception as exc:
            print(f"[ANALYZER ERROR]: RTMPose init failed: {exc}")

        # 2. Load 50-Class Conversational ExtraTrees
        if USER_CLF_PATH.exists() and USER_LABEL_MAP_PATH.exists():
            try:
                self.user_clf = joblib.load(USER_CLF_PATH)
                with open(USER_LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                    self.user_label_map = json.load(f)
                print(f"[ANALYZER]: 50-Class User Classifier ready ({len(self.user_label_map)} signs).")
            except Exception as exc:
                print(f"[ANALYZER ERROR]: User classifier failed to load: {exc}")

        # 3. Load 1,000-Class Bi-GRU WLASL Model
        if GRU_WEIGHTS_PATH.exists():
            try:
                ckpt = torch.load(GRU_WEIGHTS_PATH, map_location="cpu", weights_only=False)
                cfg = ckpt.get("config", {"input_size": 118, "hidden_size": 256, "num_layers": 2, "num_classes": 1000, "dropout": 0.4})
                self.gru_classes = ckpt.get("classes", [])
                if not self.gru_classes and GRU_CLASSES_PATH.exists():
                    with open(GRU_CLASSES_PATH, "r", encoding="utf-8") as f:
                        self.gru_classes = json.load(f)

                self.gru_model = GestureGRU(**cfg)
                self.gru_model.load_state_dict(ckpt["model_state"], strict=False)
                self.gru_model.eval()
                print(f"[ANALYZER]: 1,000-Class WLASL Bi-GRU Model ready ({len(self.gru_classes)} signs).")
            except Exception as exc:
                print(f"[ANALYZER ERROR]: GRU model failed to load: {exc}")

    def analyze(
        self,
        video_path: str | Path,
        output_annotated_path: str | Path | None = None,
        progress_cb: Callable[[float], None] | None = None,
    ) -> dict[str, object]:
        """
        Processes an uploaded video file and returns a complete analysis report.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return {"error": f"Video file not found: {video_path}"}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Could not open video file: {video_path}"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / max(fps, 1.0)

        # Video writer if output requested
        writer = None
        if output_annotated_path is not None:
            out_p = Path(output_annotated_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

        frame_idx = 0
        all_keypoints_seq = []
        all_features_118 = []
        hand_detected_frames = 0
        active_motion_energy = 0.0
        prev_kpts = None

        print(f"[ANALYZER]: Processing '{video_path.name}' ({total_frames} frames, {duration_s:.1f}s)...")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                h, w = frame.shape[:2]
                frame_idx += 1

                # Progress callback
                if progress_cb and frame_idx % 10 == 0:
                    progress_cb(frame_idx / total_frames)

                # Extract landmarks with RTMPose Wholebody
                res = self.pose_model(frame, bboxes=[[0, 0, w, h]]) if self.pose_model else None
                norm_kpts = np.zeros((133, 3), dtype=np.float32)

                if res is not None and isinstance(res, tuple) and len(res) == 2:
                    kpts_list, scores_list = res
                    if kpts_list is not None and len(kpts_list) > 0:
                        best_idx = 0
                        kpts = kpts_list[best_idx]
                        scores = scores_list[best_idx]

                        norm_kpts[:, 0] = kpts[:, 0] / max(w, 1)
                        norm_kpts[:, 1] = kpts[:, 1] / max(h, 1)
                        norm_kpts[:, 2] = scores

                        # Measure motion energy
                        if prev_kpts is not None:
                            diff = np.linalg.norm(norm_kpts[:, :2] - prev_kpts[:, :2], axis=1)
                            active_motion_energy += float(np.mean(diff))
                        prev_kpts = norm_kpts.copy()

                        # Check hand visibility
                        r_score = float(np.mean(scores[RIGHT_HAND_IDX]))
                        l_score = float(np.mean(scores[LEFT_HAND_IDX]))
                        if r_score > 0.22 or l_score > 0.22:
                            hand_detected_frames += 1

                        # Draw skeletal overlays if saving video
                        if writer is not None:
                            if r_score > 0.22:
                                self._draw_hand_skeleton(frame, norm_kpts[RIGHT_HAND_IDX, :2], scores[RIGHT_HAND_IDX], w, h, (0, 240, 120))
                            if l_score > 0.22:
                                self._draw_hand_skeleton(frame, norm_kpts[LEFT_HAND_IDX, :2], scores[LEFT_HAND_IDX], w, h, (0, 200, 255))

                all_keypoints_seq.append(norm_kpts)

                # 118-feature vector: 17 body (x,y) = 34 + 21 left hand (x,y) = 42 + 21 right hand (x,y) = 42 -> 118
                feat_118 = np.concatenate([
                    norm_kpts[BODY_IDX, :2].flatten(),
                    norm_kpts[LEFT_HAND_IDX, :2].flatten(),
                    norm_kpts[RIGHT_HAND_IDX, :2].flatten(),
                ])
                all_features_118.append(feat_118)

                if writer is not None:
                    writer.write(frame)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        # ── Sign Language & Gesture Analysis ──────────────────────────────────
        hand_presence_ratio = hand_detected_frames / max(total_frames, 1)
        is_sign_language = hand_presence_ratio > 0.25 and (active_motion_energy > 0.15)
        language_identified = "American Sign Language (ASL)" if is_sign_language else "Non-Signing / Ambiguous"
        lang_confidence = min(0.99, max(0.40, hand_presence_ratio * 1.15)) if is_sign_language else 0.25

        seq_arr = np.array(all_keypoints_seq, dtype=np.float32)  # (T, 133, 3)
        top_user_signs: list[dict[str, object]] = []
        top_wlasl_signs: list[dict[str, object]] = []
        detected_sign_words: list[str] = []

        # 1. ExtraTrees Evaluation (High Precision Conversational Signs)
        if self.user_clf is not None and len(seq_arr) >= 15 and is_sign_language:
            try:
                # Interpolate or sample 45 frames
                indices = np.linspace(0, len(seq_arr) - 1, 45).astype(int)
                sample_45 = seq_arr[indices]
                feats = extract_sequence_features(sample_45)
                probs = self.user_clf.predict_proba([feats])[0]
                sorted_indices = np.argsort(probs)[::-1][:5]
                for idx in sorted_indices:
                    sign_name = self.user_label_map.get(str(idx), f"class_{idx}")
                    score = float(probs[idx])
                    top_user_signs.append({"sign": sign_name, "confidence": round(score, 3)})

                if top_user_signs and top_user_signs[0]["confidence"] > 0.20:
                    detected_sign_words.append(str(top_user_signs[0]["sign"]))
            except Exception as exc:
                print(f"[ANALYZER]: User classifier error: {exc}")

        # 2. 1,000-Class WLASL GRU Evaluation (Broad ASL Dictionary)
        if self.gru_model is not None and len(all_features_118) >= 10 and is_sign_language:
            try:
                indices = np.linspace(0, len(all_features_118) - 1, 45).astype(int)
                sample_118 = np.array([all_features_118[i] for i in indices], dtype=np.float32)
                tensor_in = torch.from_numpy(sample_118).unsqueeze(0)  # (1, 45, 118)
                with torch.no_grad():
                    logits = self.gru_model(tensor_in)
                    probs = torch.softmax(logits, dim=-1)[0]
                    top_scores, top_indices = torch.topk(probs, 5)
                    for s, i in zip(top_scores.tolist(), top_indices.tolist()):
                        wlasl_label = self.gru_classes[i] if i < len(self.gru_classes) else f"sign_{i}"
                        top_wlasl_signs.append({"sign": wlasl_label, "confidence": round(float(s), 3)})
            except Exception as exc:
                print(f"[ANALYZER]: GRU inference error: {exc}")

        # 3. Restructure with Gemini Flash into Natural English
        smoothed_translation = ""
        if detected_sign_words:
            smoothed_translation = smooth_asl_sentence(detected_sign_words)
        elif top_wlasl_signs and is_sign_language:
            smoothed_translation = smooth_asl_sentence([str(top_wlasl_signs[0]["sign"])])
        else:
            smoothed_translation = "No clear sign language sequence identified."

        return {
            "file_name": video_path.name,
            "duration_seconds": round(duration_s, 2),
            "total_frames": total_frames,
            "fps": round(fps, 1),
            "is_sign_language": is_sign_language,
            "detected_language": language_identified,
            "language_confidence": round(lang_confidence, 2),
            "hand_presence_ratio": round(hand_presence_ratio, 2),
            "top_conversational_signs": top_user_signs,
            "top_wlasl_vocabulary_signs": top_wlasl_signs,
            "final_english_translation": smoothed_translation,
            "annotated_video_path": str(output_annotated_path) if output_annotated_path else None,
        }

    def _draw_hand_skeleton(self, frame, hand_kpts: np.ndarray, scores: np.ndarray, img_w: int, img_h: int, color=(0, 240, 120)):
        points = []
        for i in range(21):
            x = int(hand_kpts[i][0] * img_w)
            y = int(hand_kpts[i][1] * img_h)
            points.append((x, y))
            if scores[i] > 0.25:
                cv2.circle(frame, (x, y), 3, color, -1)

        max_len = 0.16 * max(img_w, img_h)
        for p1, p2 in HAND_CONNECTIONS:
            if scores[p1] > 0.25 and scores[p2] > 0.25:
                dist = np.hypot(points[p1][0] - points[p2][0], points[p1][1] - points[p2][1])
                if dist < max_len:
                    cv2.line(frame, points[p1], points[p2], (255, 255, 255), 2, cv2.LINE_AA)


# Singleton
_analyzer = None

def get_video_analyzer() -> VideoSignAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = VideoSignAnalyzer()
    return _analyzer
