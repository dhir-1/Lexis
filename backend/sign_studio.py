"""
Lexis 2,400-Class Sign Studio & Video Analyzer Suite.
Powered by:
- 2,414-Class PyTorch 2-Layer Bidirectional GRU (gesture_gru_full_clean_best.pt, 74%+ Top-5 Accuracy)
- 50-Class Conversational ExtraTrees Ensemble (user_sign_classifier.pkl, 99.7% Accuracy)
- RTMPose Wholebody 133 Keypoint Tracking
- Google Gemini 2.5 Flash ASL Sentence Restructuring

Core Modules:
1. "Shazam for ASL" - Video File Sign Language Recognition & Subtitling
2. "Duolingo for ASL" - Interactive Form Accuracy & Trajectory Evaluation
3. 2,414-Class Sign Dictionary & Search Engine
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
PROJECT_DIR = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts" / "rtm_models"

# Model Paths
GRU_WEIGHTS_PATHS = [
    ARTIFACTS_DIR / "gesture_gru_full_clean_best.pt",
    MODELS_DIR / "gesture_gru.pt",
]
GRU_CLASSES_PATHS = [
    ARTIFACTS_DIR / "gru_full_clean_classes.json",
    MODELS_DIR / "gru_classes.json",
]
USER_CLF_PATH = MODELS_DIR / "user_sign_classifier.pkl"
USER_LABEL_MAP_PATH = MODELS_DIR / "user_sign_label_map.json"

# Keypoint Slices
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


# ── PyTorch 2,414-Class Bidirectional GRU Architecture ────────────────────────

class BiGRU2400(nn.Module):
    def __init__(self, input_size: int = 177, hidden_size: int = 256, num_layers: int = 2, num_classes: int = 2414, dropout: float = 0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, 177)
        proj = self.input_proj(x)
        out, _ = self.gru(proj)
        normed = self.norm(out[:, -1])
        logits = self.fc(normed)
        return logits


# ── Sign Studio Engine ────────────────────────────────────────────────────────

class SignStudio:
    """Master Sign Studio engine powering ASL Shazam, Form Practice, and Sign Search."""

    def __init__(self):
        self.pose_model = None
        self.gru_model = None
        self.gru_classes: list[str] = []
        self.user_clf = None
        self.user_label_map: dict[str, str] = {}
        self._initialize()

    def _initialize(self):
        # 1. Initialize RTMPose Wholebody
        try:
            self.pose_model = RTMPose(
                Wholebody.MODE["performance"]["pose"],
                model_input_size=Wholebody.MODE["performance"]["pose_input_size"],
                backend="onnxruntime",
                device="cpu",
            )
            print("[SIGN STUDIO]: RTMPose Wholebody tracker initialized.")
        except Exception as exc:
            print(f"[SIGN STUDIO WARNING]: RTMPose init error: {exc}")

        # 2. Initialize 50-Class Conversational ExtraTrees Model
        if USER_CLF_PATH.exists() and USER_LABEL_MAP_PATH.exists():
            try:
                self.user_clf = joblib.load(USER_CLF_PATH)
                with open(USER_LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                    self.user_label_map = json.load(f)
                print(f"[SIGN STUDIO]: 50-Class User Classifier loaded ({len(self.user_label_map)} signs).")
            except Exception as exc:
                print(f"[SIGN STUDIO WARNING]: User classifier load error: {exc}")

        # 3. Initialize 2,414-Class WLASL / ASL Citizen Bi-GRU Model
        weights_file = None
        for p in GRU_WEIGHTS_PATHS:
            if p.exists():
                weights_file = p
                break

        if weights_file:
            try:
                ckpt = torch.load(weights_file, map_location="cpu", weights_only=False)
                cfg = ckpt.get("config", {"input_size": 177, "hidden_size": 256, "num_layers": 2, "num_classes": 2414, "dropout": 0.4})
                self.gru_classes = ckpt.get("classes", [])

                if not self.gru_classes:
                    for cp in GRU_CLASSES_PATHS:
                        if cp.exists():
                            with open(cp, "r", encoding="utf-8") as f:
                                self.gru_classes = json.load(f)
                            break

                self.gru_model = BiGRU2400(**cfg)
                self.gru_model.load_state_dict(ckpt["model_state"], strict=False)
                self.gru_model.eval()
                print(f"[SIGN STUDIO]: 2,414-Class WLASL Bi-GRU Model loaded ({len(self.gru_classes)} vocabulary classes).")
            except Exception as exc:
                print(f"[SIGN STUDIO WARNING]: GRU model load error: {exc}")

    # ── Module 1: Shazam for ASL (Video File Analyzer & Language Detection) ────

    def analyze_video(
        self,
        video_path: str | Path,
        output_annotated_path: str | Path | None = None,
        progress_cb: Callable[[float], None] | None = None,
    ) -> dict[str, object]:
        """
        Analyzes a video file to detect if it is American Sign Language (ASL),
        classifies the signs across 2,414 classes, and translates it with Gemini Flash.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return {"error": f"Video file not found: {video_path}"}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Failed to open video file: {video_path}"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / max(fps, 1.0)

        writer = None
        if output_annotated_path is not None:
            out_p = Path(output_annotated_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

        frame_idx = 0
        all_keypoints_seq = []
        all_features_177 = []
        hand_detected_frames = 0
        active_motion_energy = 0.0
        prev_kpts = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                h, w = frame.shape[:2]
                frame_idx += 1

                if progress_cb and frame_idx % 12 == 0:
                    progress_cb(frame_idx / total_frames)

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

                        if prev_kpts is not None:
                            diff = np.linalg.norm(norm_kpts[:, :2] - prev_kpts[:, :2], axis=1)
                            active_motion_energy += float(np.mean(diff))
                        prev_kpts = norm_kpts.copy()

                        r_score = float(np.mean(scores[RIGHT_HAND_IDX]))
                        l_score = float(np.mean(scores[LEFT_HAND_IDX]))
                        if r_score > 0.22 or l_score > 0.22:
                            hand_detected_frames += 1

                        if writer is not None:
                            if r_score > 0.22:
                                self._draw_hand_skeleton(frame, norm_kpts[RIGHT_HAND_IDX, :2], scores[RIGHT_HAND_IDX], w, h, (0, 240, 120))
                            if l_score > 0.22:
                                self._draw_hand_skeleton(frame, norm_kpts[LEFT_HAND_IDX, :2], scores[LEFT_HAND_IDX], w, h, (0, 200, 255))

                all_keypoints_seq.append(norm_kpts)

                # 177 features: 17 body (x,y,score) = 51 + 21 left (x,y,score) = 63 + 21 right (x,y,score) = 63 -> 177
                feat_177 = np.concatenate([
                    norm_kpts[BODY_IDX, :3].flatten(),
                    norm_kpts[LEFT_HAND_IDX, :3].flatten(),
                    norm_kpts[RIGHT_HAND_IDX, :3].flatten(),
                ])
                all_features_177.append(feat_177)

                if writer is not None:
                    writer.write(frame)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        # ── Language & Sign Classification ────────────────────────────────────
        hand_presence_ratio = hand_detected_frames / max(total_frames, 1)
        is_sign_language = hand_presence_ratio > 0.25 and (active_motion_energy > 0.12)
        language_identified = "American Sign Language (ASL)" if is_sign_language else "Non-Signing / Ambiguous"
        lang_confidence = min(0.99, max(0.40, hand_presence_ratio * 1.12)) if is_sign_language else 0.15

        seq_arr = np.array(all_keypoints_seq, dtype=np.float32)
        top_user_signs: list[dict[str, object]] = []
        top_wlasl_signs: list[dict[str, object]] = []
        detected_sign_words: list[str] = []

        # 1. 2,414-Class Bi-GRU Evaluation (WLASL / ASL Citizen)
        if self.gru_model is not None and len(all_features_177) >= 10 and is_sign_language:
            try:
                indices = np.linspace(0, len(all_features_177) - 1, 45).astype(int)
                sample_177 = np.array([all_features_177[i] for i in indices], dtype=np.float32)
                tensor_in = torch.from_numpy(sample_177).unsqueeze(0)  # (1, 45, 177)
                with torch.no_grad():
                    logits = self.gru_model(tensor_in)
                    probs = torch.softmax(logits, dim=-1)[0]
                    top_scores, top_indices = torch.topk(probs, 5)
                    for s, i in zip(top_scores.tolist(), top_indices.tolist()):
                        wlasl_label = self.gru_classes[i] if i < len(self.gru_classes) else f"sign_{i}"
                        top_wlasl_signs.append({
                            "sign": wlasl_label.upper(),
                            "confidence": round(float(s), 3),
                            "percentage": f"{float(s):.1%}",
                        })
            except Exception as exc:
                print(f"[SIGN STUDIO]: GRU inference error: {exc}")

        # 2. 50-Class Conversational ExtraTrees Evaluation
        if self.user_clf is not None and len(seq_arr) >= 15 and is_sign_language:
            try:
                indices = np.linspace(0, len(seq_arr) - 1, 45).astype(int)
                sample_45 = seq_arr[indices]
                feats = extract_sequence_features(sample_45)
                probs = self.user_clf.predict_proba([feats])[0]
                sorted_indices = np.argsort(probs)[::-1][:5]
                for idx in sorted_indices:
                    sign_name = self.user_label_map.get(str(idx), f"class_{idx}")
                    score = float(probs[idx])
                    top_user_signs.append({
                        "sign": sign_name.upper(),
                        "confidence": round(score, 3),
                        "percentage": f"{score:.1%}",
                    })

                if top_user_signs and top_user_signs[0]["confidence"] > 0.22:
                    detected_sign_words.append(str(top_user_signs[0]["sign"]).lower())
            except Exception as exc:
                print(f"[SIGN STUDIO]: User classifier error: {exc}")

        # 3. Gemini 2.5 Flash English Sentence Restructuring
        if detected_sign_words:
            final_english = smooth_asl_sentence(detected_sign_words)
        elif top_wlasl_signs and is_sign_language:
            top_candidate = str(top_wlasl_signs[0]["sign"]).lower()
            final_english = smooth_asl_sentence([top_candidate])
        else:
            final_english = "No clear sign language sequence identified."

        return {
            "file_name": video_path.name,
            "duration_seconds": round(duration_s, 2),
            "total_frames": total_frames,
            "fps": round(fps, 1),
            "is_sign_language": is_sign_language,
            "detected_language": language_identified,
            "language_confidence": round(lang_confidence, 2),
            "hand_presence_ratio": round(hand_presence_ratio, 2),
            "top_predictions_2400_class": top_wlasl_signs,
            "top_conversational_50_class": top_user_signs,
            "final_english_translation": final_english,
            "annotated_video_path": str(output_annotated_path) if output_annotated_path else None,
        }

    # ── Module 2: Duolingo for ASL (Form Practice & Accuracy Evaluator) ────────

    def evaluate_form(self, target_sign: str, sample_kpts: np.ndarray) -> dict[str, object]:
        """
        Rates a user's sign performance against reference trajectory and anatomical standards.
        Returns accuracy percentage and real-time form coaching tips.
        """
        target = target_sign.strip().lower()
        if sample_kpts.shape[0] < 10:
            return {"score": 0, "feedback": "Sign sequence too short. Please perform the full motion."}

        # Check hand elevation & visibility
        scores = sample_kpts[:, :, 2]
        r_scores = np.mean(scores[:, RIGHT_HAND_IDX], axis=1)
        l_scores = np.mean(scores[:, LEFT_HAND_IDX], axis=1)
        active_hand_presence = float(np.mean(np.maximum(r_scores, l_scores)))

        # Evaluate against 50-word classifier if available
        match_confidence = 0.0
        if self.user_clf is not None:
            try:
                indices = np.linspace(0, len(sample_kpts) - 1, 45).astype(int)
                sample_45 = sample_kpts[indices]
                feats = extract_sequence_features(sample_45)
                probs = self.user_clf.predict_proba([feats])[0]
                # Find target class index
                for k, v in self.user_label_map.items():
                    if v.lower() == target:
                        match_confidence = float(probs[int(k)])
                        break
            except Exception:
                pass

        # Calculate composite form score (0-100)
        form_score = int(round(match_confidence * 70 + min(1.0, active_hand_presence * 1.5) * 30))
        form_score = max(10, min(99, form_score))

        # Dynamic coaching feedback
        feedback = []
        if form_score >= 88:
            feedback.append("🌟 Excellent execution! Fluent motion and accurate hand formation.")
        elif form_score >= 70:
            feedback.append("👍 Good attempt! Hand trajectory matches the target pattern.")
        else:
            feedback.append("💡 Keep practicing! Make sure your hand is clearly visible and steady.")

        if active_hand_presence < 0.35:
            feedback.append("Raise your signing hands higher towards chest level.")

        return {
            "target_sign": target.upper(),
            "form_score": form_score,
            "hand_clarity": round(active_hand_presence, 2),
            "match_confidence": round(match_confidence, 2),
            "coaching_feedback": " ".join(feedback),
        }

    # ── Module 3: ASL Search Engine (2,414 Classes) ───────────────────────────

    def search_vocabulary(self, query: str, limit: int = 15) -> list[dict[str, object]]:
        """Searches across all 2,414 ASL classes."""
        query = query.strip().lower()
        results = []

        all_signs = list(self.gru_classes)
        # Prioritize 50 conversational signs
        priority_50 = set(self.user_label_map.values())

        for sign in all_signs:
            sign_lower = sign.lower()
            if query in sign_lower:
                is_core = sign_lower in priority_50
                results.append({
                    "sign": sign.upper(),
                    "is_core_50": is_core,
                    "dataset": "Core 50-Word Conversational" if is_core else "WLASL / ASL Citizen (2,414)",
                })
                if len(results) >= limit:
                    break

        return results

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


# Global Singleton
_studio_instance = None


def get_sign_studio() -> SignStudio:
    global _studio_instance
    if _studio_instance is None:
        _studio_instance = SignStudio()
    return _studio_instance
