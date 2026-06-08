"""
WLASL Batch Feature Extractor
Extracts MediaPipe landmarks (both hands + pose) from WLASL videos.
Processes in batches of 1000 videos. Has resume support.
Output: data/wlasl_sequences/{gloss}/{video_id}.npy
Each .npy shape: (T, 225) — T frames, 225 features per frame
Features: left_hand(63) + right_hand(63) + pose(99)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
VIDEOS_DIR      = BASE_DIR / "data" / "wlasl" / "videos"
NSLT_JSON       = BASE_DIR / "data" / "wlasl" / "nslt_100.json"
WLASL_JSON      = BASE_DIR / "data" / "wlasl" / "WLASL_v0.3.json"
CLASS_LIST      = BASE_DIR / "data" / "wlasl" / "wlasl_class_list"
OUTPUT_DIR      = BASE_DIR / "data" / "wlasl_sequences"
MISSING_LOG     = BASE_DIR / "data" / "wlasl_extract_missing.txt"

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 1000
MAX_FRAMES      = 64     # pad/truncate all sequences to this length
FEATURE_DIM     = 225    # 63 + 63 + 99

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands  = mp.solutions.hands
mp_pose   = mp.solutions.pose


def load_class_list(path: Path) -> dict[int, str]:
    """Returns {class_idx: gloss}"""
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1].strip().lower()
    return mapping


def load_nslt(path: Path) -> dict[str, dict]:
    """Returns {video_id: {subset, class_idx}}"""
    with open(path, "r") as f:
        raw = json.load(f)
    result = {}
    for video_id, info in raw.items():
        result[video_id] = {
            "subset": info["subset"],
            "class_idx": info["action"][0],
        }
    return result


def load_wlasl_meta(path: Path) -> dict[str, dict]:
    """Returns {video_id: {frame_start, frame_end, bbox}}"""
    with open(path, "r") as f:
        raw = json.load(f)
    meta = {}
    for entry in raw:
        for instance in entry["instances"]:
            meta[str(instance["video_id"])] = {
                "frame_start": instance["frame_start"],
                "frame_end":   instance["frame_end"],
                "bbox":        instance["bbox"],  # [x1, y1, x2, y2]
            }
    return meta


def normalize_hand(landmarks) -> np.ndarray:
    """Normalize 21 hand landmarks to translation+scale invariant 63-float vector."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = coords[0].copy()
    coords -= wrist
    scale = np.max(np.linalg.norm(coords[:, :2], axis=1))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    coords /= scale
    return coords.flatten()  # (63,)


def normalize_pose(landmarks) -> np.ndarray:
    """
    Normalize 33 pose landmarks relative to hip center and shoulder width.
    Returns 99-float vector.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # Anchor at hip center (landmarks 23 and 24)
    hip_center = (coords[23] + coords[24]) / 2.0
    coords -= hip_center

    # Scale by shoulder width (landmarks 11 and 12)
    shoulder_width = np.linalg.norm(coords[11, :2] - coords[12, :2])
    if not np.isfinite(shoulder_width) or shoulder_width < 1e-6:
        shoulder_width = 1.0
    coords /= shoulder_width

    return coords.flatten()  # (99,)


def extract_sequence(
    video_path: Path,
    frame_start: int,
    frame_end: int,
    bbox: list[int],
    hands_detector,
    pose_detector,
) -> np.ndarray | None:
    """
    Extract landmark sequence from one video.
    Returns array of shape (T, 225) or None if extraction fails.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fs = max(0, frame_start - 1)          # convert 1-indexed to 0-indexed
    fe = total_frames if frame_end == -1 else min(frame_end, total_frames)

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, fs)

    x1, y1, x2, y2 = bbox
    sequence = []

    frame_idx = fs
    while frame_idx < fe:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Crop to bbox if valid
        h, w = frame.shape[:2]
        cx1 = max(0, min(x1, w - 1))
        cy1 = max(0, min(y1, h - 1))
        cx2 = max(cx1 + 1, min(x2, w))
        cy2 = max(cy1 + 1, min(y2, h))
        if cx2 > cx1 and cy2 > cy1:
            frame = frame[cy1:cy2, cx1:cx2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract hand landmarks
        hand_result = hands_detector.process(rgb)
        pose_result = pose_detector.process(rgb)

        left_hand  = np.zeros(63,  dtype=np.float32)
        right_hand = np.zeros(63,  dtype=np.float32)
        pose_feat  = np.zeros(99,  dtype=np.float32)

        if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
            for hand_lms, handedness in zip(
                hand_result.multi_hand_landmarks,
                hand_result.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                features = normalize_hand(hand_lms.landmark)
                if label == "Left":
                    left_hand = features
                else:
                    right_hand = features

        if pose_result.pose_landmarks:
            pose_feat = normalize_pose(pose_result.pose_landmarks.landmark)

        frame_features = np.concatenate([left_hand, right_hand, pose_feat])  # (225,)
        sequence.append(frame_features)
        frame_idx += 1

    cap.release()

    if len(sequence) < 4:  # too short to be useful
        return None

    return np.array(sequence, dtype=np.float32)  # (T, 225)


def pad_or_truncate(sequence: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Pad with zeros or truncate to fixed length."""
    T = sequence.shape[0]
    if T >= max_frames:
        # Uniform sampling to reduce to max_frames
        indices = np.linspace(0, T - 1, max_frames, dtype=int)
        return sequence[indices]
    else:
        pad = np.zeros((max_frames - T, FEATURE_DIM), dtype=np.float32)
        return np.vstack([sequence, pad])


def get_already_done() -> set[str]:
    """Returns set of video_ids already extracted (resume support)."""
    done = set()
    if not OUTPUT_DIR.exists():
        return done
    for gloss_dir in OUTPUT_DIR.iterdir():
        if gloss_dir.is_dir():
            for npy_file in gloss_dir.glob("*.npy"):
                done.add(npy_file.stem)  # stem = video_id
    return done


def process_batch(
    batch: list[tuple[str, str, str]],  # [(video_id, gloss, subset), ...]
    wlasl_meta: dict,
    hands_detector,
    pose_detector,
    missing_log,
) -> tuple[int, int]:
    """Process one batch. Returns (saved, skipped)."""
    saved = 0
    skipped = 0

    for video_id, gloss, subset in tqdm(batch, desc="Batch", leave=False):
        video_path = VIDEOS_DIR / f"{video_id}.mp4"

        if not video_path.exists():
            missing_log.write(f"{video_id}\t{gloss}\tmissing_file\n")
            skipped += 1
            continue

        meta = wlasl_meta.get(video_id, {})
        frame_start = meta.get("frame_start", 1)
        frame_end   = meta.get("frame_end", -1)
        bbox        = meta.get("bbox", [0, 0, 9999, 9999])

        try:
            sequence = extract_sequence(
                video_path, frame_start, frame_end, bbox,
                hands_detector, pose_detector
            )
        except Exception as exc:
            missing_log.write(f"{video_id}\t{gloss}\terror:{exc}\n")
            skipped += 1
            continue

        if sequence is None:
            missing_log.write(f"{video_id}\t{gloss}\tno_landmarks\n")
            skipped += 1
            continue

        sequence = pad_or_truncate(sequence)  # (64, 225)

        out_dir = OUTPUT_DIR / gloss / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{video_id}.npy", sequence)
        saved += 1

    return saved, skipped


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[EXTRACT] Loading metadata...")
    class_map  = load_class_list(CLASS_LIST)   # {idx: gloss}
    nslt       = load_nslt(NSLT_JSON)          # {video_id: {subset, class_idx}}
    wlasl_meta = load_wlasl_meta(WLASL_JSON)   # {video_id: {frame_start, frame_end, bbox}}

    # Build full job list
    all_jobs: list[tuple[str, str, str]] = []
    for video_id, info in nslt.items():
        gloss  = class_map.get(info["class_idx"], f"class_{info['class_idx']}")
        subset = info["subset"]
        all_jobs.append((video_id, gloss, subset))

    print(f"[EXTRACT] Total videos in nslt_100: {len(all_jobs)}")

    # Resume — skip already done
    already_done = get_already_done()
    jobs = [(vid, gloss, subset) for vid, gloss, subset in all_jobs if vid not in already_done]
    print(f"[EXTRACT] Already done: {len(already_done)} | Remaining: {len(jobs)}")

    if not jobs:
        print("[EXTRACT] All videos already extracted.")
        return

    # Split into batches
    batches = [jobs[i:i + BATCH_SIZE] for i in range(0, len(jobs), BATCH_SIZE)]
    print(f"[EXTRACT] Total batches: {len(batches)} (batch size: {BATCH_SIZE})")

    total_saved   = 0
    total_skipped = 0

    with open(MISSING_LOG, "a") as missing_log:
        for batch_idx, batch in enumerate(batches):
            print(f"\n[EXTRACT] Batch {batch_idx + 1}/{len(batches)} — {len(batch)} videos")
            start = time.time()

            # Reinitialize MediaPipe per batch to avoid memory leaks
            hands_detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            pose_detector = mp.solutions.pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )

            saved, skipped = process_batch(
                batch, wlasl_meta, hands_detector, pose_detector, missing_log
            )

            hands_detector.close()
            pose_detector.close()

            total_saved   += saved
            total_skipped += skipped
            elapsed = time.time() - start

            print(f"[EXTRACT] Batch {batch_idx + 1} done — saved: {saved}, skipped: {skipped}, time: {elapsed:.1f}s")
            print(f"[EXTRACT] Running total — saved: {total_saved}, skipped: {total_skipped}")

    print(f"\n[EXTRACT] Complete — total saved: {total_saved}, total skipped: {total_skipped}")
    print(f"[EXTRACT] Sequences saved to: {OUTPUT_DIR}")
    print(f"[EXTRACT] Missing/failed log: {MISSING_LOG}")


if __name__ == "__main__":
    main()