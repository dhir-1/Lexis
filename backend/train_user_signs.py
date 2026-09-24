from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "user_recorded_signs"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODELS_DIR / "user_sign_classifier.pkl"
LABEL_MAP_PATH = MODELS_DIR / "user_sign_label_map.json"

# Keypoint subsets
NOSE_IDX = 0
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)


def extract_frame_features(frame_kpts: np.ndarray) -> np.ndarray:
    """
    Extract robust normalized spatial features from a single frame (133, 3).
    Includes:
    - Relative hand locations wrt shoulders, nose, and each other
    - Wrist-normalized hand shapes (right + left)
    - Finger extension distances
    """
    xy = frame_kpts[:, :2]
    scores = frame_kpts[:, 2]

    nose = xy[NOSE_IDX]
    l_sh = xy[LEFT_SHOULDER_IDX]
    r_sh = xy[RIGHT_SHOULDER_IDX]
    mid_sh = (l_sh + r_sh) * 0.5
    sh_dist = float(np.linalg.norm(l_sh - r_sh))
    if not np.isfinite(sh_dist) or sh_dist < 1e-4:
        sh_dist = 0.25

    # Hands
    r_hand = xy[RIGHT_HAND_SLICE].copy()
    l_hand = xy[LEFT_HAND_SLICE].copy()
    r_scores = scores[RIGHT_HAND_SLICE]
    l_scores = scores[LEFT_HAND_SLICE]

    # Hand presence flags
    r_present = float(np.mean(r_scores) > 0.15)
    l_present = float(np.mean(l_scores) > 0.15)

    # Relative spatial positions normalized by shoulder distance
    r_wrist_rel_nose = (r_hand[0] - nose) / sh_dist
    r_wrist_rel_chest = (r_hand[0] - mid_sh) / sh_dist
    l_wrist_rel_nose = (l_hand[0] - nose) / sh_dist
    l_wrist_rel_chest = (l_hand[0] - mid_sh) / sh_dist
    hands_rel_dist = (r_hand[0] - l_hand[0]) / sh_dist

    # Normalized Right Hand Shape (wrist-centered + scale invariant)
    r_hand_centered = r_hand - r_hand[0]
    r_scale = float(np.max(np.linalg.norm(r_hand_centered, axis=1)))
    if not np.isfinite(r_scale) or r_scale < 1e-4:
        r_scale = 1.0
    r_hand_norm = (r_hand_centered / r_scale).flatten()

    # Normalized Left Hand Shape
    l_hand_centered = l_hand - l_hand[0]
    l_scale = float(np.max(np.linalg.norm(l_hand_centered, axis=1)))
    if not np.isfinite(l_scale) or l_scale < 1e-4:
        l_scale = 1.0
    l_hand_norm = (l_hand_centered / l_scale).flatten()

    # Key fingertip distances to wrist (thumb=4, index=8, middle=12, ring=16, pinky=20)
    fingertip_idx = [4, 8, 12, 16, 20]
    r_finger_lengths = np.linalg.norm(r_hand_centered[fingertip_idx], axis=1) / r_scale
    l_finger_lengths = np.linalg.norm(l_hand_centered[fingertip_idx], axis=1) / l_scale

    feats = np.concatenate([
        [r_present, l_present],
        r_wrist_rel_nose,
        r_wrist_rel_chest,
        l_wrist_rel_nose,
        l_wrist_rel_chest,
        hands_rel_dist,
        r_finger_lengths,
        l_finger_lengths,
        r_hand_norm,
        l_hand_norm,
    ])
    return feats.astype(np.float32)


def build_sequence_vector_from_matrix(feats_matrix: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    """
    Construct full sequence vector instantly from pre-extracted frame features matrix.
    Runs in < 0.1ms.
    """
    num_frames = feats_matrix.shape[0]

    # 1. Temporal keyframe sampling (5 evenly spaced moments: 0%, 25%, 50%, 75%, 100%)
    sample_indices = np.linspace(0, num_frames - 1, 5, dtype=int)
    sampled_keyframes = feats_matrix[sample_indices].flatten()

    # 2. Sequence statistics (Mean, Std, Net Movement Delta: End - Start)
    mean_feats = np.mean(feats_matrix, axis=0)
    std_feats = np.std(feats_matrix, axis=0)
    delta_feats = feats_matrix[-1] - feats_matrix[0]

    # 3. Kinematic motion velocities
    if num_frames > 1:
        step_diffs = np.linalg.norm(feats_matrix[1:] - feats_matrix[:-1], axis=1)
        mean_vel = float(np.mean(step_diffs))
        max_vel = float(np.max(step_diffs))
        std_vel = float(np.std(step_diffs))
    else:
        mean_vel, max_vel, std_vel = 0.0, 0.0, 0.0

    # 4. Wrist path straightness & circularity (Right & Left)
    rw_xy = sequence[:, 112, :2]
    lw_xy = sequence[:, 91, :2]

    rw_delta = rw_xy[-1] - rw_xy[0]
    rw_path_len = float(np.sum(np.linalg.norm(rw_xy[1:] - rw_xy[:-1], axis=1))) if num_frames > 1 else 0.0
    rw_straightness = float(np.linalg.norm(rw_delta)) / (rw_path_len + 1e-4)

    lw_delta = lw_xy[-1] - lw_xy[0]
    lw_path_len = float(np.sum(np.linalg.norm(lw_xy[1:] - lw_xy[:-1], axis=1))) if num_frames > 1 else 0.0
    lw_straightness = float(np.linalg.norm(lw_delta)) / (lw_path_len + 1e-4)

    kinematics = np.array([
        mean_vel,
        max_vel,
        std_vel,
        rw_path_len,
        rw_straightness,
        lw_path_len,
        lw_straightness,
    ], dtype=np.float32)

    return np.concatenate([
        sampled_keyframes,
        mean_feats,
        std_feats,
        delta_feats,
        kinematics,
    ]).astype(np.float32)


def extract_sequence_features(sequence: np.ndarray) -> np.ndarray:
    """
    Extract a compact spatio-temporal feature vector from a (T, 133, 3) sequence.
    """
    num_frames = sequence.shape[0]
    frame_feats_list = [extract_frame_features(sequence[t]) for t in range(num_frames)]
    feats_matrix = np.stack(frame_feats_list, axis=0)  # Shape: (T, D)
    return build_sequence_vector_from_matrix(feats_matrix, sequence)




def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Horizontally mirror a (T, 133, 3) landmark sequence.
    Flips X coordinates (1.0 - x) and swaps left/right body & hand landmarks.
    Produces an ambidextrous left-handed clone of right-handed signs.
    """
    flipped = seq.copy()
    # 1. Flip X coordinates
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]

    # 2. Swap bilateral body keypoints
    body_swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    for left_idx, right_idx in body_swap_pairs:
        tmp = flipped[:, left_idx].copy()
        flipped[:, left_idx] = flipped[:, right_idx]
        flipped[:, right_idx] = tmp

    # 3. Swap bilateral hands: Left Hand (91-111) <-> Right Hand (112-132)
    left_hand = flipped[:, 91:112].copy()
    right_hand = flipped[:, 112:133].copy()
    flipped[:, 91:112] = right_hand
    flipped[:, 112:133] = left_hand

    return flipped


def augment_sequence(seq: np.ndarray, num_augments: int = 12) -> list[np.ndarray]:
    """
    Generate realistic temporal and spatial sub-windows from a single recorded sample.
    Extracts sliding windows (e.g., length 30-42), slight scale/jitter variations,
    and synthetic horizontal mirroring for ambidextrous support.
    """
    T = seq.shape[0]
    augmented = [seq]  # Include original sequence

    if T < 20:
        return augmented

    # 1. Sliding temporal windows (e.g. 30 to 40 frames)
    for win_len in [30, 35, 40]:
        if win_len >= T:
            continue
        max_start = T - win_len
        for start_idx in np.linspace(0, max_start, 4, dtype=int):
            sub_seq = seq[start_idx : start_idx + win_len].copy()
            augmented.append(sub_seq)

    # 2. Synthetic horizontal mirroring (Ambidextrous Left/Right Hand clone)
    mirrored_seq = mirror_sequence(seq)
    augmented.append(mirrored_seq)
    if T >= 35:
        augmented.append(mirrored_seq[:35].copy())
        augmented.append(mirrored_seq[-35:].copy())

    # 3. Resampling & slight noise/scaling
    rng = np.random.RandomState(42)
    for _ in range(num_augments):
        win_size = rng.randint(max(25, T - 15), T + 1)
        start = rng.randint(0, max(1, T - win_size + 1))
        sub = seq[start : start + win_size].copy()

        # Slight spatial jitter (Gaussian noise on x,y with small std)
        noise = rng.normal(0, 0.003, size=sub.shape).astype(np.float32)
        noise[:, :, 2] = 0  # Do not jitter scores
        sub_jittered = sub + noise
        augmented.append(sub_jittered)

    return augmented


def load_dataset(augment: bool = True):
    if not DATA_DIR.exists():
        print(f"[ERROR]: Data directory {DATA_DIR} does not exist.")
        return None, None, None

    word_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    X_list = []
    y_list = []
    class_names = []

    for label_idx, word_dir in enumerate(word_dirs):
        samples = list(word_dir.glob("sample_*.npy"))
        if not samples:
            continue

        class_names.append(word_dir.name)
        current_class_idx = len(class_names) - 1

        for sample_path in samples:
            try:
                seq = np.load(sample_path)
                if seq.ndim != 3 or seq.shape[1] != 133:
                    continue

                if augment:
                    aug_seqs = augment_sequence(seq)
                    for aseq in aug_seqs:
                        feat_vec = extract_sequence_features(aseq)
                        X_list.append(feat_vec)
                        y_list.append(current_class_idx)
                else:
                    feat_vec = extract_sequence_features(seq)
                    X_list.append(feat_vec)
                    y_list.append(current_class_idx)
            except Exception as e:
                print(f"[WARN]: Error loading {sample_path.name}: {e}")

    if not X_list:
        print("[ERROR]: No valid sign samples found to train on.")
        return None, None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, class_names



def main():
    print("=" * 65)
    print("      🧠 LEXIS - USER SIGN CLASSIFIER TRAINER")
    print("=" * 65)

    X, y, class_names = load_dataset()
    if X is None or len(class_names) == 0:
        print("[STOP]: Please record samples first using record_signs.py.")
        return 1

    num_samples = len(X)
    num_classes = len(class_names)
    print(f"\n[DATASET]: Loaded {num_samples} samples across {num_classes} classes:")
    for idx, cname in enumerate(class_names):
        count = int(np.sum(y == idx))
        print(f"  [{idx + 1:02d}] '{cname}' -> {count} samples")

    if num_classes < 2:
        print("\n[WARN]: Need at least 2 distinct words recorded to train a classifier.")
        return 1

    # Train Model (Ensemble ExtraTrees Classifier for maximum generalizability on landmark geometry)
    print(f"\n[TRAIN]: Training high-accuracy landmark classifier...")
    clf = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )


    # Cross-validation if sample size permits
    min_class_samples = min(int(np.sum(y == idx)) for idx in range(num_classes))
    if min_class_samples >= 2:
        n_splits = min(3, min_class_samples)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv)
        print(f"[EVAL]: {n_splits}-Fold Cross-Validation Accuracy: {scores.mean() * 100:.1f}% (+/- {scores.std() * 100:.1f}%)")

    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"[RESULT]: Training Set Accuracy: {train_acc * 100:.1f}%")

    # Save Model and Label Mapping
    joblib.dump(clf, MODEL_SAVE_PATH)
    label_map = {idx: name for idx, name in enumerate(class_names)}
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n[SAVED]: Model saved to -> {MODEL_SAVE_PATH}")
    print(f"[SAVED]: Label map saved to -> {LABEL_MAP_PATH}")
    print("=" * 65)
    print("✨ Classifier is ready for real-time live vision classification!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
