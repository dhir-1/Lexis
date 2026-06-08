"""Shared hand-landmark preprocessing for training and live inference."""

from __future__ import annotations

import numpy as np

LANDMARK_COUNT = 21
FEATURE_COUNT = LANDMARK_COUNT * 3


def _landmarks_to_array(landmarks) -> np.ndarray:
    """Convert MediaPipe landmarks or a flat feature row into a 21x3 array."""
    if len(landmarks) == LANDMARK_COUNT and hasattr(landmarks[0], "x"):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    else:
        coords = np.asarray(landmarks, dtype=np.float32)
        if coords.size != FEATURE_COUNT:
            raise ValueError(f"Expected {FEATURE_COUNT} values, got {coords.size}")
        coords = coords.reshape(LANDMARK_COUNT, 3)

    return coords


def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Make hand landmarks translation and scale invariant.

    We anchor the hand at the wrist landmark and scale by the largest
    distance from the wrist in the x/y plane. This keeps the model from
    overfitting to how close the hand is to the camera or where it sits in
    the frame.
    """
    coords = _landmarks_to_array(landmarks).copy()

    wrist = coords[0].copy()
    coords -= wrist

    scale = np.max(np.linalg.norm(coords[:, :2], axis=1))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    coords /= scale
    return coords.flatten()
