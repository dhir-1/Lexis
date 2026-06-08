from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
ASL_DATA_PATH = BASE_DIR / "data" / "asl_landmarks.csv"
OLD_DATA_PATH = BASE_DIR / "data" / "gesture_samples.csv"
MODEL_PATH = BASE_DIR / "models" / "gesture_classifier.pkl"

USE_ASL_DATASET = True


def main() -> int:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if USE_ASL_DATASET:
        print("[TRAIN] Loading ASL landmark dataset...")
        df = pd.read_csv(ASL_DATA_PATH)
    else:
        print("[TRAIN] Loading old gesture samples...")
        df = pd.read_csv(OLD_DATA_PATH)

    X = df.drop("label", axis=1).values
    y = df["label"].values

    print(f"[TRAIN] Dataset shape: {X.shape}, Labels: {len(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=True
        ))
    ])

    print("[TRAIN] Training MLPClassifier pipeline...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(confusion_matrix(y_test, predictions))

    if accuracy >= 0.90:
        joblib.dump(pipeline, MODEL_PATH)
        print(f"\n[TRAIN] Model saved to {MODEL_PATH}")
    else:
        print("\n[TRAIN] Accuracy below 90% — check your data before proceeding.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())