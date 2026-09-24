import json
from pathlib import Path
import joblib
import numpy as np
from train_user_signs import extract_sequence_features, extract_frame_features

DATA_DIR = Path("data/user_recorded_signs")
MODELS_DIR = Path("models")

clf = joblib.load(MODELS_DIR / "user_sign_classifier.pkl")
with open(MODELS_DIR / "user_sign_label_map.json", "r") as f:
    label_map = json.load(f)

print(f"=== CLASSIFIER CLASSES ===")
print(label_map)

print("\n=== EVALUATING RECORDED SAMPLES ===")
for word_dir in sorted(DATA_DIR.iterdir()):
    if not word_dir.is_dir():
        continue
    samples = list(word_dir.glob("sample_*.npy"))
    print(f"\n--- Word: '{word_dir.name}' ({len(samples)} samples) ---")
    for s in samples:
        arr = np.load(s)
        # Check hand visibility in this sample
        r_scores = arr[:, 112:133, 2].mean()
        l_scores = arr[:, 91:112, 2].mean()
        
        feats = extract_sequence_features(arr)
        probs = clf.predict_proba([feats])[0]
        best_idx = int(np.argmax(probs))
        pred_label = label_map[str(best_idx)]
        conf = probs[best_idx]
        print(f"  {s.name} -> RightScores: {r_scores:.2f}, LeftScores: {l_scores:.2f} | Pred: {pred_label} ({conf:.1%}) | Probs: {[f'{p:.2f}' for p in probs]}")
