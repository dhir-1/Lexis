import os
import cv2
import csv
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
from gesture_features import normalize_landmarks

DATASET_PATH = "data/asl_dataset/asl_alphabet_train"
OUTPUT_CSV = "data/asl_landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

def get_already_done_labels():
    if not os.path.exists(OUTPUT_CSV):
        return set()
    df = pd.read_csv(OUTPUT_CSV)
    return set(df["label"].unique())

def extract_features_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None
    landmarks = result.multi_hand_landmarks[0]
    normalized = normalize_landmarks(landmarks.landmark)
    if normalized is None:
        return None
    return normalized

def main():
    already_done = get_already_done_labels()
    if already_done:
        print(f"[RESUME] Skipping already done labels: {sorted(already_done)}")

    labels = sorted(os.listdir(DATASET_PATH))
    labels_to_process = [l for l in labels if l not in already_done]

    if not labels_to_process:
        print("All labels already extracted. Nothing to do.")
        return

    print(f"Labels to process: {labels_to_process}")

    skipped_total = 0
    file_exists = os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="") as csvfile:
        writer = None

        for label in labels_to_process:
            label_path = os.path.join(DATASET_PATH, label)
            if not os.path.isdir(label_path):
                continue

            image_files = [
                f for f in os.listdir(label_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            skipped = 0
            processed = 0

            print(f"\n[{label}] Processing {len(image_files)} images...")

            for i, fname in enumerate(tqdm(image_files, desc=label)):
                image_path = os.path.join(label_path, fname)
                try:
                    features = extract_features_from_image(image_path)
                    if features is None:
                        skipped += 1
                        continue

                    if writer is None:
                        columns = []
                        for j in range(21):
                            columns += [
                                f"landmark_{j}_x",
                                f"landmark_{j}_y",
                                f"landmark_{j}_z"
                            ]
                        columns.append("label")
                        writer = csv.DictWriter(csvfile, fieldnames=columns)
                        if not file_exists:
                            writer.writeheader()
                            file_exists = True

                    row = {}
                    for j in range(21):
                        row[f"landmark_{j}_x"] = features[j * 3]
                        row[f"landmark_{j}_y"] = features[j * 3 + 1]
                        row[f"landmark_{j}_z"] = features[j * 3 + 2]
                    row["label"] = label
                    writer.writerow(row)
                    processed += 1

                except Exception as e:
                    skipped += 1
                    continue

                if (i + 1) % 500 == 0:
                    csvfile.flush()
                    print(f"  [{label}] {i+1} images done, {skipped} skipped so far")

            skipped_total += skipped
            print(f"[{label}] Done — {processed} saved, {skipped} skipped")

    print(f"\nExtraction complete.")
    print(f"Total skipped (no hand detected): {skipped_total}")
    print(f"Output saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()