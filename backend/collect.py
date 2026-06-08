import cv2
import mediapipe as mp
import csv
import os

# Block 2 - Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

SIGNS = ["Hello", "Thank You", "Yes", "No", "Goodbye"]
SAMPLES_PER_SIGN = 75
DATA_PATH = "data/gesture_samples.csv"

os.makedirs("data", exist_ok=True)

# Block 3 - CSV Header Setup
fieldnames = [f"landmark_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
fieldnames.append("label")

file_exists = os.path.exists(DATA_PATH)

with open(DATA_PATH, mode="a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

# Block 4 - Main Collection Loop
cap = cv2.VideoCapture(0)
sign_index = 0
sample_count = 0

print(f"\nCollecting samples for: {SIGNS[sign_index]}")
print("Press SPACE to capture. Press Q to quit.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                key = cv2.waitKey(1) & 0xFF

                if key == ord(" "):
                    landmarks_flat = []
                    for landmark in hand_landmarks.landmark:
                        landmarks_flat.extend([landmark.x, landmark.y, landmark.z])

                    row = {}
                    for i in range(21):
                        row[f"landmark_{i}_x"] = landmarks_flat[i * 3]
                        row[f"landmark_{i}_y"] = landmarks_flat[i * 3 + 1]
                        row[f"landmark_{i}_z"] = landmarks_flat[i * 3 + 2]
                    row["label"] = SIGNS[sign_index]

                    with open(DATA_PATH, mode="a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(row)

                    sample_count += 1
                    print(f"[{SIGNS[sign_index]}] Sample {sample_count}/{SAMPLES_PER_SIGN} captured")

                    if sample_count >= SAMPLES_PER_SIGN:
                        sign_index += 1
                        sample_count = 0
                        if sign_index >= len(SIGNS):
                            print("\nAll samples collected. Run train.py now.")
                            break
                        print(f"\nNow collecting: {SIGNS[sign_index]}")

        cv2.putText(frame, f"Sign: {SIGNS[sign_index]} | Samples: {sample_count}/{SAMPLES_PER_SIGN}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Lexis - Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    cap.release()
    cv2.destroyAllWindows()
