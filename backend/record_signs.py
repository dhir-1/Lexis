from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from rtmlib import RTMPose, Wholebody

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data" / "user_recorded_signs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = int(os.getenv("VISION_CAMERA_INDEX", "0"))
FRAMES_PER_SAMPLE = 45  # ~1.5 seconds at 30 FPS
COUNTDOWN_SECONDS = 3

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

# Standard 200 Core Conversational ASL Vocabulary
VOCABULARY_LIST = [
    # Batch 1 (Words 1-5: Recorded & Verified)
    "my", "name", "water", "drink", "eat",
    
    # Batch 2 (Words 6-15: Recorded & Verified)
    "hello", "goodbye", "thank you", "please", "sorry", "help", "want", "like", "yes", "no",
    
    # Batch 3 (Next 10 Words: 16-25)
    "i love you", "you", "who", "what", "where", "why", "how", "when", "more", "stop",

    # Batch 4 (Words 26-35)
    "again", "fine", "happy", "sad", "school", "book", "car", "drive", "work", "play",

    # Batch 5 (Words 36-50: People, Places & Essentials)
    "friend", "family", "mother", "father", "baby", "brother", "sister", "man", "woman",
    "teacher", "student", "house", "home", "bathroom", "time",

    # Batch 6 (Words 51-75: Time, Food & Commerce)
    "now", "today", "yesterday", "tomorrow", "day", "night", "morning", "afternoon",
    "coffee", "milk", "tea", "apple", "bread", "cheese", "meat", "pizza", "store", "buy",
    "money", "pay", "cost", "cheap", "expensive", "big", "small",

    # Batch 7 (Words 76-100: State, Health, Mind & Actions)
    "hot", "cold", "good", "bad", "beautiful", "tired", "sick", "doctor", "hospital",
    "medicine", "hurt", "clean", "dirty", "understand", "know", "think", "forget", "remember",
    "learn", "read", "write", "sign", "language", "walk", "run",
 "sit", "stand", "go", "come", "see", "hear", "listen", "talk",
    "ask", "answer", "open", "close", "start", "finish", "wait", "meet", "live", "stay",
    "change", "need", "have", "give", "take", "bring", "find", "lose", "feel", "hope",
    "love", "hate", "angry", "scared", "surprised", "bored", "busy", "ready", "same", "different",
    "new", "old", "fast", "slow", "hard", "easy", "right", "wrong", "true", "false",
    "always", "never", "sometimes", "maybe", "with", "without", "for", "from", "about", "all",
    "some", "none", "many", "few", "first", "last", "next", "before", "after", "here",
    "there", "near", "far", "inside", "outside", "up", "down", "left", "right_dir", "front",
    "back", "center", "sun", "moon", "rain", "snow", "wind", "dog", "cat", "bird",
    "fish", "phone", "computer", "internet", "video", "music", "movie", "game", "clothes", "shoes",
]

# Practical ASL Execution Guidance Hints for HUD Display
ASL_SIGN_GUIDE = {
    "my": "Flat open palm placed on center of chest",
    "name": "Both hands 'H' shape (index+middle), tap dominant fingers across non-dominant twice",
    "water": "'W' handshape (3 middle fingers upright) tap index finger against chin twice",
    "drink": "'C' handshape mimicking a cup, tilt up towards mouth",
    "eat": "Flat O-handshape (fingertips together) brought to lips repeatedly",
    "idle": "Hands resting on desk/lap or moving naturally (Neutral non-signing state)",
    "food": "Flat O-handshape tapping lips twice (similar to eat)",

    "sleep": "Open hand over face, drawing downward and closing fingers into an 'and' shape while drooping head",
    "hello": "Open flat hand salute moving outward from temple/forehead",
    "goodbye": "Open hand raised waving fingers outward/forward",
    "thank you": "Fingertips of flat hand touch chin/lips, move outward towards camera",
    "please": "Flat open palm rubs chest in a smooth circular motion",
    "sorry": "'A' fist rubs chest in a smooth circular motion",
    "help": "Closed 'A' fist with thumb up placed on open flat non-dominant palm, lift both upward",
    "want": "Both open claw hands palms up, pulling inward towards body",
    "like": "Open hand on chest, thumb and middle finger pull outward while pinching together",
    "yes": "'S' fist nodding up and down like a head nod",
    "no": "Index and middle fingers snap down onto thumb twice",
    "i love you": "Thumb, index, and pinky extended upward (ILY handshape) facing forward",
    "you": "Index finger points directly forward towards camera/person",
    "who": "Index finger near chin/lips wiggling slightly with thumb resting on chin",
    "what": "Both hands open palms up, shaking gently side to side in front of chest",
    "where": "Dominant index finger pointing up, shaking side to side",
    "why": "Fingers touch forehead and pull away transitioning into 'Y' handshape",
    "how": "Backs of curved hands together, rolling hands outward so palms face up",
    "when": "Dominant index finger circles around non-dominant index finger and touches its tip",
    "more": "Both flat O-hands with fingertips touching each other repeatedly in center",
    "stop": "Dominant flat hand chops downward onto open palm of non-dominant hand",
    "again": "Dominant bent hand arches over and taps palm of non-dominant flat hand",
    "fine": "Open '5' hand with thumb touching center of chest",
    "happy": "Open flat hands brush upward against chest with cheerful motion",
    "sad": "Open hands in front of face moving downward while drooping expression",
    "school": "Dominant open flat hand claps down onto non-dominant palm twice",
    "book": "Both flat palms together opening outward like a book",
    "car": "Both fists holding imaginary steering wheel moving up and down",
    "drive": "Both fists steering forward and backward",
    "work": "Dominant fist taps the wrist of non-dominant fist twice",
    "play": "Both 'Y' hands shaking side to side near chest",
    "friend": "Both index fingers hooked together, then flipped and hooked the other way",
    "family": "Both 'F' hands touch index/thumbs in front, circle around until pinkies touch",
    "mother": "Open '5' hand with thumb tapping chin twice",
    "father": "Open '5' hand with thumb tapping forehead twice",
    "baby": "Both arms cradled across chest rocking gently side to side",
    "brother": "'L' hand from forehead comes down onto non-dominant 'L' hand",
    "sister": "'L' hand from chin comes down onto non-dominant 'L' hand",
    "man": "Thumb of '5' hand touches forehead then moves down to touch chest",
    "woman": "Thumb of '5' hand touches chin then moves down to touch chest",
    "teacher": "Both flat O-hands near temples push forward, then flat hands move down (person)",
    "student": "Dominant hand grabs information from non-dominant palm to forehead, then hands move down",
    "house": "Flat hands touch at fingertips (roof), then move down (walls)",
    "home": "Flat O-hand touches corner of mouth then moves to touch upper cheek/ear",
    "bathroom": "'T' handshape (thumb between index/middle) shaking side to side",
    "time": "Dominant index finger taps wrist of non-dominant arm twice",
    "now": "Both 'Y' hands bent downward, dropping simultaneously in front of chest",
    "today": "Sign 'NOW' followed by 'DAY' (or both 'Y' hands bouncing down twice)",
    "yesterday": "'A' or 'Y' thumb touches chin then moves back to touch cheek near ear",
    "tomorrow": "'A' hand with thumb on cheek arcs forward into the air",
    "day": "Dominant arm upright with index pointing up, arcing downward onto non-dominant arm",
    "night": "Dominant bent hand rests over edge of non-dominant horizontal wrist",
    "morning": "Dominant flat hand rises upward from behind non-dominant arm (sunrise)",
    "afternoon": "Dominant flat arm resting at a 45-degree angle over non-dominant arm",
    "coffee": "Both fists on top of each other, top fist grinding in circles",
    "milk": "Dominant fist squeezing repeatedly like milking",
    "tea": "'F' hand dipping and stirring inside 'O' hand (cup)",
    "apple": "Knuckle of 'X' index finger twists against cheek",
    "bread": "Dominant fingertips slice downward on back of non-dominant hand twice",
    "cheese": "Heel of dominant palm twists on heel of non-dominant palm",
    "meat": "Dominant thumb and index pinch the fleshy web of non-dominant hand",
    "pizza": "'Double Z' traced with bent V-hand or 'P' handshape",
    "store": "Both flat O-hands fingertips pointing down, swinging forward/outward twice",
    "buy": "Dominant flat hand on non-dominant palm moves forward handing out money",
    "money": "Dominant flat O-hand taps non-dominant flat palm twice",
    "pay": "Dominant index finger slides forward off flat non-dominant palm",
    "cost": "Dominant 'X' hooked finger swipes down back of non-dominant flat hand",
    "cheap": "Dominant flat hand brushes down across non-dominant palm easily",
    "expensive": "Sign 'MONEY' then throw hand outward opening fingers",
    "big": "Both 'L' or curved hands start close and pull wide apart",
    "small": "Both flat open palms face each other, pressing close together",
    "hot": "Clawed hand at mouth twists quickly outward and away",
    "cold": "Both fists shivering near chest like shivering in cold",
    "good": "Flat hand on chin moves forward onto non-dominant palm",
    "bad": "Flat hand on chin flips downward facing the floor",
    "beautiful": "Open hand circles around entire face and closes into a fist at chin",
    "tired": "Both bent hands on chest drop/slump downward heavily",
    "sick": "Middle finger of dominant hand on forehead, middle of non-dominant on stomach",
    "doctor": "Dominant 'M' or curved fingertips tap non-dominant pulse wrist twice",
    "hospital": "'H' fingers trace a cross on upper non-dominant arm/shoulder",
    "medicine": "Dominant middle finger twists in center of non-dominant palm",
    "hurt": "Both index fingers pointing towards each other twisting back and forth",
    "clean": "Dominant flat palm wipes smoothly across non-dominant palm twice",
    "dirty": "Back of dominant hand under chin with wiggling fingers",
    "understand": "Index finger flicks upward like a lightbulb turning on near forehead",
    "know": "Fingertips of flat hand tap temple/side of forehead",
    "think": "Index finger touches side of forehead",
    "forget": "Flat hand wipes across forehead closing into an 'A' fist",
    "remember": "Dominant 'A' thumb from forehead comes down onto non-dominant 'A' thumb",
    "learn": "Dominant hand scoops knowledge from non-dominant palm up into forehead",
    "read": "Dominant 'V' fingers scan up and down across flat non-dominant palm (page)",
    "write": "Dominant hand holding imaginary pen scribbles across non-dominant palm",
    "sign": "Both index fingers draw alternating backward circles near chest",
    "language": "Both 'L' hands touching thumbs, undulating outward to both sides",
    "walk": "Both flat hands pointing down alternating forward and back like feet walking",
    "run": "Both 'L' hands with index fingers hooked together moving rapidly forward",
    "sit": "Dominant curved 'H' fingers hook over non-dominant curved 'H' fingers",
    "stand": "Dominant 'V' fingers standing upright on non-dominant flat palm",
    "go": "Both index fingers pointing and arching forward away from body",
    "come": "Both index fingers arching inward towards body",
    "see": "'V' fingers near eyes pointing forward",
    "hear": "Index finger pointing to ear",
    "listen": "Cupped 'C' or '3' hand behind ear leaning forward",
    "talk": "'4' fingers tapping chin repeatedly",
    "ask": "Index finger pointing forward bending into 'X' hook towards person",
    "answer": "Both index fingers at chin moving forward and down pointing at person",
    "open": "Both flat hands together palms in, swinging open like doors",
    "close": "Both flat hands apart, swinging together palms meeting",
    "start": "Dominant index finger twists between index and middle fingers of non-dominant hand (turning key)",
    "finish": "Both open '5' hands palms up flick downward to palms out",
    "wait": "Both open hands palms up with fingers wiggling",
    "meet": "Both index fingers pointing up come together and touch knuckles in center",
    "live": "Both 'L' or 'A' hands start at waist and move straight up chest",
    "stay": "Both 'Y' hands push downward firmly",
    "change": "Both 'A' or 'X' fists touch wrists and twist positions",
    "need": "Dominant 'X' hooked finger bent downward repeatedly (same as must/should)",
    "have": "Both bent hands fingertips touch chest",
    "give": "Both flat O-hands start near body and extend forward to other person",
    "take": "Open hand reaches out and grasps backward into a fist towards body",
    "bring": "Both open flat palms facing up moving together from side to front",
    "find": "Open hand reaches down and pinches thumb and index together pulling up",
    "lose": "Both flat O-hands touching at knuckles drop open facing down",
    "feel": "Middle finger bent touching center of chest and sliding upward",
    "hope": "Both hands bent at forehead, nodding downward in sync",
    "love": "Both arms crossed over chest hugging body",
    "hate": "Both hands '8' shape (middle finger to thumb) flicking forward sharply",
    "angry": "Clawed hands near face pulling outward with tense expression",
    "scared": "Both open hands pull in towards chest with trembling motion",
    "surprised": "Both index fingers and thumbs pinch near eyes and snap wide open",
    "bored": "Dominant index finger twists on the tip/side of nose",
    "busy": "Dominant 'B' hand sweeps rapidly back and forth across non-dominant wrist",
    "ready": "Both 'R' hands start crossed and sweep outward simultaneously",
    "same": "'Y' hand moving side to side between two points",
    "different": "Both index fingers crossed at tips, pulling sharply apart",
    "new": "Dominant curved hand scoops across back of non-dominant flat hand",
    "old": "'C' hand at chin pulls downward closing into an 'S' fist (beard)",
    "fast": "Both 'L' hands pull backward snapping into 'A' fists quickly",
    "slow": "Dominant flat hand strokes slowly up the back of non-dominant forearm",
    "hard": "Dominant bent 'V' knuckles strike down onto non-dominant bent 'V' knuckles",
    "easy": "Dominant fingertips brush upward against fingertips of non-dominant hand twice",
    "right": "Dominant '1' fist lands on top of non-dominant '1' fist (correct)",
    "wrong": "'Y' handshape knuckles placed firmly against chin",
    "true": "Dominant index finger upright moves straight forward from lips",
    "false": "Dominant index finger brushes horizontally across tip of nose",
    "always": "Dominant index finger pointing up drawing continuous circles in air",
    "never": "Dominant flat hand draws a question-mark-like flourish downward in air",
    "sometimes": "Dominant index finger bounces upward off non-dominant flat palm several times",
    "maybe": "Both flat open palms facing up weighing up and down alternately",
    "with": "Both 'A' fists apart coming together with knuckles touching",
    "without": "Sign 'WITH' followed by opening both hands and pulling apart",
    "for": "Dominant index finger touches forehead then twists and points forward",
    "from": "Dominant 'X' hooked finger pulls back away from non-dominant upright index finger",
    "about": "Dominant index finger circles around tips of non-dominant pinched O-fingers",
    "all": "Dominant flat hand circles around non-dominant hand and lands on its palm",
    "some": "Dominant flat hand slices across middle of non-dominant flat palm",
    "none": "Both 'O' hands shake outwards to sides",
    "many": "Both closed fists explode upward into open '5' hands",
    "few": "Thumb slides across fingers from index to pinky (curling)",
    "first": "Dominant index finger strikes the upright thumb of non-dominant fist",
    "last": "Dominant pinky strikes down past pinky of non-dominant hand",
    "next": "Dominant flat hand hops over non-dominant flat hand moving forward",
    "before": "Dominant flat hand moves backward towards shoulder",
    "after": "Dominant flat hand moves forward away from non-dominant hand",
    "here": "Both open hands palms up circling gently in front of body",
    "there": "Index finger points outward to a specific location",
    "near": "Dominant curved hand hovers close to back of non-dominant curved hand",
    "far": "Dominant 'A' fist arcs far forward from non-dominant fist",
    "inside": "Dominant fingers tuck down inside non-dominant 'C' cup hand",
    "outside": "Dominant fingers pull outward and up from non-dominant 'C' hand",
    "up": "Index finger points straight up",
    "down": "Index finger points straight down",
    "left": "'L' hand moves to the left",
    "right_dir": "'R' hand moves to the right",
    "front": "Flat hand moves forward in front of face/chest",
    "back": "'A' fist with thumb pointing over shoulder",
    "center": "Dominant bent fingers circle and land in center of non-dominant flat palm",
    "sun": "Index finger draws a circle in air above and opens into bright '5' hand",
    "moon": "'C' hand framing eye looking up at sky",
    "rain": "Both open clawed hands push downward with fluttering fingers",
    "snow": "Both open hands flutter gently downward like falling snowflakes",
    "wind": "Both open hands wave side to side gently or forcefully",
    "dog": "Snap fingers or pat thigh twice",
    "cat": "'F' fingers pinch near cheek pulling outward (whiskers)",
    "bird": "Thumb and index finger at mouth opening and closing like a beak",
    "fish": "Flat hand with thumb up undulating forward like a swimming fish",
    "phone": "'Y' hand placed against ear and mouth",
    "computer": "'C' hand arcs forward along back of horizontal non-dominant forearm",
    "internet": "Middle fingertips of both open hands touching and twisting back and forth",
    "video": "Dominant 'V' fingers move forward from eyes / camera sign",
    "music": "Dominant flat hand waves back and forth over non-dominant forearm (conducting)",
    "movie": "Dominant open hand twists back and forth behind non-dominant flat palm",
    "game": "Both 'A' fists with thumbs up tap knuckles together twice",
    "clothes": "Both open hands brush downward against chest twice",
    "shoes": "Both 'S' fists tap together side by side twice",
}


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


def draw_hand_skeleton(frame, hand_kpts: np.ndarray, scores: np.ndarray, img_w: int, img_h: int, color=(0, 240, 120)) -> None:
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


def main():
    _register_cuda_dll_paths()

    print("=" * 65)
    print("      🎥 LEXIS - INTERACTIVE SIGN DATASET RECORDER")
    print("=" * 65)
    print(f"[INFO]: Total vocabulary target: {len(VOCABULARY_LIST)} signs.")
    print(f"[INFO]: Target frames per sample: {FRAMES_PER_SAMPLE} (~1.5 seconds)")
    print(f"[INFO]: Output directory: {DATA_DIR}")
    print("-" * 65)
    print("CONTROLS:")
    print("  [SPACE]  : Start 3-second countdown and record sign sample")
    print("  [N]      : Jump to NEXT word in vocabulary")
    print("  [P]      : Jump to PREVIOUS word in vocabulary")
    print("  [D]      : Delete most recently recorded sample for this word")
    print("  [Q/ESC]  : Save and Quit recorder")
    print("=" * 65)

    print("\n[INIT]: Initializing RTMPose Wholebody detector...")
    try:
        pose_model = RTMPose(
            Wholebody.MODE["performance"]["pose"],
            model_input_size=Wholebody.MODE["performance"]["pose_input_size"],
            backend="onnxruntime",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("[INIT]: RTMPose Wholebody ready.")
    except Exception as exc:
        print(f"[ERROR]: Could not initialize RTMPose: {exc}")
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR]: Could not open webcam at index {CAMERA_INDEX}.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    word_idx = 0
    state = "IDLE"  # "IDLE", "COUNTDOWN", "RECORDING"
    countdown_start = 0.0
    recorded_frames = []
    status_message = "Press SPACE to start recording"
    status_color = (200, 200, 200)

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                continue

            h, w = raw_frame.shape[:2]
            display_frame = cv2.flip(raw_frame, 1)
            now = time.monotonic()

            current_word = VOCABULARY_LIST[word_idx]
            word_folder = DATA_DIR / current_word
            word_folder.mkdir(parents=True, exist_ok=True)
            existing_samples = list(word_folder.glob("sample_*.npy"))
            num_samples = len(existing_samples)

            # RTMPose Detection
            norm_kpts = None
            try:
                result = pose_model(display_frame, bboxes=[[0, 0, w, h]])
                if result is not None and isinstance(result, tuple) and len(result) == 2:
                    all_keypoints, all_scores = result
                    if all_keypoints is not None and len(all_keypoints) > 0:
                        mean_scores = all_scores.mean(axis=1) if getattr(all_scores, "ndim", 1) > 1 else all_scores
                        best_idx = int(np.argmax(mean_scores))
                        kpts = all_keypoints[best_idx]
                        scores = all_scores[best_idx]

                        norm_kpts = np.zeros((133, 3), dtype=np.float32)
                        norm_kpts[:, 0] = kpts[:, 0] / max(w, 1)
                        norm_kpts[:, 1] = kpts[:, 1] / max(h, 1)
                        norm_kpts[:, 2] = scores

                        # Anatomical sanity check on hand landmarks:
                        # In foreshortened poses (pointing towards camera), if wrist snaps away across chest,
                        # re-anchor it behind the MCP knuckles so the 45-frame sequence tensor stays clean.
                        for hand_slice in [RIGHT_HAND_IDX, LEFT_HAND_IDX]:
                            hwrist = norm_kpts[hand_slice[0], :2]
                            hmcp = np.mean(norm_kpts[[hand_slice[5], hand_slice[9], hand_slice[13], hand_slice[17]], :2], axis=0)
                            wrist_dist = float(np.linalg.norm(hwrist - hmcp))
                            if wrist_dist > 0.18:
                                norm_kpts[hand_slice[0], :2] = hmcp + np.array([0.0, 0.04], dtype=np.float32)

                        # Draw hand skeletons with ghost hand suppression
                        right_scores = scores[RIGHT_HAND_IDX]
                        left_scores = scores[LEFT_HAND_IDX]
                        r_mean = float(np.mean(right_scores))
                        l_mean = float(np.mean(left_scores))

                        # If both hands overlap closely on screen, suppress the weaker ghost duplicate
                        if r_mean > 0.20 and l_mean > 0.20:
                            r_center = np.mean(norm_kpts[RIGHT_HAND_IDX, :2], axis=0)
                            l_center = np.mean(norm_kpts[LEFT_HAND_IDX, :2], axis=0)
                            if np.linalg.norm(r_center - l_center) < 0.08:
                                if r_mean > l_mean:
                                    left_scores = np.zeros_like(left_scores)
                                else:
                                    right_scores = np.zeros_like(right_scores)

                        if np.mean(right_scores) > 0.22:
                            draw_hand_skeleton(display_frame, norm_kpts[RIGHT_HAND_IDX], right_scores, w, h, (0, 240, 120))
                        if np.mean(left_scores) > 0.22:
                            draw_hand_skeleton(display_frame, norm_kpts[LEFT_HAND_IDX], left_scores, w, h, (0, 200, 255))
            except Exception:
                pass

            # State Machine: IDLE -> COUNTDOWN -> RECORDING
            if state == "COUNTDOWN":
                elapsed = now - countdown_start
                remaining = COUNTDOWN_SECONDS - elapsed
                if remaining > 0:
                    status_message = f"GET READY: {int(remaining) + 1}..."
                    status_color = (0, 220, 255)
                    # Overlay big countdown text in center of screen
                    cv2.putText(display_frame, str(int(remaining) + 1), (w // 2 - 40, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 255), 6, cv2.LINE_AA)
                else:
                    state = "RECORDING"
                    recorded_frames = []
                    status_message = f"🔴 RECORDING '{current_word.upper()}' NOW!"
                    status_color = (0, 0, 255)

            elif state == "RECORDING":
                # Collect landmark tensor for this frame
                if norm_kpts is not None:
                    recorded_frames.append(norm_kpts)
                else:
                    recorded_frames.append(np.zeros((133, 3), dtype=np.float32))

                progress_pct = len(recorded_frames) / FRAMES_PER_SAMPLE
                status_message = f"🔴 RECORDING '{current_word.upper()}': {len(recorded_frames)}/{FRAMES_PER_SAMPLE} frames ({progress_pct:.0%})"
                status_color = (0, 0, 255)

                # Progress bar at top
                bar_w = int(w * progress_pct)
                cv2.rectangle(display_frame, (0, 0), (bar_w, 10), (0, 0, 255), -1)

                if len(recorded_frames) >= FRAMES_PER_SAMPLE:
                    # Save sample array: shape (45, 133, 3)
                    sample_array = np.array(recorded_frames, dtype=np.float32)
                    sample_file = word_folder / f"sample_{num_samples + 1:03d}_{int(time.time())}.npy"
                    np.save(sample_file, sample_array)

                    state = "IDLE"
                    status_message = f"✅ Saved sample #{num_samples + 1} for '{current_word}'! Press SPACE to record again or N for next."
                    status_color = (0, 255, 100)
                    print(f"[DATASET]: Saved {sample_file.name} for word '{current_word}' (Total samples: {num_samples + 1})")

            # Draw Header HUD
            overlay = display_frame.copy()
            hud_height = 145
            cv2.rectangle(overlay, (0, 0), (w, hud_height), (18, 18, 18), -1)
            cv2.addWeighted(overlay, 0.82, display_frame, 0.18, 0, display_frame)

            # Line 1: Word & Progress Count
            word_title = f"WORD [{word_idx + 1}/{len(VOCABULARY_LIST)}]:  \"{current_word.upper()}\""
            sample_count_str = f"Samples: {num_samples} recorded"
            cv2.putText(display_frame, word_title, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, sample_count_str, (w - 300, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (100, 255, 100), 2, cv2.LINE_AA)

            # Line 2: ASL Gesture Hint
            hint = ASL_SIGN_GUIDE.get(current_word, "Perform standard ASL sign clearly in front of camera.")
            # Truncate or fit if too long
            if len(hint) > 80:
                hint_display = hint[:77] + "..."
            else:
                hint_display = hint
            cv2.putText(display_frame, f"ASL GUIDE: {hint_display}", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)

            # Line 3: System Status
            cv2.putText(display_frame, f"STATUS: {status_message}", (24, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.56, status_color, 2, cv2.LINE_AA)


            # Draw Bottom Controls Bar
            overlay_b = display_frame.copy()
            cv2.rectangle(overlay_b, (0, h - 45), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay_b, 0.75, display_frame, 0.25, 0, display_frame)
            controls_str = "[SPACE] Record Sample   |   [N] Next Word   |   [P] Prev Word   |   [D] Delete Last   |   [Q] Quit"
            cv2.putText(display_frame, controls_str, (24, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Lexis - Sign Dataset Recorder", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord(" ") and state == "IDLE":
                state = "COUNTDOWN"
                countdown_start = time.monotonic()
            elif key in (ord("n"), ord("N")) and state == "IDLE":
                word_idx = (word_idx + 1) % len(VOCABULARY_LIST)
                status_message = f"Switched to '{VOCABULARY_LIST[word_idx]}'. Press SPACE to record."
                status_color = (200, 200, 200)
            elif key in (ord("p"), ord("P")) and state == "IDLE":
                word_idx = (word_idx - 1) % len(VOCABULARY_LIST)
                status_message = f"Switched to '{VOCABULARY_LIST[word_idx]}'. Press SPACE to record."
                status_color = (200, 200, 200)
            elif key in (ord("d"), ord("D")) and state == "IDLE":
                if existing_samples:
                    latest = sorted(existing_samples)[-1]
                    try:
                        latest.unlink()
                        status_message = f"🗑️ Deleted {latest.name}. Remaining: {len(existing_samples) - 1}"
                        status_color = (0, 160, 255)
                        print(f"[DATASET]: Deleted {latest.name}")
                    except Exception as e:
                        status_message = f"Error deleting: {e}"
                        status_color = (0, 0, 255)
                else:
                    status_message = "No samples to delete for this word."
                    status_color = (0, 160, 255)

        return 0

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[RECORDER]: Session ended.")


if __name__ == "__main__":
    raise SystemExit(main())
