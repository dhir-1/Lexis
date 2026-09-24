import sys
from pathlib import Path
import json

# Add venv site packages
sys.path.insert(0, r"c:\Users\dhira\Desktop\Lexis\backend\venv\Lib\site-packages")
import numpy as np

DATA_DIR = Path(r"c:\Users\dhira\Desktop\Lexis\backend\data\user_recorded_signs")

NOSE_IDX = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)

def analyze_sample(seq):
    # seq shape: (45, 133, 3)
    T = seq.shape[0]
    
    # Check hands presence
    r_scores = seq[:, RIGHT_HAND, 2]
    l_scores = seq[:, LEFT_HAND, 2]
    r_mean = float(np.mean(r_scores))
    l_mean = float(np.mean(l_scores))
    
    active_hand = "Right" if r_mean > l_mean else "Left"
    two_handed = (r_mean > 0.35 and l_mean > 0.35)
    
    # Hand slice of primary hand
    h_slice = RIGHT_HAND if active_hand == "Right" else LEFT_HAND
    h_kpts = seq[:, h_slice, :2] # (T, 21, 2)
    
    # Body positions
    nose_y = np.mean(seq[:, NOSE_IDX, 1])
    chest_y = np.mean((seq[:, LEFT_SHOULDER, 1] + seq[:, RIGHT_SHOULDER, 1]) / 2.0)
    wrist_y = np.mean(h_kpts[:, 0, 1]) # 0 is wrist
    wrist_x = np.mean(h_kpts[:, 0, 0])
    
    # Height zone
    if wrist_y < nose_y + 0.05:
        zone = "Head/Face"
    elif wrist_y < chest_y + 0.10:
        zone = "Chest/Chin"
    elif wrist_y < 0.85:
        zone = "Mid-Torso/Front"
    else:
        zone = "Lap/Low"
        
    # Finger extension on primary hand (last 15 frames)
    recent_hand = np.mean(h_kpts[-15:], axis=0) # (21, 2)
    hwrist = recent_hand[0]
    
    # Tips: 4 (thumb), 8 (index), 12 (mid), 16 (ring), 20 (pinky)
    # MCPs: 2 (thumb), 5 (index), 9 (mid), 13 (ring), 17 (pinky)
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    names = ["Thumb", "Index", "Mid", "Ring", "Pinky"]
    
    extended_fingers = []
    for t_idx, m_idx, fname in zip(tips, mcps, names):
        dist_tip = np.linalg.norm(recent_hand[t_idx] - hwrist)
        dist_mcp = np.linalg.norm(recent_hand[m_idx] - hwrist)
        if dist_tip > dist_mcp * 1.15:
            extended_fingers.append(fname)
            
    # Motion Dynamics
    wrist_pts = h_kpts[:, 0, :] # (T, 2)
    step_diffs = np.linalg.norm(wrist_pts[1:] - wrist_pts[:-1], axis=1)
    total_movement = float(np.sum(step_diffs))
    net_displacement = float(np.linalg.norm(wrist_pts[-1] - wrist_pts[0]))
    straightness = net_displacement / (total_movement + 1e-5)
    
    if total_movement < 0.12:
        motion_type = "Static / Stationary"
    elif straightness < 0.40:
        motion_type = "Circular / Rubbing"
    elif net_displacement > 0.15:
        # Check direction of movement
        dy = wrist_pts[-1, 1] - wrist_pts[0, 1]
        dx = wrist_pts[-1, 0] - wrist_pts[0, 0]
        if dy < -0.10:
            motion_type = "Upward Motion"
        elif dy > 0.10:
            motion_type = "Downward Motion"
        elif abs(dx) > 0.10:
            motion_type = "Lateral Motion"
        else:
            motion_type = "Dynamic Movement"
    else:
        motion_type = "Small Gesture"
        
    return {
        "active_hand": "Both" if two_handed else active_hand,
        "confidence": round(float(max(r_mean, l_mean)), 2),
        "zone": zone,
        "fingers": extended_fingers if extended_fingers else ["Fist/Closed"],
        "motion": motion_type,
        "movement_dist": round(total_movement, 3),
        "straightness": round(straightness, 2)
    }

def main():
    word_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    report = {}
    
    for wd in word_dirs:
        samples = list(wd.glob("sample_*.npy"))
        if not samples:
            continue
            
        sample_results = []
        for s in samples:
            try:
                arr = np.load(s)
                sample_results.append(analyze_sample(arr))
            except Exception as e:
                pass
                
        if not sample_results:
            continue
            
        # Summary for word
        hands = [sr["active_hand"] for sr in sample_results]
        zones = [sr["zone"] for sr in sample_results]
        motions = [sr["motion"] for sr in sample_results]
        confs = [sr["confidence"] for sr in sample_results]
        fingers_list = [", ".join(sr["fingers"]) for sr in sample_results]
        
        report[wd.name] = {
            "num_samples": len(sample_results),
            "dominant_hand": max(set(hands), key=hands.count),
            "dominant_zone": max(set(zones), key=zones.count),
            "dominant_motion": max(set(motions), key=motions.count),
            "typical_fingers": max(set(fingers_list), key=fingers_list.count),
            "mean_confidence": round(float(np.mean(confs)), 2),
            "consistency": "High" if len(set(zones)) == 1 and len(set(hands)) == 1 else "Medium"
        }
        
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
