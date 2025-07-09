# /optimize_threshold_rl.py
# This script uses Q-Learning to find the optimal confidence threshold
# by processing the 'test_clip.mp4' file.

import numpy as np
import cv2
from ultralytics import YOLO
import random
import os

# --- CONFIGURATION ---
MODEL_PATH = "models/best.pt"
VIDEO_PATH = "test_clip.mp4" # <-- NOW USES THE NEW TEST CLIP
ROI_COORDS = (950, 50, 1500, 330)

# --- RL Q-Learning Parameters ---
STATES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
ACTIONS = [0, 1]
ALPHA, GAMMA, EPSILON, EPISODES = 0.1, 0.6, 0.1, 50
q_table = np.zeros([len(STATES), len(ACTIONS)])

def get_reward(detections_count):
    if 2 < detections_count < 15: return 10
    elif detections_count == 0: return -20
    else: return -abs(detections_count - 8)

def run_rl_optimization():
    print("--- Starting RL Optimization ---")
    if not all(os.path.exists(p) for p in [MODEL_PATH, VIDEO_PATH]):
        print(f"FATAL ERROR: Missing required files. Ensure '{MODEL_PATH}' and '{VIDEO_PATH}' exist.")
        print("Run 'python split.py' first to create the test clip.")
        return None

    model = YOLO(MODEL_PATH)
    print("Model loaded.")

    for i in range(1, EPISODES + 1):
        state_index = random.randint(0, len(STATES) - 1)
        total_reward = 0
        cap = cv2.VideoCapture(VIDEO_PATH)
        while True:
            success, frame = cap.read()
            if not success: break
            action = random.choice(ACTIONS) if random.uniform(0, 1) < EPSILON else np.argmax(q_table[state_index])
            next_state_index = max(0, state_index - 1) if action == 0 else min(len(STATES) - 1, state_index + 1)
            current_threshold = STATES[next_state_index]
            frame_roi = frame[ROI_COORDS[1]:ROI_COORDS[3], ROI_COORDS[0]:ROI_COORDS[2]]
            results = model.predict(frame_roi, conf=current_threshold, verbose=False)
            reward = get_reward(len(results[0].boxes))
            total_reward += reward
            old_value = q_table[state_index, action]
            next_max = np.max(q_table[next_state_index])
            q_table[state_index, action] = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
            state_index = next_state_index
        cap.release()
        if i % 10 == 0: print(f"Episode: {i}, Total Reward: {total_reward}")

    best_state_index = np.argmax(np.mean(q_table, axis=1))
    optimal_threshold = STATES[best_state_index]
    print(f"\n✅ Optimal Confidence Threshold found: {optimal_threshold}")
    return optimal_threshold

if __name__ == "__main__":
    best_threshold = run_rl_optimization()
    if best_threshold:
        print("\n--- RECOMMENDATION ---")
        print(f"Please update the 'CONF_THRESHOLD' in 'test.py' or 'src/process_video.py' to {best_threshold}.")