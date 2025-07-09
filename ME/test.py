import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict
import os
import time

# --- Configuration Constants ---
MODEL_PATH = "models/best.pt"
INPUT_VIDEO_PATH = "test_clip.mp4"  # <-- NOW USES THE NEW TEST CLIP
OUTPUT_VIDEO_PATH = "output_video_local_test.mp4"
ROI_COORDS = (900, 50, 1500, 330)
CONF_THRESHOLD = 0.35 # Use the value you found from optimization

# --- Kalman Filter Class (Self-contained) ---
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimate=0.0):
        self.process_variance, self.measurement_variance = process_variance, measurement_variance
        self.estimate, self.error_covariance = estimate, 1.0

    def update(self, measurement):
        predicted_estimate = self.estimate
        predicted_error_covariance = self.error_covariance + self.process_variance
        kalman_gain = predicted_error_covariance / (predicted_error_covariance + self.measurement_variance)
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.error_covariance = (1 - kalman_gain) * predicted_error_covariance
        return self.estimate

# --- Main Testing Logic ---
def run_test():
    print("--- Starting Local Test ---")
    start_time = time.time()

    if not all(os.path.exists(p) for p in [MODEL_PATH, INPUT_VIDEO_PATH]):
        print(f"FATAL ERROR: Missing required files. Ensure '{MODEL_PATH}' and '{INPUT_VIDEO_PATH}' exist.")
        print("Run 'python split.py' first to create the test clip.")
        return

    model = YOLO(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded.")

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    print(f"Processing '{INPUT_VIDEO_PATH}' ({total_frames} frames)...")

    kalman_filters = defaultdict(lambda: [KalmanFilter(1e-4, 0.1) for _ in range(4)])
    frame_num = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Finished processing all frames.")
            break

        frame_num += 1
        print(f"Processing frame {frame_num}/{total_frames}...")

        roi_x1, roi_y1, roi_x2, roi_y2 = ROI_COORDS
        frame_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        results = model.track(frame_roi, persist=True, verbose=False)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        
        if results[0].boxes.id is not None:
            for box, track_id, conf, cls_id in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.id.int().cpu().tolist(), results[0].boxes.conf.cpu().tolist(), results[0].boxes.cls.cpu().tolist()):
                if conf < CONF_THRESHOLD: continue
                x1, y1, x2, y2 = box
                kf_x1, kf_y1, kf_x2, kf_y2 = kalman_filters[track_id]
                x1_s, y1_s, x2_s, y2_s = kf_x1.update(x1), kf_y1.update(y1), kf_x2.update(x2), kf_y2.update(y2)
                draw_x1, draw_y1 = int(x1_s + roi_x1), int(y1_s + roi_y1)
                draw_x2, draw_y2 = int(x2_s + roi_x1), int(y2_s + roi_y1)
                label = f"ID {track_id}: {model.names[int(cls_id)]} {conf:.2f}"
                cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (draw_x1, draw_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        writer.write(frame)

    end_time = time.time()
    cap.release()
    writer.release()
    print(f"\n✅ Local testing complete in {end_time - start_time:.2f} seconds.")
    print(f"   Output saved to '{OUTPUT_VIDEO_PATH}'.")

if __name__ == "__main__":
    run_test()