# /src/app.py
import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict
import os
import time

# --- Configuration ---
MODEL_PATH = "models/best.pt"
VIDEO_PATH = "dispatch_video.mp4"
ROI_COORDS = (930, 58, 1500, 307)
CONF_THRESHOLD = 0.35
FEEDBACK_DIR = "feedback_inbox"

os.makedirs(FEEDBACK_DIR, exist_ok=True)

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

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Dispatch Monitoring System")
st.title("👨‍🍳 Dispatch Area - Intelligent Monitoring System")
st.write("This application uses a fine-tuned YOLOv8 model to track items in real-time.")

# --- Model Loading (with caching) ---
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please download the trained 'best.pt' and place it in the 'models/' directory.")
        return None
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print("Model loaded successfully.")
    return model

model = load_model(MODEL_PATH)

# --- Main Application Logic ---
col1, col2 = st.columns([3, 1])
with col1:
    st.header("Live Video Feed")
    image_placeholder = st.empty()
with col2:
    st.header("Controls & Feedback")
    if st.button("🚩 Flag Frame for Review"):
        st.session_state.flag_frame = True
        st.success("Frame will be saved for review!")
        time.sleep(2)

# --- Video Processing ---
if model:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file at {VIDEO_PATH}")
    else:
        kalman_filters = defaultdict(lambda: [KalmanFilter(1e-4, 0.1) for _ in range(4)])
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if 'flag_frame' in st.session_state and st.session_state.flag_frame:
                filename = os.path.join(FEEDBACK_DIR, f"flagged_frame_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)
                st.session_state.flag_frame = False

            roi_x1, roi_y1, roi_x2, roi_y2 = ROI_COORDS
            frame_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            results = model.track(frame_roi, persist=True, verbose=False)

            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
            cv2.putText(frame, "ROI", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

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
            
            # New, corrected line
            image_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            time.sleep(0.01)
        cap.release()