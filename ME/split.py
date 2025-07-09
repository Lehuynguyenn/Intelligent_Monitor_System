# /split.py
# VERSION 2: Uses sequential sampling for robust clip creation.

import cv2
import os

# --- CONFIGURATION ---
INPUT_VIDEO_PATH = "dispatch_video.mp4"
OUTPUT_VIDEO_PATH = "test_clip.mp4"  # The name of our single test video
TARGET_DURATION_SECONDS = 600  # 10 minutes = 600 seconds
# --------------------

def create_sequential_test_clip():
    print("--- Starting Test Clip Creator (Sequential Sampling Mode) ---")

    # Check if the input video exists
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"FATAL ERROR: Main video not found at '{INPUT_VIDEO_PATH}'.")
        return

    # Open the video file for reading
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"FATAL ERROR: Could not open video file at '{INPUT_VIDEO_PATH}'.")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    source_duration = total_frames / fps if fps > 0 else 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    if total_frames == 0 or fps == 0:
        print("FATAL ERROR: Could not read video properties.")
        cap.release()
        return

    print(f"Source Video Info: {total_frames} frames, {fps:.2f} FPS, ~{source_duration:.0f} seconds long.")

    # Determine the frame skipping interval
    # We want to select enough frames to make a video of TARGET_DURATION_SECONDS
    # The keep_ratio determines what fraction of frames we need to keep.
    keep_ratio = TARGET_DURATION_SECONDS / source_duration if source_duration > 0 else 1

    if keep_ratio >= 1:
        print("Warning: Target duration is longer than or equal to source video. Copying the whole video.")
        # In this case, we just copy the file. It's faster.
        import shutil
        shutil.copyfile(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)
        print(f"✅ Full video copied to '{OUTPUT_VIDEO_PATH}'.")
        return
    
    # The interval at which we will save a frame.
    # e.g., if keep_ratio is 0.33, interval will be ~3, so we keep 1 of every 3 frames.
    frame_interval = int(1 / keep_ratio)
    print(f"Will keep 1 of every {frame_interval} frames to create a {TARGET_DURATION_SECONDS}-second clip.")

    # --- Video Writer Setup ---
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, codec, fps, (frame_width, frame_height))
    print(f"Creating test clip at '{OUTPUT_VIDEO_PATH}'...")

    frame_count = 0
    frames_written = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Check if this frame is one we should keep
        if frame_count % frame_interval == 0:
            writer.write(frame)
            frames_written += 1
            if frames_written % int(fps * 10) == 0: # Print progress every 10 seconds of video written
                 print(f"  ...wrote {frames_written} frames")
        
        frame_count += 1

    # --- Cleanup ---
    cap.release()
    writer.release()
    print(f"\n✅ Test clip creation complete. Wrote {frames_written} frames.")
    print(f"   Output saved to '{OUTPUT_VIDEO_PATH}'.")

if __name__ == "__main__":
    create_sequential_test_clip()