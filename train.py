    # /train.py
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
DATA_CONFIG_PATH = 'config/data.yaml'
PREVIOUS_MODEL_PATH = 'models/best.pt'
EPOCHS = 150
IMAGE_SIZE = 640

def main():
    print("--- Starting Model Training ---")

    if not os.path.exists(PREVIOUS_MODEL_PATH):
        print(f"Warning: Previous model not found at '{PREVIOUS_MODEL_PATH}'.")
        print("Starting training from a standard pre-trained model ('yolov8n.pt').")
        load_model_path = 'yolov8n.pt'
    else:
        print(f"Loading model from previous training run: '{PREVIOUS_MODEL_PATH}'")
        load_model_path = PREVIOUS_MODEL_PATH

    model = YOLO(load_model_path)

    print("\nStarting training with enhanced augmentations...\n")

    model.train(
        data=DATA_CONFIG_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        lr0=0.001,
        lrf=0.01,
        copy_paste=0.5,
        mixup=0.1,
        mosaic=1.0,
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        shear=5.0,
        perspective=0.001,
        flipud=0.1,
        fliplr=0.5,
    )

    print("\n--- Training Complete ---")
    print("The best model has been saved to the 'runs/detect/train/weights/' directory.")

if __name__ == "__main__":
    main()