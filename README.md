# Dispatch Monitoring System

This project is an intelligent monitoring system that processes a video of a commercial kitchen's dispatch area. It uses a fine-tuned YOLOv8 model to perform object detection and tracking, and saves the result as a new video file with all detections rendered.

The entire application is containerized using Docker for one-command execution and guaranteed reproducibility.

## Features

-   **Object Tracking:** Detects and tracks 6 distinct classes of items.
-   **Region of Interest (ROI):** Processing is focused on a specific counter area for efficiency.
-   **Kalman Filter Smoothing:** Produces exceptionally smooth and stable bounding box movements.
-   **Batch Processing:** Reads an input video and generates a new, fully annotated output video file.
-   **Dockerized Deployment:** Managed with Docker Compose for easy, one-command setup and execution.

## Technology Stack

-   **Model:** YOLOv8 (fine-tuned with PyTorch)
-   **Core Libraries:** OpenCV, PyTorch, Ultralytics, NumPy
-   **Deployment:** Docker & Docker Compose

## Project Structure

```
Intelligent_Monitor_System/
├── models/
│   └── best.pt             # Trained PyTorch model
├── src/
│   └── process_video.py    # Main video processing script
├── dispatch_video.mp4        # Input video file
├── Dockerfile                # Defines the Docker container
├── docker-compose.yml        # Manages the Docker service
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## How to Run This Project

This project is designed to be run with Docker. No local Python environment setup is needed.

### Prerequisites

-   [Docker](https://www.docker.com/get-started) installed and running.
-   [Git](https://git-scm.com/) installed.

### Execution Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-github-repository-url>
    cd Intelligent_Monitor_System
    # If using Git LFS, pull the large model/video files
    # git lfs pull
    ```

2.  **Place Required Files:**
    *   Ensure your trained `best.pt` model is inside the `models/` folder.
    *   Ensure your input `dispatch_video.mp4` file is in the project's root directory.

3.  **Run the Application:**
    Navigate to the project root directory in your terminal and execute the following command:
    ```bash
    docker compose up --build
    ```
    - The `--build` flag is only necessary the first time, or after you make changes to the code or `requirements.txt`.
    - This command will build the Docker image and then start the container. The script `src/process_video.py` will run automatically.

4.  **Monitor Progress & Find the Result:**
    You will see log output in your terminal indicating the progress of the video processing. Once the script is finished, the container will stop.

    A new file named **`output_video.mp4`** will be created in your project's root directory. You can open this file to see the final result.