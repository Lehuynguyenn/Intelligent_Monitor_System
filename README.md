# Dispatch Monitoring System

This project is an intelligent monitoring system that processes a video of a commercial kitchen's dispatch area. It uses a fine-tuned YOLOv8 model to perform object detection and tracking, and saves the result as a new video file with all detections rendered.

## Features

-   **Object Tracking:** Detects and tracks 6 distinct classes of items.
-   **Region of Interest (ROI):** Processing is focused on a specific counter area for efficiency.
-   **Kalman Filter Smoothing:** Implements Kalman Filters to produce smoother, more stable bounding box movements.
-   **Batch Processing:** The application reads an input video and generates a new, annotated output video file.
-   **Dockerized Deployment:** The entire application is containerized using Docker and managed with Docker Compose for easy, one-command execution.

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

## Setup and Installation

This project is designed to be run with Docker, eliminating complex local setup.

### Prerequisites

-   [Docker](https://www.docker.com/get-started) installed and running.
-   [Git](https://git-scm.com/) installed on your system.
-   (Recommended) [Git LFS](https://git-lfs.com) for handling large model files.

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-github-repository-url>
    cd Intelligent_Monitor_System

    # If using Git LFS, pull the large model files
    # git lfs pull
    ```

2.  **Place Required Files:**
    *   Ensure your trained `best.pt` model is inside the `models/` folder.
    *   Ensure your input `dispatch_video.mp4` file is in the project's root directory.

## Usage

Use Docker Compose to build the image and run the processing script with a single command.

1.  **Navigate to the project root directory** in your terminal.

2.  **Run the application:**
    ```bash
    docker compose up --build
    ```
    - The `--build` flag is only necessary the first time you run it, or after you make changes to the code or `requirements.txt`.
    - This command will build the Docker image and then start the container. The script `src/process_video.py` will execute automatically.

3.  **Monitor Progress:**
    You will see log output in your terminal indicating the progress of the video processing, frame by frame.

4.  **Find the Result:**
    Once the script is finished, the container will stop. A new file named `output_video.mp4` will be created in the project's root directory on your local machine. You can open this file to see the final result with all detections and tracking IDs.