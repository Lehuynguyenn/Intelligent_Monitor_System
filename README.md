# Intelligent Dispatch Monitoring System

This is my very first project on GitHub as well as firstly related AI.
If there are something went wrong, please help me to fix it or give feedback.

## Project Structure

The project is organized into distinct directories for configuration, models, and source code, following best practices for maintainability.

```
Intelligent_Monitor_System/
├── .streamlit
├── .venv
├── config/
│   └── data.yaml           
├── Dataset/       
│   ├── Classification      
│   └── Detection  
│       ├── train
│       ├── val
│       └── dataset.yaml
├── feedback_inbox
├── Miscellaneous/
│   └── run/detect/train/...
│   # --- Development & Utility Scripts ---
│   ├── train.py                  
│   ├── test.py                  
│   ├── split.py                  
│   ├── optimize_threshold_rl.py  
│   └── ...    
├── models/
│   └── best.pt             
├── src/
│   └── scripts  
│       └── web.py  
├── .dockerignore
├── .gitattributes        
├── .gitignore     
├── dispatch_video.mp4     
├── docker-compose.yml  
├── Dockerfile                
├── output_video_local_test.mp4               
├── README.md            
└──  requirements.txt       
```

---

## Running with Docker

This method builds and runs the application ensured a consistent environment.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Lehuynguyenn/Intelligent_Monitor_System.git
    cd Intelligent_Monitor_System
    ```

2.  **Build and Run with Docker Compose:**
    This single command builds the Docker image and starts the web service.
    ```bash
    docker compose up --build
    ```
    - The `--build` flag is only needed the first time. For subsequent runs, `docker compose up` is sufficient.

3.  **View the Application:**
    Once the server starts, open your web browser and navigate to:
    ```
    http://localhost:8501
    ```
4.  **Stop the Application:**
    To stop the server, go back to your terminal and press
    ```
    Ctrl+C
    ```

---

## Development & Testing

For development and debugging, you can run the project directly on your local machine.

1.  **Setup Local Environment:**
    ```bash
    # Create and activate a Python virtual environment
    python3 -m venv .venv
    source .venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```
### Choose a Testing Method:

#### Method A: Interactive Web UI (Recommended for Visualization)

This method launches a web application to view the model's performance in real-time.

1.  **Run the Interactive Web UI:**
    This is the best method for local testing and visualization.
    ```bash
    streamlit run src/web.py
    ```
2.  **View the Application:**
    Once the server starts, open your web browser and navigate to:
    ```
    http://localhost:8501
    ```
3.  **Stop the Application:**
    To stop the server, go back to your terminal and press
    ```
    Ctrl+C
    ```

#### Method B: Fast Batch Processing Test

This method is for quickly testing the end-to-end processing pipeline on a smaller video clip.

1.  **Create a Test Clip:**
    First, run the splitting script to create a 10-minute test video.
    ```bash
    python split.py
    ```
    This will create a `test_clip.mp4` file in your project root.

2.  **Run the Local Test Script:**
    This script processes the shorter `test_clip.mp4`.
    ```bash
    python test.py
    ```
    The output will be saved as `output_video_local_test.mp4`.
