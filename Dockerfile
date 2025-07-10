# /Dockerfile
# Use an official Python runtime as a parent image.
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed by OpenCV for GUI elements
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the application when the container starts
# CORRECTED to use web.py
CMD ["streamlit", "run", "src/web.py", "--server.port", "8501", "--server.address", "0.0.0.0"]