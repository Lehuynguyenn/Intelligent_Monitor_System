# /Dockerfile

# Start with a slim, official Python base image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies required by OpenCV
# This prevents common errors when running graphics libraries in a headless environment.
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy the requirements file first to leverage Docker's build cache.
# If requirements.txt doesn't change, Docker won't re-install packages on every build.
COPY requirements.txt ./

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container's working directory.
# The .dockerignore file will prevent large folders from being copied.
COPY . .

# Set the default command to run when the container starts.
# This will execute your main processing script.
CMD ["python", "src/web.py"]