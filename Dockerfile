FROM python:3.9-slim

# Install system dependencies including FFmpeg and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Update pip and install PyTorch with CUDA 11.7 support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code into the container
COPY . .

# Expose the port used by FastAPI server
EXPOSE 8080

# Command to run the FastAPI application with Uvicorn server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]




# FROM python:3.9-slim

# # Install system dependencies including FFmpeg and OpenCV dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     python3-dev \
#     ffmpeg \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory inside the container
# WORKDIR /app

# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code into the container
# COPY . .

# # Expose the port used by FastAPI server (8080)
# EXPOSE 8080

# # Command to run the FastAPI application with Uvicorn server
# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]




# FROM python:3.9-slim

# # Install system dependencies including FFmpeg and OpenCV dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     python3-dev \
#     ffmpeg \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*

# # Set working directory inside the container
# WORKDIR /app

# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code into the container
# COPY . .

# # Expose the port used by FastAPI server
# EXPOSE 8000

# # Command to run the FastAPI application with Uvicorn server
# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
