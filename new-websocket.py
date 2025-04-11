import os
import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from collections import deque
from torch import nn

# Initialize FastAPI app
app = FastAPI()

# CORS configuration for Kubernetes ingress compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and device initialization
detector = None
pose_estimator = None
lstm_model = None
device = None

BUFFER_SIZE = 30  # Number of frames to buffer for LSTM processing
DYSPLASIA_THRESHOLD = 0.65  # Dysplasia detection threshold


@app.on_event("startup")
async def load_models():
    """
    Load ML models at startup.
    """
    global detector, pose_estimator, lstm_model, device

    # Use GPU if available; otherwise, fallback to CPU
    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Load YOLOv8 detection model
    detector = YOLO('yolov8n.pt').to(device).eval()

    # Load YOLOv8 pose estimation model
    pose_estimator = YOLO('yolov8n-pose.pt').to(device).eval()

    # Define LSTM model for dysplasia detection
    class DysplasiaLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=51, hidden_size=64, batch_first=True)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
                torch.nn.Sigmoid()
            )

        def forward(self, x):
            x, _ = self.lstm(x)
            return self.classifier(x[:, -1, :])

    lstm_model = DysplasiaLSTM()

    # Load pre-trained LSTM weights
    state_dict = torch.load('canine_dysplasia_lstm.pt', map_location=device)
    
    # Adjust state_dict keys if necessary (e.g., rename layers)
    adjusted_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('classifier.3', 'classifier.2')
        adjusted_state_dict[new_key] = value

    lstm_model.load_state_dict(adjusted_state_dict)
    lstm_model.to(device).eval()


@app.websocket("/ws/process-video")
async def process_video_stream(websocket: WebSocket):
    """
    Handle real-time video processing via WebSocket.
    """
    await websocket.accept()
    
    pose_buffer = deque(maxlen=BUFFER_SIZE)
    max_prob = 0.0

    try:
        while True:
            # Receive frame from client (as binary data)
            frame_bytes = await websocket.receive_bytes()

            # Decode the frame (assumes JPEG format)
            frame_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                raise HTTPException(status_code=400, detail="Invalid frame received")

            # Resize and convert frame to RGB format for YOLOv8 processing
            frame_resized = cv2.resize(frame, (640, 640))
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Perform dog detection using YOLOv8 detection model
            results = detector(rgb_frame)
            boxes = results.xyxy[0].cpu().numpy()
            
            dog_boxes = [box for box in boxes if int(box[5]) == 16]  # COCO class 16: Dog

            for box in dog_boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                dog_roi = rgb_frame[y1:y2, x1:x2]

                if dog_roi.size == 0:
                    continue

                # Perform keypoint extraction using YOLOv8 pose estimation model
                pose_results = pose_estimator(dog_roi)
                if pose_results.xykpt[0] is not None:
                    kps = pose_results.xykpt[0].cpu().numpy().flatten()

                    if kps.shape == (51,):
                        pose_buffer.append(kps)

                    # Pad buffer if necessary
                    while len(pose_buffer) < BUFFER_SIZE:
                        pose_buffer.append(np.zeros(51))

                    # Perform dysplasia prediction when buffer is full
                    if len(pose_buffer) == BUFFER_SIZE:
                        seq_tensor = torch.tensor(np.array(pose_buffer), dtype=torch.float32).unsqueeze(0).to(device)

                        with torch.no_grad():
                            prob = lstm_model(seq_tensor).item()

                        max_prob = max(max_prob, prob)

                        diagnosis = "Dysplasia Detected" if max_prob > DYSPLASIA_THRESHOLD else "Normal"
                        
                        # Send results back to client via WebSocket
                        await websocket.send_json({
                            "status": "success",
                            "dysplasia_probability": round(max_prob * 100, 2),
                            "diagnosis": diagnosis,
                        })
    
    except Exception as e:
        await websocket.close(code=1006)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
