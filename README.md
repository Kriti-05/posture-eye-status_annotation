# YOLO + MediaPipe Video Analysis

## Overview
This project provides a **Flask API** for video analysis using **YOLOv11 pose detection** and **MediaPipe FaceMesh**. The API extracts:

- **Posture**: Straight or Hunched (based on nose-to-hip ratio)
- **Eye State**: Open or Closed (based on Eye Aspect Ratio)
- Annotates the input video with posture, eye state, and skeleton points.

**Libraries/Models Used:**

- `Ultralytics YOLO` (`yolo11n-pose.pt`) for human pose estimation
- `MediaPipe FaceMesh` for eye detection and facial landmarks
- `OpenCV` for video processing and annotation
- `Flask` for API service

---

## Setup Instructions

1. Clone the repository:
 
git clone https://github.com/<your-username>/yolo-mediapipe-video-analysis.git
cd yolo-mediapipe-video-analysis

2. Create a virtual environment:
python -m venv venv

3. Activate the virtual environment:
For windows
venv\Scripts\activate

4. Install dependencies:
pip install -r requirements.txt

---

## Running the Flask Server
python app.py

---

## Test API endpoint
curl -X POST -F "video=@<video name>.mp4" http://127.0.0.1:5000/analyze

---

## Notes
- Only .mp4 videos are supported.
- Baseline posture is calculated from the first 3 seconds of the video.
- Annotated frames include posture and eyes status.
