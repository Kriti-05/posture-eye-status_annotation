# 🧍‍♂️ YOLO + MediaPipe Video Analysis

## 📘 Overview
This project provides a **Flask API** for automatic video analysis using **YOLOv11 pose estimation** and **MediaPipe FaceMesh**.  
It detects **human posture** and **eye state**, then annotates each frame with:

- Posture status (**Straight** or **Hunched**)  
- Eye state (**Open** or **Closed**)  
- Skeleton keypoints and bounding box overlays  

---

## ⚙️ Features

| Task | Model Used | Logic |
|------|-------------|--------|
| **Pose Estimation** | YOLOv11 Pose (`yolo11n-pose.pt`) | Detects body keypoints (nose, shoulders, hips) |
| **Posture Detection** | Custom geometric ratio | `ratio = (nose–hip distance) / (shoulder–hip distance)`<br>Compared against baseline (first 3s) → “Straight” or “Hunched” |
| **Eye State Detection** | MediaPipe FaceMesh | Uses detected eye landmarks to crop eyes and classify them using `dima806/closed_eyes_image_detection` |
| **Annotation** | OpenCV | Draws skeleton, keypoints, and text (Posture + Eye State) on frames |
| **API Hosting** | Flask | `/analyze` endpoint accepts a video and returns annotated output |

---

## 🧩 How Posture Is Determined

- YOLO provides body keypoints such as **nose**, **shoulders**, and **hips**.
- A **baseline ratio** is calculated from the **first 3 seconds** of the video (assumed to be straight posture).
- During analysis:
  ```python
  ratio = abs(nose[1] - mid_hip[1]) / abs(mid_shoulder[1] - mid_hip[1])
  posture = "Straight" if ratio >= baseline_ratio * 0.9 else "Hunched"
- If the current head height is less than 90% of the baseline (i.e., the nose is closer to the hips), posture is marked as Hunched.


## 👁️ Eye Detection Logic (MediaPipe + Hugging Face)

- MediaPipe FaceMesh extracts facial landmarks, including eye regions.
- Cropped eye images are classified by the Hugging Face model
dima806/closed_eyes_image_detection.
- The model predicts each eye as OpenEye or CloseEye. If both are closed → overall status = Closed, otherwise Open.

## Run the flask server

- Install all dependencies directly
  pip install flask ultralytics mediapipe opencv-python torch torchvision torchaudio pillow numpy transformers

- Run the flask server
  python app.py

- Test the API end point
  curl -X POST -F "video=@<video_name>.mp4" http://127.0.0.1:5000/analyze


