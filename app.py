from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import os
import time   # ⏱️ Added

# -------------------- CONFIG --------------------
app = Flask(__name__)

# YOLO pose model (for posture)
model = YOLO("yolo11n-pose.pt")

# Hugging Face model for open/closed eye classification
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("dima806/closed_eyes_image_detection")
eye_model = AutoModelForImageClassification.from_pretrained("dima806/closed_eyes_image_detection").to(device)
eye_model.eval()

# Mediapipe for face landmarks
mp_face = mp.solutions.face_mesh

# Output folders setup
output_folder = os.path.join("Downloads", "annotated_videos")
os.makedirs(output_folder, exist_ok=True)
annotated_video_path = os.path.join(output_folder, "annotated_video.mp4")

# Folder for annotated frames
frames_folder = os.path.join("Downloads", "annotated_frames")
os.makedirs(frames_folder, exist_ok=True)

# ------------------------------------------------
@app.route("/")
def home():
    return "YOLO + dima806 Eye Detection + MediaPipe Server Running!"

# ------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze_video():
    start_time = time.time()   # ⏱️ Start timer

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    # Save input
    video_file = request.files["video"]
    input_path = os.path.join(output_folder, "input_video.mp4")
    video_file.save(input_path)

    # Video setup
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

    # Compute baseline posture (first 3 seconds)
    baseline_frames = int(fps * 3)
    baseline_ratios = []
    for i in range(baseline_frames):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, verbose=False)
        for r in results:
            kpts = r.keypoints.xy.cpu().numpy()
            if len(kpts) > 0:
                kp = kpts[0]
                nose = kp[0]
                shoulders = kp[5:7]
                hips = kp[11:13]
                mid_shoulder = np.mean(shoulders, axis=0)
                mid_hip = np.mean(hips, axis=0)
                ratio = abs(nose[1] - mid_hip[1]) / abs(mid_shoulder[1] - mid_hip[1])
                baseline_ratios.append(ratio)

    baseline_ratio = np.mean(baseline_ratios) if baseline_ratios else 1.0

    # Rewind video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    results_json = []

    # FaceMesh for landmark detection
    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            posture, eyes_status = "Unknown", "Unknown"

            # ------------- YOLO POSTURE DETECTION -------------
            results = model.predict(frame, verbose=False)
            for r in results:
                kpts = r.keypoints.xy.cpu().numpy()
                if len(kpts) > 0:
                    kp = kpts[0]
                    nose = kp[0]
                    shoulders = kp[5:7]
                    hips = kp[11:13]
                    mid_shoulder = np.mean(shoulders, axis=0)
                    mid_hip = np.mean(hips, axis=0)
                    ratio = abs(nose[1] - mid_hip[1]) / abs(mid_shoulder[1] - mid_hip[1])
                    posture = "Straight" if ratio >= baseline_ratio * 0.9 else "Hunched"

                    # Draw posture annotation
                    cv2.putText(frame, f"Posture: {posture}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Draw keypoints
                    for pt in kp:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

            # ------------- EYE STATE DETECTION -------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    h, w, _ = frame.shape

                    # Landmark indices for eyes
                    left_eye_idx = [33, 160, 158, 133, 153, 144]
                    right_eye_idx = [362, 385, 387, 263, 373, 380]

                    left_eye_pts = np.array([[int(face_landmarks.landmark[i].x * w),
                                              int(face_landmarks.landmark[i].y * h)] for i in left_eye_idx])
                    right_eye_pts = np.array([[int(face_landmarks.landmark[i].x * w),
                                               int(face_landmarks.landmark[i].y * h)] for i in right_eye_idx])

                    # --- Expanded Eye Cropping Function ---
                    def get_eye_crop(eye_pts, frame, margin_scale=1.8):
                        x_min, y_min = np.min(eye_pts, axis=0)
                        x_max, y_max = np.max(eye_pts, axis=0)

                        # Expand bounding box to include entire eye
                        w_box = x_max - x_min
                        h_box = y_max - y_min
                        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

                        new_w = w_box * margin_scale
                        new_h = h_box * margin_scale

                        x_min = int(max(0, cx - new_w / 2))
                        x_max = int(min(frame.shape[1], cx + new_w / 2))
                        y_min = int(max(0, cy - new_h / 2))
                        y_max = int(min(frame.shape[0], cy + new_h / 2))

                        return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

                    # Bigger crops for full eye coverage
                    left_crop, (lx1, ly1, lx2, ly2) = get_eye_crop(left_eye_pts, frame)
                    right_crop, (rx1, ry1, rx2, ry2) = get_eye_crop(right_eye_pts, frame)

                    # Draw yellow boxes
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

                    # --- Prediction Function for new model ---
                    def predict_eye_state(crop):
                        if crop.size == 0:
                            return None
                        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        inputs = processor(img, return_tensors="pt").to(device)
                        with torch.no_grad():
                            logits = eye_model(**inputs).logits
                            pred = logits.softmax(dim=1).argmax(dim=1).item()
                            label = eye_model.config.id2label[pred]
                        if label.lower() == "closeeye":
                            return "Closed"
                        elif label.lower() == "openeye":
                            return "Open"
                        

                    left_state = predict_eye_state(left_crop)
                    right_state = predict_eye_state(right_crop)

                    # Logical rule for both eyes
                    if left_state == "Closed" and right_state == "Closed":
                        eyes_status = "Closed"
                    else:
                        eyes_status = "Open"
                     

                    # Draw eye status
                    cv2.putText(frame, f"Eyes: {eyes_status}", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # ---------- SAVE EACH FRAME ----------
            frame_filename = os.path.join(frames_folder, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_filename, frame)

            # ---------- LOG RESULTS ----------
            results_json.append({
                "frame": frame_idx,
                "posture": posture,
                "eyes": eyes_status
            })

            out_vid.write(frame)

    cap.release()
    out_vid.release()

    # ⏱️ End timer and print total runtime
    end_time = time.time()
    print(f"\n✅ Total processing time: {round(end_time - start_time, 2)} seconds\n")

    return jsonify({
        "message": "Analysis complete",
        "annotated_video_path": annotated_video_path,
        "annotated_frames_folder": frames_folder,
        "results": results_json
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
