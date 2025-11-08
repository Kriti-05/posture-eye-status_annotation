from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import os

app = Flask(__name__)

# ---------------- CONFIG ----------------
model = YOLO("yolo11n-pose.pt")  # pose model
eye_thresh = 0.22  # EAR threshold

mp_face = mp.solutions.face_mesh

# Ensure Downloads/annotated_videos folder exists
output_folder = os.path.join("Downloads", "annotated_videos")
os.makedirs(output_folder, exist_ok=True)
annotated_video_path = os.path.join(output_folder, "annotated_video.mp4")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

@app.route("/")
def home():
    return "YOLO + MediaPipe Video Analysis Server is running!"

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    input_path = os.path.join(output_folder, "input_video.mp4")
    video_file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

    # Baseline posture for first 3 seconds
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

    # Rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    results_json = []

    # Create FaceMesh instance inside request
    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            results = model.predict(frame, verbose=False)
            posture, eyes_status = "Unknown", "Unknown"

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

                    # Draw skeleton points
                    for pt in kp:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

            # Eye detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    left_eye_idx = [33, 160, 158, 133, 153, 144]
                    right_eye_idx = [362, 385, 387, 263, 373, 380]

                    left_eye = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in left_eye_idx])
                    right_eye = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in right_eye_idx])

                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    eyes_status = "Closed" if avg_ear < eye_thresh else "Open"

                    # Draw eyes status
                    cv2.putText(frame, f"Eyes: {eyes_status}", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            results_json.append({
                "frame": frame_idx,
                "posture": posture,
                "eyes": eyes_status
            })

            # Write annotated frame
            out_vid.write(frame)

    cap.release()
    out_vid.release()

    return jsonify({
        "results": results_json,
        "annotated_video_path": annotated_video_path
    })

@app.route("/download", methods=["GET"])
def download_video():
    if os.path.exists(annotated_video_path):
        return send_file(annotated_video_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
