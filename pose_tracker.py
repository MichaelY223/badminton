"""Play a video with the tracked player's skeleton and features overlaid,
and write the annotated video to videos/output/output_skeleton.mp4."""
import argparse
import os

import cv2
import mediapipe as mp
import numpy as np

from feature_extraction import ARM_SIDE, extract_frame_features, get_point, landmarks_are_reliable
from pose_pipeline import PlayerPoseTracker

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Playback window is capped to this size so the display fits on screen
MAX_DISPLAY_WIDTH = 1920
MAX_DISPLAY_HEIGHT = 1080


def put_angle_text(image, text, point, frame_shape):
    h, w = frame_shape[:2]
    coord = tuple(np.multiply(point, [w, h]).astype(int))
    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

# CL arguments for video path, player side, and mirror mode. Defaults to the near player and long_singles.mp4
parser = argparse.ArgumentParser(description="View pose tracking with feature overlays.")
parser.add_argument("video_path", nargs="?", default="videos/input/long_singles.mp4")
parser.add_argument("--player", default="near", choices=["near", "far"])
parser.add_argument("--mirror", action="store_true")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height)
display_size = (int(width * scale), int(height * scale))

os.makedirs("videos/output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("videos/output/output_skeleton.mp4", fourcc, fps, (width, height))

tracker = PlayerPoseTracker(court_side=args.player, mirror=args.mirror, model_complexity=1)
prev_wrist_px = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results, crop_box, frame = tracker.process(frame)
    image = frame.copy()

    if results is None or not results.pose_landmarks or not landmarks_are_reliable(results.pose_landmarks.landmark, mp_pose.PoseLandmark, ARM_SIDE):
        prev_wrist_px = None
    else:
        landmarks = results.pose_landmarks.landmark
        features, prev_wrist_px = extract_frame_features(
            landmarks, mp_pose.PoseLandmark, width, height, fps, prev_wrist_px, ARM_SIDE
        )

        for text, point in [
            (f"Elbow: {features['elbow_angle']:.0f}", get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_ELBOW"))),
            (f"Shoulder: {features['shoulder_angle']:.0f}", get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_SHOULDER"))),
            (f"Wrist: {features['wrist_angle']:.0f}", get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_WRIST"))),
            (f"Knee: {features['knee_angle']:.0f}", get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_KNEE"))),
        ]:
            put_angle_text(image, text, point, image.shape)

        cv2.putText(image, f"Trunk rotation: {features['trunk_rotation']:.0f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Torso lean: {features['torso_lean']:.0f} deg", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Contact height: {features['contact_height']:.2f} torso", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Wrist speed: {features['wrist_speed']:.1f} torso/s", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if results is not None and results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 255, 80), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 80, 200), thickness=2, circle_radius=2))
    if crop_box is not None:
        cv2.rectangle(image, crop_box[:2], crop_box[2:], (0, 200, 255), 2)

    cv2.imshow("Video", cv2.resize(image, display_size))
    out.write(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
tracker.close()
cv2.destroyAllWindows()
