import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from feature_extraction import ARM_SIDE, extract_frame_features, get_point

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

VIDEO_PATH = "videos/input/smash.mp4"

# Playback window is capped to this width so the display fits on screen
# regardless of source video resolution
DISPLAY_WIDTH = 960


def put_angle_text(image, text, point, frame_shape):
    h, w = frame_shape[:2]
    coord = tuple(np.multiply(point, [w, h]).astype(int))
    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(VIDEO_PATH)

# Get source video properties so the output matches
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
display_height = int(height * DISPLAY_WIDTH / width)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # see note below
out = cv2.VideoWriter("videos/output/output_skeleton.mp4", fourcc, fps, (width, height))

prev_wrist_px = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks and compute smash-relevant features
        try:
            landmarks = results.pose_landmarks.landmark
            features, prev_wrist_px = extract_frame_features(
                landmarks, mp_pose.PoseLandmark, width, height, prev_wrist_px, ARM_SIDE
            )

            shoulder_px = get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_SHOULDER"))
            elbow_px = get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_ELBOW"))
            wrist_px_norm = get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_WRIST"))
            knee_px = get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_KNEE"))

            for text, point in [
                (f"Elbow: {features['elbow_angle']:.0f}", elbow_px),
                (f"Shoulder: {features['shoulder_angle']:.0f}", shoulder_px),
                (f"Wrist: {features['wrist_angle']:.0f}", wrist_px_norm),
                (f"Knee: {features['knee_angle']:.0f}", knee_px),
            ]:
                put_angle_text(image, text, point, image.shape)

            cv2.putText(image, f"Trunk rotation: {features['trunk_rotation']:.0f} deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Torso lean: {features['torso_lean']:.0f} deg", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Contact height (rel. shoulder): {features['contact_height']:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Wrist velocity: {features['wrist_velocity']:.0f} px/frame", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            prev_wrist_px = None

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 255, 80), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 80, 200), thickness=2, circle_radius=2)
                                  )

        display = cv2.resize(image, (DISPLAY_WIDTH, display_height))
        cv2.imshow("Video", display)

        out.write(image)  # write the annotated BGR frame (source resolution)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
