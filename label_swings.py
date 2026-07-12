import csv
import os

import cv2
import mediapipe as mp
import numpy as np

from feature_extraction import ARM_SIDE, extract_frame_features, get_point

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

VIDEO_PATH = "videos/input/singles_test_clip.mp4"

# Playback window is capped to this width so the display fits on screen
# regardless of source video resolution
MAX_DISPLAY_WIDTH = 1920
MAX_DISPLAY_HEIGHT = 1080

LABELS_PATH = "data/labels/labels.csv"
LABEL_FIELDS = ["video", "start_frame", "end_frame", "start_time", "end_time", "shot_type"]


def append_label(video_name, start_frame, end_frame, fps, shot_type):
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    write_header = not os.path.exists(LABELS_PATH) or os.path.getsize(LABELS_PATH) == 0
    with open(LABELS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "video": video_name,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_frame / fps,
            "end_time": end_frame / fps,
            "shot_type": shot_type,
        })


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

scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height)
display_width = int(width * scale)
display_height = int(height * scale)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("videos/output/output_skeleton.mp4", fourcc, fps, (width, height))

prev_wrist_px = None
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0
last_written_idx = -1
paused = False
video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
pending_start_frame = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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
            cv2.putText(image, f"Wrist velocity: {features['wrist_displacement']:.0f} px/frame", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Error occurred: {e}")
            prev_wrist_px = None

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 255, 80), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 80, 200), thickness=2, circle_radius=2)
                                  )

        status = f"Frame {frame_idx}/{total_frames - 1}{'  [PAUSED]' if paused else ''}"
        if pending_start_frame is not None:
            status += f"  [swing start @ {pending_start_frame}]"
        cv2.putText(image, status, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        display = cv2.resize(image, (display_width, display_height))
        cv2.imshow("Video", display)

        # Only write frames in forward order so the output video stays in sequence
        # even when the user scrubs backward to re-inspect a swing
        if frame_idx > last_written_idx:
            out.write(image)
            last_written_idx = frame_idx

        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d'):  # step forward one frame
            paused = True
            frame_idx = min(frame_idx + 1, total_frames - 1)
            prev_wrist_px = None
        elif key == ord('a'):  # step back one frame
            paused = True
            frame_idx = max(frame_idx - 1, 0)
            prev_wrist_px = None
        elif key == ord('s'):  # mark swing start at current frame
            paused = True
            pending_start_frame = frame_idx
            print(f"Swing start marked at frame {frame_idx}")
        elif key == ord('c'):  # cancel a pending swing start
            if pending_start_frame is not None:
                print(f"Cancelled swing start at frame {pending_start_frame}")
                pending_start_frame = None
        elif key == ord('e'):  # mark swing end and prompt for shot type
            if pending_start_frame is None:
                print("No swing start marked yet (press 's' first)")
            else:
                paused = True
                start_frame, end_frame = pending_start_frame, frame_idx
                if end_frame < start_frame:
                    start_frame, end_frame = end_frame, start_frame
                shot_type = input(
                    f"Shot type for frames {start_frame}-{end_frame} "
                    "(smash/clear/drop/drive/net/serve, blank to cancel): "
                ).strip()
                if shot_type:
                    append_label(video_name, start_frame, end_frame, fps, shot_type)
                    print(f"Saved label: {video_name} {start_frame}-{end_frame} {shot_type}")
                else:
                    print("Cancelled (no shot type entered)")
                pending_start_frame = None
        elif not paused:
            frame_idx += 1

cap.release()
cv2.destroyAllWindows()
