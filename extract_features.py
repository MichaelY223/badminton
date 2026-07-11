import argparse
import os

import cv2
import mediapipe as mp
import pandas as pd

from feature_extraction import ARM_SIDE, FEATURE_NAMES, extract_frame_features

mp_pose = mp.solutions.pose

FEATURES_OUTPUT_DIR = "data/features"


def extract_video_features(video_path, arm_side=ARM_SIDE):
    """Run pose estimation over a video and return one feature row per frame."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    prev_wrist_px = None
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            row = {"frame_idx": frame_idx, "timestamp": frame_idx / fps, "landmarks_detected": False}
            row.update({name: None for name in FEATURE_NAMES})

            if results.pose_landmarks:
                features, prev_wrist_px = extract_frame_features(
                    results.pose_landmarks.landmark, mp_pose.PoseLandmark,
                    width, height, prev_wrist_px, arm_side,
                )
                row["landmarks_detected"] = True
                row.update(features)
            else:
                # Reset velocity tracking so the next detected frame isn't read as one
                # huge jump spanning the whole gap of missed frames
                prev_wrist_px = None

            rows.append(row)
            frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract per-frame pose features from a video to CSV.")
    parser.add_argument("video_path")
    parser.add_argument("--arm-side", default=ARM_SIDE, choices=["LEFT", "RIGHT"])
    args = parser.parse_args()

    os.makedirs(FEATURES_OUTPUT_DIR, exist_ok=True)
    df = extract_video_features(args.video_path, args.arm_side)

    stem = os.path.splitext(os.path.basename(args.video_path))[0]
    out_path = os.path.join(FEATURES_OUTPUT_DIR, f"{stem}.csv")
    df.to_csv(out_path, index=False)

    detected = int(df["landmarks_detected"].sum())
    print(f"Wrote {len(df)} rows ({detected} with detected landmarks) to {out_path}")


if __name__ == "__main__":
    main()
