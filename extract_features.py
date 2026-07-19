import argparse
import os

import cv2
import mediapipe as mp
import pandas as pd

from feature_extraction import ARM_SIDE, FEATURE_NAMES, extract_frame_features, landmarks_are_reliable
from pose_pipeline import PlayerPoseTracker

mp_pose = mp.solutions.pose

FEATURES_OUTPUT_DIR = "data/features"


def features_path_for(video_path, player):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(FEATURES_OUTPUT_DIR, f"{stem}_{player}.csv")


def extract_video_features(video_path, player="near", mirror=False, arm_side=ARM_SIDE):
    """Run the detect-crop-pose pipeline over a video, one feature row per frame."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = PlayerPoseTracker(court_side=player, mirror=mirror, model_complexity=2)

    rows = []
    prev_wrist_px = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results, _, _ = tracker.process(frame)

        row = {"frame_idx": frame_idx, "timestamp": frame_idx / fps, "landmarks_detected": False}
        row.update({name: None for name in FEATURE_NAMES})

        if results is not None and results.pose_landmarks and \
                landmarks_are_reliable(results.pose_landmarks.landmark, mp_pose.PoseLandmark, arm_side):
            features, prev_wrist_px = extract_frame_features(
                results.pose_landmarks.landmark, mp_pose.PoseLandmark,
                width, height, fps, prev_wrist_px, arm_side,
            )
            row["landmarks_detected"] = True
            row.update(features)
        else:
            # Reset speed tracking so the next detected frame isn't read as one
            # huge jump spanning the whole gap of missed frames
            prev_wrist_px = None

        rows.append(row)
        frame_idx += 1

    cap.release()
    tracker.close()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract per-frame pose features for one player to CSV.")
    parser.add_argument("video_path")
    parser.add_argument("--player", default="near", choices=["near", "far"],
                        help="which player to track (near = bottom of frame)")
    parser.add_argument("--mirror", action="store_true",
                        help="flip frames horizontally (left-handed player)")
    args = parser.parse_args()

    os.makedirs(FEATURES_OUTPUT_DIR, exist_ok=True)
    df = extract_video_features(args.video_path, args.player, args.mirror)

    out_path = features_path_for(args.video_path, args.player)
    df.to_csv(out_path, index=False)

    detected = int(df["landmarks_detected"].sum())
    print(f"Wrote {len(df)} rows ({detected} with detected landmarks) to {out_path}")


if __name__ == "__main__":
    main()
