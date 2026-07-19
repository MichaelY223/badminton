import os

import cv2
import mediapipe as mp
import pandas as pd

from feature_extraction import ARM_SIDE, FEATURE_NAMES, extract_frame_features

mp_pose = mp.solutions.pose

LABELS_PATH = "data/labels/labels.csv"
FEATURES_DIR = "data/features"
CANDIDATES_DIR = "data/labels/candidates"
VIDEO_DIR = "videos/input"
OUTPUT_PATH = "data/training/dataset.csv"

INCLUDED_VIDEOS = {
    "singles_test_clip": "singles_test_clip.mp4",
    "long_singles": "long_singles.mp4",
}

# Frames beyond a video's last labeled swing haven't been reviewed yet, so they
# can't be trusted as "not swing" negatives
REVIEW_BUFFER_FRAMES = 30

# Rolling window (in frames) used to summarize recent motion, since a swing is a
# multi-frame motion pattern that a single frame's joint angles can't capture
ROLL_WINDOW = 10


def extract_features_up_to(video_path, max_frame, arm_side=ARM_SIDE):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []
    prev_wrist_px = None
    frame_idx = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and frame_idx <= max_frame:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            row = {"frame_idx": frame_idx, "timestamp": frame_idx / fps, "landmarks_detected": False}
            row.update({name: None for name in FEATURE_NAMES})

            if results.pose_landmarks:
                features, prev_wrist_px = extract_frame_features(
                    results.pose_landmarks.landmark, mp_pose.PoseLandmark, width, height, prev_wrist_px, arm_side,
                )
                row["landmarks_detected"] = True
                row.update(features)
            else:
                prev_wrist_px = None

            rows.append(row)
            frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)


def load_or_extract_features(video_name, video_file, max_frame):
    features_path = os.path.join(FEATURES_DIR, f"{video_name}.csv")
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        has_all_columns = set(FEATURE_NAMES).issubset(df.columns)
        if has_all_columns and df["frame_idx"].max() >= max_frame:
            return df

    video_path = os.path.join(VIDEO_DIR, video_file)
    print(f"Extracting {video_name} features up to frame {max_frame}...")
    df = extract_features_up_to(video_path, max_frame)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_csv(features_path, index=False)
    return df


def add_rolling_motion_features(features):
    """Summarize recent wrist motion so the model sees a window, not just one frame.

    Computed over the full sequential frame range (before filtering down to reviewed
    frames) so the rolling stats reflect true temporal neighbors, not gaps introduced
    by candidate-window filtering.
    """
    roll = features["wrist_displacement_norm"].rolling(ROLL_WINDOW, min_periods=1)
    features["wrist_displacement_norm_roll_max"] = roll.max()
    features["wrist_displacement_norm_roll_std"] = roll.std().fillna(0.0)
    return features


def known_frame_mask(frame_idx, video_name, video_labels):
    """Which frames were actually reviewed and can be trusted as "not swing" negatives.

    If label_swings.py was run with --candidates, the reviewer only saw the padded
    windows around each candidate up through the last labeled swing - the gaps
    between candidates were never displayed, so they can't be treated as negatives.
    Otherwise the video was scrubbed continuously, so everything through the last
    labeled swing (plus a small buffer) was reviewed.
    """
    reviewed_boundary = int(video_labels["end_frame"].max())
    candidates_path = os.path.join(CANDIDATES_DIR, f"{video_name}.csv")
    if os.path.exists(candidates_path):
        candidates = pd.read_csv(candidates_path)
        reviewed = candidates[candidates["start_frame"] <= reviewed_boundary]
        print(f"  {video_name}: treating {len(reviewed)}/{len(candidates)} candidate windows "
              f"(start_frame <= {reviewed_boundary}) as reviewed")
        mask = pd.Series(False, index=frame_idx.index)
        for _, c in reviewed.iterrows():
            mask |= frame_idx.between(c["start_frame"], c["end_frame"])
        return mask
    return frame_idx <= reviewed_boundary + REVIEW_BUFFER_FRAMES


def main():
    labels = pd.read_csv(LABELS_PATH)
    labels = labels[labels["video"].isin(INCLUDED_VIDEOS)]

    all_rows = []
    for video_name, video_file in INCLUDED_VIDEOS.items():
        video_labels = labels[labels["video"] == video_name]
        extract_cap = int(video_labels["end_frame"].max()) + REVIEW_BUFFER_FRAMES

        features = load_or_extract_features(video_name, video_file, extract_cap)
        features = features[features["landmarks_detected"]].sort_values("frame_idx").copy()
        features = add_rolling_motion_features(features)

        known = known_frame_mask(features["frame_idx"], video_name, video_labels)
        features = features[known].copy()

        is_swing = pd.Series(False, index=features.index)
        for _, row in video_labels.iterrows():
            is_swing |= features["frame_idx"].between(row["start_frame"], row["end_frame"])
        features["is_swing"] = is_swing.astype(int)
        features["video"] = video_name

        all_rows.append(features)

    dataset = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(dataset)} rows to {OUTPUT_PATH}")
    print(dataset.groupby("video")["is_swing"].agg(["sum", "count"]))


if __name__ == "__main__":
    main()
