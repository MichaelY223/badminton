"""Build a window-level training dataset from hit-event labels.

One row per window: positives are centered on labeled hit frames, negatives are
sampled from frames the labeler actually reviewed (data/labels/reviewed/) at a
safe distance from every hit. Window statistics - including angular velocities -
summarize the temporal shape a single frame can't capture.
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd

from extract_features import extract_video_features, features_path_for
from feature_extraction import FEATURE_NAMES

HITS_PATH = "data/labels/hits.csv"
REVIEWED_DIR = "data/labels/reviewed"
VIDEO_DIR = "videos/input"
OUTPUT_PATH = "data/training/windows.csv"

WINDOW_SEC = 0.5        # window half-width around the hit: covers backswing -> follow-through
MIN_COVERAGE = 0.8      # fraction of window frames that must have detected landmarks
NEG_MIN_GAP_SEC = 1.0   # negative window centers stay at least this far from any hit
NEG_STRIDE_SEC = 0.5    # spacing between candidate negative centers

# Features summarized over the window. Raw wrist_x/wrist_y are excluded: they
# encode camera framing, not the swing.
BASE_COLS = ["elbow_angle", "shoulder_angle", "wrist_angle", "knee_angle", "other_knee_angle",
             "trunk_rotation", "torso_lean", "contact_height", "wrist_speed"]
# Angular velocity carries the "rate of change" signal a static pose lacks
DERIV_COLS = ["elbow_angle", "shoulder_angle", "wrist_angle", "knee_angle", "other_knee_angle",
              "trunk_rotation"]


def find_video_file(video_name):
    matches = glob.glob(os.path.join(VIDEO_DIR, f"{video_name}.*"))
    if not matches:
        raise FileNotFoundError(f"No video file for '{video_name}' in {VIDEO_DIR}")
    return matches[0]


def video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def load_features(video_name, player):
    video_path = find_video_file(video_name)
    path = features_path_for(video_path, player)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if set(FEATURE_NAMES).issubset(df.columns):
            return df
    print(f"Extracting features for {video_name} ({player}) - this may take a while...")
    df = extract_video_features(video_path, player)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_reviewed_intervals(video_name, player):
    path = os.path.join(REVIEWED_DIR, f"{video_name}_{player}.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return [(int(r["start_frame"]), int(r["end_frame"])) for _, r in df.iterrows()]


def window_features(features, center, half_width, fps):
    """Summary statistics for the window [center-half_width, center+half_width].

    Returns None if too few frames in the window have detected landmarks.
    """
    lo, hi = center - half_width, center + half_width
    win = features[(features["frame_idx"] >= lo) & (features["frame_idx"] <= hi)]
    win = win[win["landmarks_detected"]]
    if len(win) < MIN_COVERAGE * (2 * half_width + 1):
        return None

    row = {}
    for col in BASE_COLS:
        vals = win[col].astype(float)
        row[f"{col}_mean"] = vals.mean()
        row[f"{col}_std"] = vals.std()
        row[f"{col}_min"] = vals.min()
        row[f"{col}_max"] = vals.max()

    # Derivatives only between genuinely consecutive frames, so detection gaps
    # don't masquerade as huge velocities
    consecutive = win["frame_idx"].diff() == 1
    for col in DERIV_COLS:
        deriv = (win[col].diff() * fps)[consecutive].astype(float)
        row[f"{col}_vel_absmax"] = deriv.abs().max() if len(deriv) else 0.0
        row[f"{col}_vel_std"] = deriv.std() if len(deriv) > 1 else 0.0
    return row


def main():
    if not os.path.exists(HITS_PATH):
        print(f"No hit labels yet ({HITS_PATH} missing) - label some hits first with label_swings.py.")
        return
    hits = pd.read_csv(HITS_PATH)
    all_rows = []

    for (video_name, player), group in hits.groupby(["video", "player"]):
        video_path = find_video_file(video_name)
        fps = video_fps(video_path)
        half_width = round(WINDOW_SEC * fps)
        features = load_features(video_name, player)
        hit_frames = group["hit_frame"].astype(int).to_numpy()

        n_pos = n_neg = 0
        for _, hit in group.iterrows():
            row = window_features(features, int(hit["hit_frame"]), half_width, fps)
            if row is None:
                continue
            row.update({"video": video_name, "player": player, "center_frame": int(hit["hit_frame"]),
                        "is_swing": 1, "stroke_type": hit["stroke_type"]})
            all_rows.append(row)
            n_pos += 1

        min_gap = round(NEG_MIN_GAP_SEC * fps)
        stride = max(round(NEG_STRIDE_SEC * fps), 1)
        for start, end in load_reviewed_intervals(video_name, player):
            for center in range(start + half_width, end - half_width + 1, stride):
                if len(hit_frames) and np.min(np.abs(hit_frames - center)) < min_gap:
                    continue
                row = window_features(features, center, half_width, fps)
                if row is None:
                    continue
                row.update({"video": video_name, "player": player, "center_frame": center,
                            "is_swing": 0, "stroke_type": ""})
                all_rows.append(row)
                n_neg += 1

        print(f"{video_name} ({player}): {n_pos} swing windows, {n_neg} negative windows")

    if not all_rows:
        print("No windows produced - label some hits first (label_swings.py).")
        return

    dataset = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(dataset)} windows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
