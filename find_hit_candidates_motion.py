"""Find candidate hit windows from wrist-motion peaks.

Fallback for videos with unusable audio - prefer find_hit_candidates_audio.py (audio
onsets), which localizes the contact instant far more precisely than motion peaks.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from build_features_csv import extract_video_features, features_path_for
from pose_features import FEATURE_NAMES

CANDIDATES_DIR = "data/labels/candidates"

# Centered rolling mean smooths out single-frame landmark jitter, which otherwise
# spikes wrist speed higher than real swings do
SMOOTH_WINDOW = 6

# Percentile-based threshold, kept low to favor recall: missing a true swing costs
# more (it just never gets reviewed) than a false candidate does (a quick reject)
PEAK_HEIGHT_PERCENTILE = 65
MIN_PEAK_DISTANCE = 15  # frames between peaks, so one swing doesn't yield multiple candidates
WINDOW_PADDING = 15     # frames of context before/after the peak in each candidate window


def find_candidates(video_path, video_name, player):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    features_path = features_path_for(video_path, player)
    df = pd.read_csv(features_path) if os.path.exists(features_path) else None
    if df is None or not set(FEATURE_NAMES).issubset(df.columns) or df["frame_idx"].max() < total_frames - 1:
        print(f"Extracting features for {video_name} ({player}) - this may take a while...")
        df = extract_video_features(video_path, player)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        df.to_csv(features_path, index=False)

    df = df[df["landmarks_detected"]].sort_values("frame_idx").reset_index(drop=True)
    smoothed = df["wrist_speed"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()

    height = np.percentile(smoothed, PEAK_HEIGHT_PERCENTILE)
    peak_positions, _ = find_peaks(smoothed, height=height, distance=MIN_PEAK_DISTANCE)
    peak_frames = df["frame_idx"].iloc[peak_positions].to_numpy()

    return pd.DataFrame({
        "video": video_name,
        "peak_frame": peak_frames,
        "start_frame": np.maximum(peak_frames - WINDOW_PADDING, 0),
        "end_frame": np.minimum(peak_frames + WINDOW_PADDING, total_frames - 1),
    })


def main():
    parser = argparse.ArgumentParser(description="Find candidate hit windows from wrist-motion peaks.")
    parser.add_argument("video_path")
    parser.add_argument("--player", default="near", choices=["near", "far"])
    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    candidates = find_candidates(args.video_path, video_name, args.player)

    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    out_path = os.path.join(CANDIDATES_DIR, f"{video_name}_motion.csv")
    candidates.to_csv(out_path, index=False)
    print(f"Found {len(candidates)} motion candidates, saved to {out_path}")


if __name__ == "__main__":
    main()
