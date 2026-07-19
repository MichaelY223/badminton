import argparse
import os

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from build_dataset import extract_features_up_to

FEATURES_DIR = "data/features"
VIDEO_DIR = "videos/input"
CANDIDATES_DIR = "data/labels/candidates"
LABELS_PATH = "data/labels/labels.csv"

# Centered rolling mean smooths out single-frame landmark jitter, which otherwise
# spikes wrist_displacement_norm higher than real swings do (checked against the
# existing labeled data before picking this over raw/rolling-max)
SMOOTH_WINDOW = 6

# Percentile-based threshold, kept low to favor recall: missing a true swing costs
# more (it just never gets reviewed) than a false candidate does (a quick reject)
PEAK_HEIGHT_PERCENTILE = 65
MIN_PEAK_DISTANCE = 15  # frames between peaks, so one swing doesn't yield multiple candidates
WINDOW_PADDING = 15  # frames of context before/after the peak in each candidate window


def find_candidates(video_name, video_file, existing_labels=None):
    video_path = os.path.join(VIDEO_DIR, video_file)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    features_path = os.path.join(FEATURES_DIR, f"{video_name}.csv")
    df = pd.read_csv(features_path) if os.path.exists(features_path) else None
    if df is None or df["frame_idx"].max() < total_frames - 1:
        print(f"Extracting features for {video_name} (full video, this may take a while)...")
        df = extract_features_up_to(video_path, max_frame=total_frames - 1)
        os.makedirs(FEATURES_DIR, exist_ok=True)
        df.to_csv(features_path, index=False)

    df = df[df["landmarks_detected"]].sort_values("frame_idx").reset_index(drop=True)
    smoothed = df["wrist_displacement_norm"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()

    height = np.percentile(smoothed, PEAK_HEIGHT_PERCENTILE)
    peak_positions, _ = find_peaks(smoothed, height=height, distance=MIN_PEAK_DISTANCE)
    peak_frames = df["frame_idx"].iloc[peak_positions].to_numpy()

    candidates = pd.DataFrame({
        "video": video_name,
        "peak_frame": peak_frames,
        "start_frame": np.maximum(peak_frames - WINDOW_PADDING, 0),
        "end_frame": peak_frames + WINDOW_PADDING,
    })

    if existing_labels is not None and len(existing_labels):
        def overlaps_existing(row):
            return ((existing_labels["start_frame"] <= row["end_frame"]) &
                     (existing_labels["end_frame"] >= row["start_frame"])).any()
        candidates = candidates[~candidates.apply(overlaps_existing, axis=1)]

    return candidates.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Find candidate swing windows from wrist-motion peaks.")
    parser.add_argument("video_path")
    args = parser.parse_args()

    video_file = os.path.basename(args.video_path)
    video_name = os.path.splitext(video_file)[0]

    existing_labels = None
    if os.path.exists(LABELS_PATH):
        labels = pd.read_csv(LABELS_PATH)
        existing_labels = labels[labels["video"] == video_name]

    candidates = find_candidates(video_name, video_file, existing_labels)

    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    out_path = os.path.join(CANDIDATES_DIR, f"{video_name}.csv")
    candidates.to_csv(out_path, index=False)
    print(f"Found {len(candidates)} candidate swing windows, saved to {out_path}")


if __name__ == "__main__":
    main()
