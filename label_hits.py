"""Label badminton hit events.

Playback:  space = pause/play   a/d = step back/forward one frame   q = quit
Candidates (--candidates): n/p = next/previous candidate window
Labeling:  press the stroke-type key at the contact frame:
           1=clear 2=smash 3=drop 4=drive 5=net 6=lift 7=serve 0=other
           u = undo the last saved hit for this video/player

Every displayed frame is logged as "reviewed" (data/labels/reviewed/), so
frames you actually watched - and only those - can serve as not-swing
negatives when the training dataset is built.
"""
import argparse
import csv
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from pose_features import ARM_SIDE, extract_frame_features, get_point, landmarks_are_reliable
from pose_tracker import PlayerPoseTracker

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Playback window is capped to this size so the display fits on screen
MAX_DISPLAY_WIDTH = 1920
MAX_DISPLAY_HEIGHT = 1080

HITS_PATH = "data/labels/hits.csv"
HIT_FIELDS = ["video", "player", "hit_frame", "hit_time", "stroke_type"]
REVIEWED_DIR = "data/labels/reviewed"

STROKE_KEYS = {
    ord('1'): "clear", ord('2'): "smash", ord('3'): "drop", ord('4'): "drive",
    ord('5'): "net", ord('6'): "lift", ord('7'): "serve", ord('0'): "other",
}
LEGEND = "1:clear 2:smash 3:drop 4:drive 5:net 6:lift 7:serve 0:other u:undo"

CANDIDATE_LEAD_FRAMES = 15  # jump slightly before each candidate so you see the wind-up


def append_hit(video_name, player, hit_frame, fps, stroke_type):
    os.makedirs(os.path.dirname(HITS_PATH), exist_ok=True)
    write_header = not os.path.exists(HITS_PATH) or os.path.getsize(HITS_PATH) == 0
    with open(HITS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HIT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "video": video_name,
            "player": player,
            "hit_frame": hit_frame,
            "hit_time": hit_frame / fps,
            "stroke_type": stroke_type,
        })


def undo_last_hit(video_name, player):
    """Remove the most recent hit for this video/player. Returns the removed row or None."""
    if not os.path.exists(HITS_PATH):
        return None
    df = pd.read_csv(HITS_PATH)
    mask = (df["video"] == video_name) & (df["player"] == player)
    if not mask.any():
        return None
    last_idx = df[mask].index[-1]
    removed = df.loc[last_idx].to_dict()
    df.drop(index=last_idx).to_csv(HITS_PATH, index=False)
    return removed


def frames_to_intervals(frames):
    """Sorted set of frame indices -> merged [start, end] intervals."""
    intervals = []
    for f in sorted(frames):
        if intervals and f <= intervals[-1][1] + 1:
            intervals[-1][1] = max(intervals[-1][1], f)
        else:
            intervals.append([f, f])
    return intervals


def save_reviewed(video_name, player, viewed_frames):
    """Merge this session's viewed frames into the per-video reviewed-intervals file."""
    if not viewed_frames:
        return
    os.makedirs(REVIEWED_DIR, exist_ok=True)
    path = os.path.join(REVIEWED_DIR, f"{video_name}_{player}.csv")
    frames = set(viewed_frames)
    if os.path.exists(path):
        for _, row in pd.read_csv(path).iterrows():
            frames.update(range(int(row["start_frame"]), int(row["end_frame"]) + 1))
    intervals = frames_to_intervals(frames)
    pd.DataFrame(intervals, columns=["start_frame", "end_frame"]).to_csv(path, index=False)
    print(f"Reviewed coverage saved: {len(intervals)} interval(s) in {path}")


def put_angle_text(image, text, point, frame_shape):
    h, w = frame_shape[:2]
    coord = tuple(np.multiply(point, [w, h]).astype(int))
    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


parser = argparse.ArgumentParser(description="Step through a video and label hit events with stroke types.")
parser.add_argument("video_path")
parser.add_argument("--candidates", help="CSV of candidate hit windows (from find_hit_candidates_audio.py or "
                                          "find_hit_candidates_motion.py) "
                                          "to jump between with 'n'/'p' instead of scrubbing manually")
parser.add_argument("--player", default="near", choices=["near", "far"],
                    help="which player to track and label (near = bottom of frame)")
parser.add_argument("--mirror", action="store_true", help="flip frames horizontally (left-handed player)")
args = parser.parse_args()

video_name = os.path.splitext(os.path.basename(args.video_path))[0]

cap = cv2.VideoCapture(args.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height)
display_size = (int(width * scale), int(height * scale))

# model_complexity=1 keeps frame-stepping responsive; extraction uses 2 offline
tracker = PlayerPoseTracker(court_side=args.player, mirror=args.mirror, model_complexity=1)

prev_wrist_px = None
frame_idx = 0
paused = False
last_hit_msg = ""
viewed_frames = set()

candidates = None
candidate_idx = 0
if args.candidates:
    candidates = pd.read_csv(args.candidates)
    candidates = candidates[candidates["video"] == video_name].sort_values("start_frame").reset_index(drop=True)
    if len(candidates):
        frame_idx = max(int(candidates.loc[0, "start_frame"]) - CANDIDATE_LEAD_FRAMES, 0)
        paused = True
        print(f"Loaded {len(candidates)} candidates for {video_name}. 'n'/'p' to jump between them.")
    else:
        print(f"No candidates found for {video_name} in {args.candidates}")

try:
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        results, crop_box, frame = tracker.process(frame)
        image = frame.copy()
        viewed_frames.add(frame_idx)

        if results is None or not results.pose_landmarks or \
                not landmarks_are_reliable(results.pose_landmarks.landmark, mp_pose.PoseLandmark, ARM_SIDE):
            prev_wrist_px = None
        else:
            landmarks = results.pose_landmarks.landmark
            features, prev_wrist_px = extract_frame_features(
                landmarks, mp_pose.PoseLandmark, width, height, fps, prev_wrist_px, ARM_SIDE
            )

            for text, point in [
                (f"Elbow: {features['elbow_angle']:.0f}", get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_ELBOW"))),
                (f"Shoulder: {features['shoulder_angle']:.0f}", get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_SHOULDER"))),
                (f"Knee: {features['knee_angle']:.0f}", get_point(landmarks, getattr(mp_pose.PoseLandmark, f"{ARM_SIDE}_KNEE"))),
            ]:
                put_angle_text(image, text, point, image.shape)

            cv2.putText(image, f"Wrist speed: {features['wrist_speed']:.1f} torso/s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Contact height: {features['contact_height']:.2f} torso", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if results is not None and results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 255, 80), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 80, 200), thickness=2, circle_radius=2))
        if crop_box is not None:
            cv2.rectangle(image, crop_box[:2], crop_box[2:], (0, 200, 255), 2)

        status = f"Frame {frame_idx}/{total_frames - 1}  [{args.player}]{'  [PAUSED]' if paused else ''}"
        if candidates is not None and len(candidates):
            status += f"  [candidate {candidate_idx + 1}/{len(candidates)}]"
        if last_hit_msg:
            status += f"  [{last_hit_msg}]"
        cv2.putText(image, status, (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, LEGEND, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Label hits", cv2.resize(image, display_size))

        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d'):
            paused = True
            frame_idx = min(frame_idx + 1, total_frames - 1)
            prev_wrist_px = None
        elif key == ord('a'):
            paused = True
            frame_idx = max(frame_idx - 1, 0)
            prev_wrist_px = None
        elif key in STROKE_KEYS:
            stroke_type = STROKE_KEYS[key]
            append_hit(video_name, args.player, frame_idx, fps, stroke_type)
            last_hit_msg = f"saved {stroke_type} @ {frame_idx}"
            print(f"Saved hit: {video_name} ({args.player}) frame {frame_idx} {stroke_type}")
            paused = True
        elif key == ord('u'):
            removed = undo_last_hit(video_name, args.player)
            if removed is None:
                print("No hit to undo for this video/player")
            else:
                last_hit_msg = f"undid {removed['stroke_type']} @ {removed['hit_frame']}"
                print(f"Undid hit: frame {removed['hit_frame']} {removed['stroke_type']}")
        elif key == ord('n') and candidates is not None and len(candidates):
            candidate_idx = min(candidate_idx + 1, len(candidates) - 1)
            frame_idx = max(int(candidates.loc[candidate_idx, "start_frame"]) - CANDIDATE_LEAD_FRAMES, 0)
            paused = True
            prev_wrist_px = None
        elif key == ord('p') and candidates is not None and len(candidates):
            candidate_idx = max(candidate_idx - 1, 0)
            frame_idx = max(int(candidates.loc[candidate_idx, "start_frame"]) - CANDIDATE_LEAD_FRAMES, 0)
            paused = True
            prev_wrist_px = None
        elif not paused:
            frame_idx += 1
finally:
    save_reviewed(video_name, args.player, viewed_frames)
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
