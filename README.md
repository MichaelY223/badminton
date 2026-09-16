# Badminton Swing Analysis

A computer-vision pipeline for analyzing badminton play from video: track a chosen
player's pose, extract swing-relevant features, label hit events, and train a
classifier to distinguish swings from non-swings.

The pipeline: **video → player-locked pose tracking → per-frame features → hit-event
labels → window dataset → classifier**.

## How it works

1. **Pose tracking** ([pose_tracker.py](pose_tracker.py)) — a YOLOv8 person detector
   picks out one target player by court side (near/far), crops tightly around them,
   and runs MediaPipe Pose on the crop. This keeps the tracked skeleton locked to one
   player in multi-person footage and improves pose quality on small/distant subjects
   compared to running MediaPipe on the full frame.
2. **Feature extraction** ([pose_features.py](pose_features.py),
   [build_features_csv.py](build_features_csv.py)) — per frame, computes joint angles
   (elbow, shoulder, wrist, knees), trunk rotation, torso lean, contact height, and
   wrist speed. Motion is normalized to torso-lengths per second so features are
   comparable across videos shot at different frame rates and camera distances.
3. **Hit candidate detection** ([find_hit_candidates_audio.py](find_hit_candidates_audio.py),
   [find_hit_candidates_motion.py](find_hit_candidates_motion.py)) — generates candidate
   hit windows from racket-impact audio transients or wrist-speed peaks, so labeling
   long videos means reviewing a short list of candidates instead of scrubbing the
   whole timeline.
4. **Labeling** ([label_hits.py](label_hits.py)) — an interactive OpenCV player for
   stepping through a video (or its candidate windows) and marking the contact frame
   and stroke type (clear, smash, drop, drive, net, lift, serve, other) for each swing.
5. **Dataset building** ([build_dataset.py](build_dataset.py)) — turns hit labels into
   a window-level training set: positive windows centered on labeled hits, negative
   windows sampled from reviewed-but-unlabeled footage, summarized with window
   statistics (mean/std/min/max) and angular velocities.
6. **Baseline classifier** ([baseline_swing_classifier.py](baseline_swing_classifier.py)) —
   a gradient-boosted-tree swing/no-swing classifier, evaluated leave-one-video-out.
7. **Visualization** ([visualize_pose.py](visualize_pose.py)) — plays back a video with
   the tracked skeleton and live feature readout overlaid, and writes it to
   `videos/output/output_skeleton.mp4`.

See [DESIGN_REVIEW.md](DESIGN_REVIEW.md) for the methodology behind these design
choices and a roadmap of planned improvements.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Place source videos under `videos/input/` (e.g. `videos/input/long_singles.mp4`).

## Usage

All commands are run with `uv run`.

**Preview pose tracking on a video:**

```bash
uv run visualize_pose.py videos/input/long_singles.mp4 --player near
```

**Extract per-frame features for a player:**

```bash
uv run build_features_csv.py videos/input/long_singles.mp4 --player near
```

Writes to `data/features/<video>_<player>.csv`.

**Generate hit candidates** (prefer audio when the recording is close to the court;
fall back to motion otherwise):

```bash
uv run find_hit_candidates_audio.py videos/input/long_singles.mp4
uv run find_hit_candidates_motion.py videos/input/long_singles.mp4 --player near
```

**Label hits**, optionally stepping between generated candidates:

```bash
uv run label_hits.py videos/input/long_singles.mp4 --player near \
    --candidates data/labels/candidates/long_singles_audio.csv
```

Controls: `space` pause/play · `a`/`d` step back/forward one frame ·
`n`/`p` next/previous candidate · `1`-`7`/`0` label stroke type at the current
frame (clear/smash/drop/drive/net/lift/serve/other) · `u` undo last hit · `q` quit.

Add `--mirror` for a left-handed player, so features are always computed as if the
hitting arm were on the right.

**Build the training dataset from labeled hits:**

```bash
uv run build_dataset.py
```

Writes `data/training/windows.csv`.

**Train and evaluate the baseline classifier:**

```bash
uv run baseline_swing_classifier.py
```

## Project layout

```
videos/input/                 source videos
videos/output/                 rendered/annotated videos
data/features/                 per-video, per-player extracted features
data/labels/hits.csv           labeled hit events
data/labels/candidates/        generated candidate hit windows
data/labels/reviewed/          frame ranges a labeler has actually viewed
data/training/windows.csv      window-level training dataset
```

`videos/` and `data/` are gitignored (all generated locally) — only code is tracked.
