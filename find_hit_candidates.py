"""Find candidate hit frames from the audio track.

A racket impact is a sharp transient with strong high-frequency content, so
onset detection runs on the >2 kHz band only - voices, footsteps and crowd
noise live mostly below that. Candidates are written in the same CSV format as
find_swing_candidates.py, so label_swings.py --candidates works unchanged.

Validation against 28 hand-labeled swings in long_singles (far-court camera,
noisy audio): 50% of swings had an onset within +/-6 frames, at ~75 candidates
per minute. On footage like that, prefer the motion-based generator
(find_swing_candidates.py); audio is worth retrying on recordings made closer
to the court.
"""
import argparse
import os

import cv2
import librosa
import numpy as np
import pandas as pd

CANDIDATES_DIR = "data/labels/candidates"
HOP_LENGTH = 512
WINDOW_PADDING = 15   # frames of context on each side of a hit in the candidate CSV
MIN_GAP_SEC = 0.35    # two hits can't be closer than this
HF_CUTOFF_HZ = 2000   # onset detection above this band only; racket impacts are
                      # broadband clicks, voices/crowd are mostly lower-frequency


def load_audio(video_path, sr=22050):
    try:
        return librosa.load(video_path, sr=sr, mono=True)
    except Exception:
        # librosa needs an ffmpeg-backed reader for video containers; fall back to
        # extracting the audio track with moviepy
        from moviepy import VideoFileClip
        import tempfile
        with VideoFileClip(video_path) as clip:
            tmp = os.path.join(tempfile.gettempdir(), "hit_candidates_audio.wav")
            clip.audio.write_audiofile(tmp, fps=sr, logger=None)
        return librosa.load(tmp, sr=sr, mono=True)


def detect_hits(video_path, strength_percentile):
    y, sr = load_audio(video_path)
    spectrum = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr)
    hf_band = spectrum[freqs >= HF_CUTOFF_HZ]
    onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(hf_band),
                                             sr=sr, hop_length=HOP_LENGTH)
    wait = max(int(MIN_GAP_SEC * sr / HOP_LENGTH), 1)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
        units="frames", backtrack=False, wait=wait,
    )
    if not len(onset_frames):
        return np.array([]), np.array([])

    strengths = onset_env[onset_frames]
    threshold = np.percentile(strengths, strength_percentile)
    keep = strengths >= threshold
    times = librosa.frames_to_time(onset_frames[keep], sr=sr, hop_length=HOP_LENGTH)
    return times, strengths[keep]


def main():
    parser = argparse.ArgumentParser(description="Detect candidate hit frames from a video's audio track.")
    parser.add_argument("video_path")
    parser.add_argument("--strength-percentile", type=float, default=50,
                        help="keep onsets at or above this percentile of onset strength "
                             "(lower = more candidates, higher recall)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    times, strengths = detect_hits(args.video_path, args.strength_percentile)
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]

    hit_frames = np.clip(np.round(times * fps).astype(int), 0, total_frames - 1)
    candidates = pd.DataFrame({
        "video": video_name,
        "peak_frame": hit_frames,
        "start_frame": np.maximum(hit_frames - WINDOW_PADDING, 0),
        "end_frame": np.minimum(hit_frames + WINDOW_PADDING, total_frames - 1),
        "onset_strength": np.round(strengths, 3),
    })

    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    out_path = os.path.join(CANDIDATES_DIR, f"{video_name}_audio.csv")
    candidates.to_csv(out_path, index=False)
    duration_min = total_frames / fps / 60
    print(f"Found {len(candidates)} audio hit candidates in {duration_min:.1f} min of video "
          f"({len(candidates) / max(duration_min, 1e-9):.1f}/min), saved to {out_path}")


if __name__ == "__main__":
    main()
