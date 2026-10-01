import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

mp_pose = mp.solutions.pose

PERSON_CLASS = 0
DETECTION_CONF = 0.4
# Fraction of box size added on each side of the crop so the racket arm stays
# in frame at full extension
CROP_MARGIN = 0.35
# EMA weight on the previous box so the crop doesn't jitter frame to frame
BOX_SMOOTHING = 0.6

PLAYER_MODES = ["auto", "near", "far"]
# Boxes shorter than this fraction of the tallest box are ignored when picking a
# player: spectators, line judges, ball kids, partial bodies at the frame edge
MIN_REL_HEIGHT = 0.3
# Per-mode weights on the target-selection cues (all cues are normalized to [0, 1]):
#   size   - box height relative to the tallest person in frame
#   center - horizontal closeness to the frame center (players are framed centrally;
#            officials and crowd sit at the sides)
#   low    - how far down the frame the feet are (proxy for closeness to camera)
#   high   - inverse of low (far court)
#   conf   - detector confidence
SCORE_WEIGHTS = {
    "auto": {"size": 0.45, "center": 0.30, "low": 0.15, "conf": 0.10},
    "near": {"size": 0.25, "center": 0.20, "low": 0.45, "conf": 0.10},
    "far":  {"size": 0.15, "center": 0.35, "high": 0.40, "conf": 0.10},
}
# Once locked, a detection continues the track if its center moved less than this
# many box-heights since the last frame and its height changed by less than this ratio
MAX_JUMP = 0.6
MAX_HEIGHT_RATIO = 1.5
# Frames without a matching detection before the lock is dropped and the target is
# re-picked by score (e.g. after a broadcast camera cut)
REACQUIRE_AFTER_FRAMES = 8


class PlayerPoseTracker:
    """Person-detect -> crop -> pose, locked onto one player.

    Fixes two failure modes of running MediaPipe on the full frame:
    - small/distant subjects: pose runs on a zoomed crop instead of the whole court
    - two-player footage: MediaPipe is single-person and silently picks whoever is
      most salient; the crop pins it to the player chosen by court side, so labels
      and skeletons always describe the same person

    court_side: "auto" (most prominent person: big, central, close to camera),
    "near" (bottom of frame) or "far" (top of frame).
    mirror: flip frames horizontally for left-handed players, so downstream
    feature code can always treat the hitting arm as RIGHT.

    Selection has two stages: when no player is locked, every detection is scored
    on size/centrality/court position and the best one is locked; after that the
    lock follows whichever detection continues the previous box, so a momentarily
    bigger umpire or the other player can't steal the crop. The lock is dropped
    after REACQUIRE_AFTER_FRAMES frames with no continuing detection.
    """

    def __init__(self, court_side="auto", mirror=False, model_complexity=2):
        if court_side not in SCORE_WEIGHTS:
            raise ValueError(f"court_side must be one of {PLAYER_MODES}, got '{court_side}'")
        self.court_side = court_side
        self.mirror = mirror
        self.detector = YOLO("yolov8n.pt")
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                 model_complexity=model_complexity)
        self.box = None  # smoothed [x1, y1, x2, y2] in pixels
        self.locked_box = None  # last raw detection of the tracked player
        self.frames_unmatched = 0
        # Last frame's detections as (box, score) for debug drawing; score is None
        # for frames where the target came from the lock rather than from scoring
        self.candidates = []

    def reset(self):
        """Forget the locked player. Call after seeking to an unrelated part of the video."""
        self.box = None
        self.locked_box = None
        self.frames_unmatched = 0

    def _score(self, boxes, confs, w, h):
        heights = boxes[:, 3] - boxes[:, 1]
        cues = {
            "size": heights / heights.max(),
            "center": 1 - np.abs((boxes[:, 0] + boxes[:, 2]) / 2 - w / 2) / (w / 2),
            "low": boxes[:, 3] / h,
            "high": 1 - boxes[:, 3] / h,
            "conf": confs,
        }
        weights = SCORE_WEIGHTS[self.court_side]
        scores = sum(weight * cues[name] for name, weight in weights.items())
        return np.where(cues["size"] >= MIN_REL_HEIGHT, scores, -np.inf)

    def _continue_lock(self, boxes):
        """Index of the detection that continues the locked track, or None."""
        lx0, ly0, lx1, ly1 = self.locked_box
        lh = ly1 - ly0
        centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        jump = np.linalg.norm(centers - [(lx0 + lx1) / 2, (ly0 + ly1) / 2], axis=1) / lh
        height_ratio = (boxes[:, 3] - boxes[:, 1]) / lh
        ok = (jump < MAX_JUMP) & (height_ratio < MAX_HEIGHT_RATIO) & (height_ratio > 1 / MAX_HEIGHT_RATIO)
        if not ok.any():
            return None
        return int(np.argmin(np.where(ok, jump, np.inf)))

    def _pick_target(self, boxes, confs, w, h):
        if self.locked_box is not None and len(boxes):
            idx = self._continue_lock(boxes)
            if idx is not None:
                self.frames_unmatched = 0
                self.candidates = [(b, None) for b in boxes]
                return boxes[idx]

        if self.locked_box is not None:
            self.frames_unmatched += 1
            if self.frames_unmatched <= REACQUIRE_AFTER_FRAMES:
                # Brief occlusion or missed detection: hold the crop where it was
                self.candidates = [(b, None) for b in boxes]
                return None
            self.reset()

        if not len(boxes):
            self.candidates = []
            return None
        scores = self._score(boxes, confs, w, h)
        self.candidates = list(zip(boxes, scores))
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            return None
        return boxes[best]

    def process(self, frame_bgr):
        """Run detection + pose on one frame.

        Returns (results, crop_box, frame_bgr). Landmarks in results are remapped to
        full-frame normalized coordinates, so drawing and feature extraction work
        exactly as they would on an uncropped frame. frame_bgr is returned because
        mirror=True flips it; callers must display/measure on the returned frame.
        crop_box is (x0, y0, x1, y1) in pixels, or None if no person was found yet.
        """
        if self.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        h, w = frame_bgr.shape[:2]

        det = self.detector.predict(frame_bgr, classes=[PERSON_CLASS], conf=DETECTION_CONF, verbose=False)[0]
        if det.boxes is not None and len(det.boxes):
            boxes = det.boxes.xyxy.cpu().numpy().astype(float)
            confs = det.boxes.conf.cpu().numpy().astype(float)
        else:
            boxes, confs = np.empty((0, 4)), np.empty(0)
        target = self._pick_target(boxes, confs, w, h)
        if target is not None:
            self.locked_box = target
            self.box = target if self.box is None else BOX_SMOOTHING * self.box + (1 - BOX_SMOOTHING) * target

        if self.box is None:
            # No detection yet in this video: fall back to the full frame
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            bx0, by0, bx1, by1 = self.box
            # Increase the crop box by a fraction of its size so the racket arm stays in frame at full extension
            mx, my = (bx1 - bx0) * CROP_MARGIN, (by1 - by0) * CROP_MARGIN

            # Top left is reduced by the margin to move it left/up but not beyond the frame
            # Bottom right is increased by the margin to move it right/down but not beyond the frame
            x0 = int(max(bx0 - mx, 0))
            y0 = int(max(by0 - my, 0))
            x1 = int(min(bx1 + mx, w))
            y1 = int(min(by1 + my, h))

        # Crops out just the part of the frame that is in the bounding box of the player to run pose detection on
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None, None, frame_bgr

        results = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            cw, ch = x1 - x0, y1 - y0
            for lm in results.pose_landmarks.landmark:
                lm.x = (x0 + lm.x * cw) / w
                lm.y = (y0 + lm.y * ch) / h

        crop_box = None if self.box is None else (x0, y0, x1, y1)
        return results, crop_box, frame_bgr

    def close(self):
        self.pose.close()
