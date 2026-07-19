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


class PlayerPoseTracker:
    """Person-detect -> crop -> pose, locked onto one player.

    Fixes two failure modes of running MediaPipe on the full frame:
    - small/distant subjects: pose runs on a zoomed crop instead of the whole court
    - two-player footage: MediaPipe is single-person and silently picks whoever is
      most salient; the crop pins it to the player chosen by court side, so labels
      and skeletons always describe the same person

    court_side: "near" (bottom of frame) or "far" (top of frame).
    mirror: flip frames horizontally for left-handed players, so downstream
    feature code can always treat the hitting arm as RIGHT.
    """

    def __init__(self, court_side="near", mirror=False, model_complexity=2):
        self.court_side = court_side
        self.mirror = mirror
        self.detector = YOLO("yolov8n.pt")
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                 model_complexity=model_complexity)
        self.box = None  # smoothed [x1, y1, x2, y2] in pixels

    def _pick_target(self, boxes):
        if not len(boxes):
            return None
        # The two largest detections are the players; court side disambiguates them
        # by bottom edge (the near player stands lower in the image)
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:2]
        pick = max if self.court_side == "near" else min
        return pick(boxes, key=lambda b: b[3])

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

        det = self.detector.predict(frame_bgr, classes=[PERSON_CLASS], conf=DETECTION_CONF,
                                    verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else np.empty((0, 4))
        target = self._pick_target(list(boxes))
        if target is not None:
            target = np.asarray(target, dtype=float)
            self.box = target if self.box is None else BOX_SMOOTHING * self.box + (1 - BOX_SMOOTHING) * target

        if self.box is None:
            # No detection yet in this video: fall back to the full frame
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            bx0, by0, bx1, by1 = self.box
            mx, my = (bx1 - bx0) * CROP_MARGIN, (by1 - by0) * CROP_MARGIN
            x0 = int(max(bx0 - mx, 0))
            y0 = int(max(by0 - my, 0))
            x1 = int(min(bx1 + mx, w))
            y1 = int(min(by1 + my, h))

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
