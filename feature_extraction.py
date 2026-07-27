import numpy as np

# Hitting arm side. Left-handed players are handled by mirroring frames in the
# pose pipeline (PlayerPoseTracker(mirror=True)), so feature code always sees RIGHT.
ARM_SIDE = "RIGHT"

FEATURE_NAMES = [
    "elbow_angle", "shoulder_angle", "wrist_angle", "knee_angle", "other_knee_angle",
    "trunk_rotation", "torso_lean", "contact_height",
    "wrist_x", "wrist_y", "wrist_speed",
]

# MediaPipe can report pose_landmarks for a frame even when a point has snapped to a
# clearly wrong position (e.g. off the edge of the frame). Landmark coordinates
# outside roughly this margin around [0, 1] are treated as not actually detected.
#
# Visibility score was tried first instead of this bounds check, but it backfired:
# the hitting arm's elbow/wrist/index have the lowest visibility of any landmark
# specifically because fast arm motion causes motion blur - and that's most likely to
# happen during an actual swing, so gating on visibility disproportionately threw out
# the frames we care about most.
COORD_BOUNDS_MARGIN = 0.1


def calculate_angle(a, b, c):
    """Angle at point b, formed by rays b->a and b->c, in degrees."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_tilt_from_vertical(top, bottom):
    """Angle of the top->bottom vector from the vertical axis, in degrees."""
    dx = top[0] - bottom[0]
    dy = top[1] - bottom[1]
    return np.degrees(np.arctan2(abs(dx), abs(dy)))


def get_point(landmarks, landmark):
    lm = landmarks[landmark.value]
    return [lm.x, lm.y]


def used_landmark_names(arm_side=ARM_SIDE):
    """Every landmark name extract_frame_features actually reads, for the given arm side."""
    other_side = "LEFT" if arm_side == "RIGHT" else "RIGHT"
    return [
        f"{arm_side}_SHOULDER", f"{arm_side}_ELBOW", f"{arm_side}_WRIST", f"{arm_side}_INDEX",
        f"{arm_side}_HIP", f"{arm_side}_KNEE", f"{arm_side}_ANKLE",
        f"{other_side}_HIP", f"{other_side}_KNEE", f"{other_side}_ANKLE", f"{other_side}_SHOULDER",
    ]


def landmarks_are_reliable(landmarks, pose_landmark_enum, arm_side=ARM_SIDE, margin=COORD_BOUNDS_MARGIN):
    """
    Whether every landmark extract_frame_features uses has a plausible position.
    Checking this catches frames where a person was detected but a point snapped somewhere nonsesical.
    """
    lo, hi = -margin, 1 + margin
    return all(
        lo <= landmarks[getattr(pose_landmark_enum, name).value].x <= hi and
        lo <= landmarks[getattr(pose_landmark_enum, name).value].y <= hi
        for name in used_landmark_names(arm_side)
    )


def extract_frame_features(landmarks, pose_landmark_enum, frame_w, frame_h, fps,
                           prev_wrist_px=None, arm_side=ARM_SIDE):
    """Compute one frame's swing-analysis features from MediaPipe pose landmarks.

    All geometry is computed in pixel space: normalized coordinates scale x by width
    and y by height, which distorts angles on non-square frames.

    Motion is expressed in torso-lengths per second: torso length (mid-shoulder to
    mid-hip) is stable under trunk rotation, unlike shoulder width which foreshortens
    mid-swing, and fps scaling keeps the same physical swing comparable between videos
    recorded at different frame rates.

    Returns (features, wrist_px). Pass wrist_px back in as prev_wrist_px on the next
    frame for a continuous wrist_speed signal; pass None to reset it (e.g. after a
    frame with no reliable landmarks, so a gap isn't read as one huge jump).
    """
    landmark_enum = pose_landmark_enum
    other_side = "LEFT" if arm_side == "RIGHT" else "RIGHT"

    def px(name):
        lm = landmarks[getattr(landmark_enum, name).value]
        return np.array([lm.x * frame_w, lm.y * frame_h])

    shoulder = px(f"{arm_side}_SHOULDER")
    elbow = px(f"{arm_side}_ELBOW")
    wrist = px(f"{arm_side}_WRIST")
    index = px(f"{arm_side}_INDEX")
    hip = px(f"{arm_side}_HIP")
    knee = px(f"{arm_side}_KNEE")
    ankle = px(f"{arm_side}_ANKLE")
    other_hip = px(f"{other_side}_HIP")
    other_knee = px(f"{other_side}_KNEE")
    other_ankle = px(f"{other_side}_ANKLE")
    opp_shoulder = px(f"{other_side}_SHOULDER")

    mid_shoulder = (shoulder + opp_shoulder) / 2
    mid_hip = (hip + other_hip) / 2
    torso_px = float(np.linalg.norm(mid_shoulder - mid_hip))
    if torso_px < 1e-6:
        torso_px = 1.0

    wrist_displacement_px = float(np.linalg.norm(wrist - prev_wrist_px)) if prev_wrist_px is not None else 0.0
    wrist_speed = (wrist_displacement_px * fps) / torso_px

    features = {
        # Elbow angle: how extended the hitting arm is
        "elbow_angle": calculate_angle(shoulder, elbow, wrist),
        # Shoulder angle: arm elevation relative to torso (racket-up position)
        "shoulder_angle": calculate_angle(elbow, shoulder, hip),
        # Wrist snap angle: elbow-wrist-index, tracks the pronation/snap
        "wrist_angle": calculate_angle(elbow, wrist, index),
        # Hip/knee angle: front-leg loading and extension for jump shots
        "knee_angle": calculate_angle(hip, knee, ankle),
        # Other knee angle: for comparison with the active leg
        "other_knee_angle": calculate_angle(other_hip, other_knee, other_ankle),
        # Trunk rotation: shoulder line vs hip line, twist between upper/lower body
        "trunk_rotation": calculate_angle(opp_shoulder, shoulder, hip) - calculate_angle(other_hip, hip, shoulder),
        # Torso lean from vertical: forward/backward body tilt
        "torso_lean": calculate_tilt_from_vertical(mid_shoulder, mid_hip),
        # Wrist height above shoulder, in torso lengths (positive = above shoulder)
        "contact_height": float(shoulder[1] - wrist[1]) / torso_px,
        # Raw normalized wrist position, kept for debugging/display only -
        # excluded from models because it encodes camera framing
        "wrist_x": wrist[0] / frame_w,
        "wrist_y": wrist[1] / frame_h,
        # Wrist speed in torso-lengths per second
        "wrist_speed": wrist_speed,
    }

    return features, wrist
