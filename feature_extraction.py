import numpy as np

# Hitting arm side - flip to "LEFT" for a left-handed player
ARM_SIDE = "RIGHT"

FEATURE_NAMES = [
    "elbow_angle", "shoulder_angle", "wrist_angle", "knee_angle",
    "trunk_rotation", "torso_lean", "contact_height",
    "wrist_x", "wrist_y", "wrist_displacement",
]


def calculate_angle(a, b, c):
    """Angle at point b, formed by rays b->a and b->c, in degrees."""
    # Each of a, b, c is a 2D point (x, y)
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

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


def extract_frame_features(landmarks, pose_landmark_enum, frame_w, frame_h, prev_wrist_px=None, arm_side=ARM_SIDE):
    """Compute one frame's swing-analysis features from MediaPipe pose landmarks.

    Returns (features, wrist_px). Pass wrist_px back in as prev_wrist_px on the next
    frame to get a continuous wrist_displacement signal; pass None to reset it (e.g. after
    a frame with no detected landmarks, so a multi-frame gap isn't read as one huge jump).
    """
    # Use the pose_landmark_enum to access the correct landmark indices
    landmark_enum = pose_landmark_enum

    # f"{armside}_SHOULDER" is a string like "RIGHT_SHOULDER", which is the name of the enum member we want to access
    # getattr(landmark_enum, ...) gets the enum member by name, and then use it to index into landmarks to get actual landmark data
    # landmarks is a list of landmark objects, each with x, y, z and visibility attributes
    # getpoint(landmarks, ...) returns a list of [x, y] coordinates for the specified landmark
    shoulder = get_point(landmarks, getattr(landmark_enum, f"{arm_side}_SHOULDER"))
    elbow = get_point(landmarks, getattr(landmark_enum, f"{arm_side}_ELBOW"))
    wrist = get_point(landmarks, getattr(landmark_enum, f"{arm_side}_WRIST"))
    index = get_point(landmarks, getattr(landmark_enum, f"{arm_side}_INDEX"))
    hip = get_point(landmarks, getattr(landmark_enum, f"{arm_side}_HIP"))
    other_hip = get_point(landmarks, getattr(landmark_enum, f"{'LEFT' if arm_side == 'RIGHT' else 'RIGHT'}_HIP")) # TEMP
    knee = get_point(landmarks, getattr(landmark_enum, f"{arm_side}_KNEE"))
    other_knee = get_point(landmarks, getattr(landmark_enum, f"{'LEFT' if arm_side == 'RIGHT' else 'RIGHT'}_KNEE")) # TEMP
    ankle = get_point(landmarks, getattr(landmark_enum, f"{arm_side}_ANKLE"))
    other_ankle = get_point(landmarks, getattr(landmark_enum, f"{'LEFT' if arm_side == 'RIGHT' else 'RIGHT'}_ANKLE")) # TEMP
    opp_shoulder = get_point(landmarks, getattr(landmark_enum, "LEFT_SHOULDER" if arm_side == "RIGHT" else "RIGHT_SHOULDER"))
    opp_hip = get_point(landmarks, getattr(landmark_enum, "LEFT_HIP" if arm_side == "RIGHT" else "RIGHT_HIP"))

    # Format: [x, y] in normalized coordinates (0-1)

    wrist_px = np.array([wrist[0] * frame_w, wrist[1] * frame_h])
    wrist_displacement = float(np.linalg.norm(wrist_px - prev_wrist_px)) if prev_wrist_px is not None else 0.0

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
        "trunk_rotation": calculate_angle(opp_shoulder, shoulder, hip) - calculate_angle(opp_hip, hip, shoulder),
        # Torso lean from vertical: forward/backward body tilt
        "torso_lean": calculate_tilt_from_vertical(shoulder, hip),
        # Wrist height relative to shoulder (negative y = above shoulder)
        "contact_height": shoulder[1] - wrist[1],
        "wrist_x": wrist[0],
        "wrist_y": wrist[1],
        "wrist_displacement": wrist_displacement,
    }

    return features, wrist_px
