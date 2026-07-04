import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

VIDEO_PATH = "videos/input/smash.mp4"

# Hitting arm side - flip to "LEFT" for a left-handed player
ARM_SIDE = "RIGHT"


def calculate_angle(a, b, c):
    """Angle at point b, formed by rays b->a and b->c, in degrees."""
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
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    return angle


def get_point(landmarks, landmark):
    lm = landmarks[landmark.value]
    return [lm.x, lm.y]


def put_angle_text(image, text, point, frame_shape):
    h, w = frame_shape[:2]
    coord = tuple(np.multiply(point, [w, h]).astype(int))
    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(VIDEO_PATH)

# Get source video properties so the output matches
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # see note below
out = cv2.VideoWriter("videos/output/output_skeleton.mp4", fourcc, fps, (width, height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks and compute smash-relevant angles
        try:
            landmarks = results.pose_landmarks.landmark
            side = mp_pose.PoseLandmark

            shoulder = get_point(landmarks, getattr(side, f"{ARM_SIDE}_SHOULDER"))
            elbow = get_point(landmarks, getattr(side, f"{ARM_SIDE}_ELBOW"))
            wrist = get_point(landmarks, getattr(side, f"{ARM_SIDE}_WRIST"))
            index = get_point(landmarks, getattr(side, f"{ARM_SIDE}_INDEX"))
            hip = get_point(landmarks, getattr(side, f"{ARM_SIDE}_HIP"))
            knee = get_point(landmarks, getattr(side, f"{ARM_SIDE}_KNEE"))
            ankle = get_point(landmarks, getattr(side, f"{ARM_SIDE}_ANKLE"))
            opp_shoulder = get_point(landmarks, getattr(side, "LEFT_SHOULDER" if ARM_SIDE == "RIGHT" else "RIGHT_SHOULDER"))
            opp_hip = get_point(landmarks, getattr(side, "LEFT_HIP" if ARM_SIDE == "RIGHT" else "RIGHT_HIP"))
            nose = get_point(landmarks, side.NOSE)

            # Elbow angle: how extended the hitting arm is on contact
            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            # Shoulder angle: arm elevation relative to torso (racket-up position)
            shoulder_angle = calculate_angle(elbow, shoulder, hip)

            # Wrist snap angle: elbow-wrist-index, tracks the pronation/snap that drives smash speed
            wrist_angle = calculate_angle(elbow, wrist, index)

            # Hip/knee angle: front-leg loading and extension for jump smashes
            knee_angle = calculate_angle(hip, knee, ankle)

            # Trunk rotation: shoulder line vs hip line, twist between upper/lower body
            trunk_rotation = calculate_angle(opp_shoulder, shoulder, hip) - calculate_angle(opp_hip, hip, shoulder)

            # Torso lean from vertical: forward/backward body tilt at contact
            torso_lean = calculate_tilt_from_vertical(shoulder, hip)

            # Contact height: wrist position relative to shoulder (negative y = above shoulder, ideal for a smash)
            contact_height = shoulder[1] - wrist[1]

            for text, point in [
                (f"Elbow: {elbow_angle:.0f}", elbow),
                (f"Shoulder: {shoulder_angle:.0f}", shoulder),
                (f"Wrist: {wrist_angle:.0f}", wrist),
                (f"Knee: {knee_angle:.0f}", knee),
            ]:
                put_angle_text(image, text, point, image.shape)

            cv2.putText(image, f"Trunk rotation: {trunk_rotation:.0f} deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Torso lean: {torso_lean:.0f} deg", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Contact height (rel. shoulder): {contact_height:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 255, 80), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 80, 200), thickness=2, circle_radius=2)
                                  )

        cv2.imshow("Video", image)

        out.write(image)  # write the annotated BGR frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()