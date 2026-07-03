import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Find model automatically
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "hand_landmarker.task")

# -----------------------------
# Load Hand Landmarker
# -----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,

    # Lower values make detection less strict.
    # Do not reduce below 0.30, otherwise false detections increase.
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40,
)

landmarker = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# Hand skeleton connections
# -----------------------------
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

print("Webcam started. Press Q to quit.")

start_time = time.perf_counter()

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Mirror view so it feels natural.
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # VIDEO mode needs a timestamp that always increases.
    timestamp_ms = int((time.perf_counter() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    height, width, _ = frame.shape
    hands_detected = len(result.hand_landmarks)

    # Draw detected hands
    for hand_index, hand in enumerate(result.hand_landmarks):

        # Find left/right label
        hand_name = "Hand"
        if result.handedness and hand_index < len(result.handedness):
            hand_name = result.handedness[hand_index][0].category_name

        # Draw landmarks
        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw hand skeleton
        for start, end in CONNECTIONS:
            x1 = int(hand[start].x * width)
            y1 = int(hand[start].y * height)
            x2 = int(hand[end].x * width)
            y2 = int(hand[end].y * height)

            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Label hand near wrist
        wrist_x = int(hand[0].x * width)
        wrist_y = int(hand[0].y * height)

        cv2.putText(
            frame,
            hand_name,
            (wrist_x - 45, wrist_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # Screen instructions
    cv2.putText(
        frame,
        f"Hands detected: {hands_detected}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Keep both hands visible and separated",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("Anime Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()