import os
import time
import joblib
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

HAND_MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "hand_landmarker.task"
)

CLASSIFIER_PATH = os.path.join(
    PROJECT_DIR, "models", "gesture_classifier.joblib"
)

# Load your trained gesture model
saved_data = joblib.load(CLASSIFIER_PATH)
classifier = saved_data["model"]
feature_names = saved_data["feature_names"]

print("Gesture classifier loaded!")
print("Classes:", saved_data["classes"])

# Load MediaPipe hand detector
base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40
)

landmarker = vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def make_sample(hand_landmarks):
    # Keep hands in the same left-to-right order used while collecting data.
    hands = sorted(hand_landmarks, key=lambda hand: hand[0].x)

    values = []

    for hand in hands[:2]:
        for landmark in hand:
            values.extend([landmark.x, landmark.y, landmark.z])

    # One-hand gestures get zero values for the missing second hand.
    while len(values) < 126:
        values.append(0.0)

    return pd.DataFrame([values], columns=feature_names)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

print("Webcam started. Press Q to quit.")

start_time = time.perf_counter()
stable_label = "No gesture"
previous_label = None
same_label_frames = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int((time.perf_counter() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    height, width, _ = frame.shape
    hands_detected = len(result.hand_landmarks)

    # Draw detected hands
    for hand in result.hand_landmarks:
        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        for start, end in CONNECTIONS:
            x1 = int(hand[start].x * width)
            y1 = int(hand[start].y * height)
            x2 = int(hand[end].x * width)
            y2 = int(hand[end].y * height)

            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    prediction_text = "Show a hand"
    confidence = 0.0

    if hands_detected >= 1:
        sample = make_sample(result.hand_landmarks)

        probabilities = classifier.predict_proba(sample)[0]
        best_index = probabilities.argmax()

        predicted_label = classifier.classes_[best_index]
        confidence = probabilities[best_index] * 100

        # Avoid rapidly changing labels from one frame to another.
        if confidence >= 65:
            if predicted_label == previous_label:
                same_label_frames += 1
            else:
                previous_label = predicted_label
                same_label_frames = 1

            if same_label_frames >= 8:
                stable_label = predicted_label

            prediction_text = stable_label
        else:
            prediction_text = "Uncertain"

    else:
        previous_label = None
        same_label_frames = 0
        stable_label = "No gesture"

    cv2.putText(
        frame,
        f"Hands: {hands_detected}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Gesture: {prediction_text.upper()}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q: Quit",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("Anime Gesture Controller - Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()