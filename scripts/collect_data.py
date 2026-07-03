import os
import csv
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "hand_landmarker.task")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
CSV_PATH = os.path.join(DATASET_DIR, "gestures.csv")

os.makedirs(DATASET_DIR, exist_ok=True)

# -----------------------------
# Gesture rules
# -----------------------------
# Gojo = one hand
# Naruto = two hands
# Idle = any non-target pose with one or two hands
REQUIRED_HANDS = {
    "gojo": 1,
    "naruto": 2,
    "idle": None
}

# -----------------------------
# MediaPipe setup
# -----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.40,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.40
)

landmarker = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# CSV setup
# Two hands × 21 landmarks × x,y,z = 126 values
# -----------------------------
header = ["label"]

for hand_slot in range(2):
    for point_number in range(21):
        header.extend([
            f"hand{hand_slot}_x{point_number}",
            f"hand{hand_slot}_y{point_number}",
            f"hand{hand_slot}_z{point_number}"
        ])

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(header)

# -----------------------------
# Hand-skeleton connections
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
# Convert one/two hands into
# a fixed 126-number sample
# -----------------------------
def make_sample(hand_landmarks):
    # Keeps the two hands in a consistent left-to-right order.
    hands = sorted(hand_landmarks, key=lambda hand: hand[0].x)

    sample = []

    for hand in hands[:2]:
        for landmark in hand:
            sample.extend([landmark.x, landmark.y, landmark.z])

    # For Gojo's one-hand sign, the other hand slot becomes zeros.
    while len(sample) < 126:
        sample.append(0.0)

    return sample


# -----------------------------
# Webcam setup
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

counts = {"gojo": 0, "naruto": 0, "idle": 0}
last_message = "Ready. Show a gesture and press a key."
start_time = time.perf_counter()

print("\nControls:")
print("G = Save one-hand Gojo sign")
print("N = Save two-hand Naruto sign")
print("I = Save idle / random non-target pose")
print("Q = Quit\n")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read camera frame.")
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

    # Draw hand landmarks and connections
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

    # Display instructions
    cv2.putText(
        frame,
        f"Hands detected: {hands_detected}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "G: Gojo (1 hand) | N: Naruto (2 hands) | I: Idle | Q: Quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Gojo: {counts['gojo']}  Naruto: {counts['naruto']}  Idle: {counts['idle']}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        last_message,
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Dataset Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    key_to_label = {
        ord("g"): "gojo",
        ord("n"): "naruto",
        ord("i"): "idle"
    }

    if key in key_to_label:
        label = key_to_label[key]
        needed = REQUIRED_HANDS[label]

        # Idle may be collected with either one or two visible hands.
        if label == "idle" and hands_detected not in (1, 2):
            last_message = "Idle needs at least one detected hand."
            print(last_message)
            continue

        # Gojo and Naruto have exact hand-count rules.
        if needed is not None and hands_detected != needed:
            last_message = (
                f"{label.title()} needs exactly {needed} hand(s). "
                f"Detected: {hands_detected}"
            )
            print(last_message)
            continue

        sample = make_sample(result.hand_landmarks)

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([label] + sample)

        counts[label] += 1
        last_message = f"Saved {label} sample #{counts[label]}"
        print(last_message)

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("\nDataset saved at:")
print(CSV_PATH)