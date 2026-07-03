import os
import time
import math
import webbrowser

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

HAND_MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "hand_landmarker.task"
)

CLASSIFIER_PATH = os.path.join(
    PROJECT_DIR, "models", "gesture_classifier.joblib"
)


# -----------------------------
# Load trained gesture classifier
# -----------------------------
saved_data = joblib.load(CLASSIFIER_PATH)
classifier = saved_data["model"]
feature_names = saved_data["feature_names"]

print("✅ Gesture classifier loaded!")
print("Classes:", saved_data["classes"])


# -----------------------------
# Load MediaPipe hand detector
# -----------------------------
base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

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
# Prepare landmark data for AI
# -----------------------------
def make_sample(hand_landmarks):
    hands = sorted(hand_landmarks, key=lambda hand: hand[0].x)

    values = []

    for hand in hands[:2]:
        for landmark in hand:
            values.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

    while len(values) < 126:
        values.append(0.0)

    return pd.DataFrame([values], columns=feature_names)


# -----------------------------
# Drawing helpers
# -----------------------------
def draw_centered_text(frame, text, y, scale, color, thickness=2):
    height, width, _ = frame.shape

    text_size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        thickness
    )

    x = (width - text_size[0]) // 2

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness
    )


def draw_hand_skeleton(frame, hand_landmarks):
    height, width, _ = frame.shape

    for hand in hand_landmarks:
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


# -----------------------------
# Gojo blue aura
# -----------------------------
def draw_gojo_aura(frame, hand_landmarks, elapsed, duration):
    height, width, _ = frame.shape

    progress = min(elapsed / duration, 1.0)
    pulse = 0.5 + 0.5 * math.sin(elapsed * 10)

    glow = np.zeros_like(frame)

    for hand in hand_landmarks:
        hand_x = int(sum(point.x for point in hand) / len(hand) * width)
        hand_y = int(sum(point.y for point in hand) / len(hand) * height)

        cv2.circle(
            glow,
            (hand_x, hand_y),
            int(90 + pulse * 35),
            (255, 120, 10),
            -1
        )

    center = (width // 2, int(height * 0.48))

    cv2.circle(
        glow,
        center,
        int(min(width, height) * (0.25 + progress * 0.15)),
        (255, 90, 10),
        -1
    )

    glow = cv2.GaussianBlur(glow, (0, 0), 30)

    frame[:] = cv2.addWeighted(frame, 1.0, glow, 0.70, 0)

    for ring_number in range(3):
        ring_progress = (progress * 1.7 + ring_number * 0.30) % 1.0
        radius = int(80 + ring_progress * min(width, height) * 0.65)

        cv2.circle(
            frame,
            center,
            radius,
            (255, 220, 120),
            2
        )

    draw_centered_text(
        frame,
        "DOMAIN EXPANSION",
        50,
        1.0,
        (255, 245, 180)
    )

    draw_centered_text(
        frame,
        "INFINITE VOID",
        95,
        1.15,
        (255, 220, 100),
        3
    )


# -----------------------------
# Naruto orange chakra aura
# -----------------------------
def draw_naruto_aura(frame, hand_landmarks, elapsed, duration):
    height, width, _ = frame.shape

    progress = min(elapsed / duration, 1.0)
    pulse = 0.5 + 0.5 * math.sin(elapsed * 12)

    orange_overlay = np.full_like(frame, (0, 90, 255))
    frame[:] = cv2.addWeighted(frame, 0.80, orange_overlay, 0.20, 0)

    glow = np.zeros_like(frame)

    center_x = width // 2
    center_y = int(height * 0.48)

    # Orange chakra around every visible hand
    for hand in hand_landmarks:
        hand_x = int(sum(point.x for point in hand) / len(hand) * width)
        hand_y = int(sum(point.y for point in hand) / len(hand) * height)

        cv2.circle(
            glow,
            (hand_x, hand_y),
            int(110 + pulse * 45),
            (0, 120, 255),
            -1
        )

    # Main body aura
    cv2.ellipse(
        glow,
        (center_x, center_y),
        (
            int(width * (0.20 + 0.05 * pulse)),
            int(height * (0.34 + 0.06 * pulse))
        ),
        0,
        0,
        360,
        (0, 90, 255),
        -1
    )

    glow = cv2.GaussianBlur(glow, (0, 0), 35)
    frame[:] = cv2.addWeighted(frame, 1.0, glow, 0.82, 0)

    # Rotating energy rings
    for ring_number in range(3):
        angle = int(elapsed * 100 + ring_number * 60)
        radius_x = int(width * (0.22 + ring_number * 0.06))
        radius_y = int(height * (0.18 + ring_number * 0.05))

        cv2.ellipse(
            frame,
            (center_x, center_y),
            (radius_x, radius_y),
            angle,
            0,
            360,
            (0, 220, 255),
            2
        )

    # Flying chakra particles
    for particle_number in range(80):
        angle = particle_number * 2.399 + elapsed * 2.4
        distance = 50 + ((particle_number * 31) % 320)

        x = int(center_x + math.cos(angle) * distance)
        y = int(center_y + math.sin(angle) * distance)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(
                frame,
                (x, y),
                1 + particle_number % 3,
                (0, 235, 255),
                -1
            )

    draw_centered_text(
        frame,
        "CHAKRA ACTIVATED",
        50,
        1.0,
        (0, 245, 255)
    )

    draw_centered_text(
        frame,
        "NARUTO CURSOR MODE",
        95,
        1.0,
        (0, 220, 255),
        3
    )


# -----------------------------
# Orange glowing cursor + real mouse movement
# Uses the right-most hand's index fingertip
# -----------------------------
def control_naruto_cursor(frame, hand_landmarks, old_mouse_x, old_mouse_y):
    if not hand_landmarks:
        return old_mouse_x, old_mouse_y

    height, width, _ = frame.shape

    # Use the hand farthest right in the camera frame.
    cursor_hand = max(hand_landmarks, key=lambda hand: hand[8].x)
    index_tip = cursor_hand[8]

    hand_x = int(index_tip.x * width)
    hand_y = int(index_tip.y * height)

    screen_width, screen_height = pyautogui.size()

    # Leave a small edge margin for easier control.
    target_mouse_x = int(
        np.interp(index_tip.x, [0.08, 0.92], [0, screen_width - 1])
    )

    target_mouse_y = int(
        np.interp(index_tip.y, [0.08, 0.92], [0, screen_height - 1])
    )

    target_mouse_x = max(0, min(screen_width - 1, target_mouse_x))
    target_mouse_y = max(0, min(screen_height - 1, target_mouse_y))

    # Smooth the real cursor to reduce jitter.
    mouse_x = int(old_mouse_x + (target_mouse_x - old_mouse_x) * 0.24)
    mouse_y = int(old_mouse_y + (target_mouse_y - old_mouse_y) * 0.24)

    try:
        pyautogui.moveTo(mouse_x, mouse_y, duration=0)
    except pyautogui.FailSafeException:
        print("PyAutoGUI safety stop activated.")

    # Orange glow visible around your fingertip in webcam window.
    glow = np.zeros_like(frame)

    cv2.circle(glow, (hand_x, hand_y), 65, (0, 130, 255), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 24)

    frame[:] = cv2.addWeighted(frame, 1.0, glow, 0.85, 0)

    cv2.circle(frame, (hand_x, hand_y), 30, (0, 230, 255), 2)
    cv2.circle(frame, (hand_x, hand_y), 12, (0, 150, 255), -1)

    cv2.line(
        frame,
        (hand_x - 45, hand_y),
        (hand_x + 45, hand_y),
        (0, 230, 255),
        1
    )

    cv2.line(
        frame,
        (hand_x, hand_y - 45),
        (hand_x, hand_y + 45),
        (0, 230, 255),
        1
    )

    cv2.putText(
        frame,
        "ORANGE CURSOR",
        (hand_x - 80, hand_y - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 230, 255),
        2
    )

    return mouse_x, mouse_y


# -----------------------------
# Webcam setup
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

print("Gesture controller started.")
print("Gojo → Blue aura → Open ChatGPT")
print("Naruto → Orange aura → Hand cursor mode")
print("M → Exit cursor mode")
print("Q → Quit")

start_time = time.perf_counter()

previous_label = None
same_label_frames = 0
stable_label = "idle"

gesture_armed = True
last_action_message = "Ready"

effect_active = False
effect_name = None
effect_start_time = 0.0
EFFECT_DURATION = 2.4

naruto_cursor_mode = False

screen_width, screen_height = pyautogui.size()
mouse_x = screen_width // 2
mouse_y = screen_height // 2


# -----------------------------
# Main loop
# -----------------------------
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

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    height, width, _ = frame.shape
    hands_detected = len(result.hand_landmarks)

    draw_hand_skeleton(frame, result.hand_landmarks)

    confidence = 0.0

    # Gesture prediction
    if hands_detected >= 1:
        sample = make_sample(result.hand_landmarks)

        probabilities = classifier.predict_proba(sample)[0]
        best_index = probabilities.argmax()

        current_label = classifier.classes_[best_index]
        confidence = probabilities[best_index] * 100

        if confidence >= 70:
            if current_label == previous_label:
                same_label_frames += 1
            else:
                previous_label = current_label
                same_label_frames = 1

            if same_label_frames >= 8:
                stable_label = current_label

        else:
            stable_label = "uncertain"

    else:
        previous_label = None
        same_label_frames = 0
        stable_label = "no_hand"

    # Re-arm after returning to idle.
    if stable_label in ("idle", "no_hand", "uncertain"):
        gesture_armed = True

    # Gojo activation
    if (
        stable_label == "gojo"
        and gesture_armed
        and not effect_active
    ):
        naruto_cursor_mode = False
        effect_active = True
        effect_name = "gojo"
        effect_start_time = time.perf_counter()

        gesture_armed = False
        last_action_message = "DOMAIN EXPANSION ACTIVATED!"

    # Naruto activation
    elif (
        stable_label == "naruto"
        and gesture_armed
        and not effect_active
    ):
        effect_active = True
        effect_name = "naruto"
        effect_start_time = time.perf_counter()

        gesture_armed = False
        last_action_message = "CHAKRA MODE ACTIVATED!"

    # Play aura, then run the final action.
    if effect_active:
        elapsed = time.perf_counter() - effect_start_time

        if effect_name == "gojo":
            draw_gojo_aura(
                frame,
                result.hand_landmarks,
                elapsed,
                EFFECT_DURATION
            )

        elif effect_name == "naruto":
            draw_naruto_aura(
                frame,
                result.hand_landmarks,
                elapsed,
                EFFECT_DURATION
            )

        if elapsed >= EFFECT_DURATION:
            if effect_name == "gojo":
                webbrowser.open("https://chatgpt.com")
                last_action_message = "CHATGPT OPENED!"

            elif effect_name == "naruto":
                naruto_cursor_mode = True
                last_action_message = "NARUTO CURSOR MODE ON!"

            effect_active = False
            effect_name = None

    # Naruto hand-controlled cursor
    if naruto_cursor_mode and not effect_active:
        mouse_x, mouse_y = control_naruto_cursor(
            frame,
            result.hand_landmarks,
            mouse_x,
            mouse_y
        )

    # Normal UI
    if not effect_active:
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
            f"Gesture: {stable_label.upper()}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        mode_text = (
            "NARUTO CURSOR MODE: ON"
            if naruto_cursor_mode
            else "NARUTO CURSOR MODE: OFF"
        )

        cv2.putText(
            frame,
            mode_text,
            (20, height - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255) if naruto_cursor_mode else (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            last_action_message,
            (20, height - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "M: Exit cursor mode | Q: Quit",
            (20, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1
        )

    cv2.imshow("Anime Gesture Controller", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("m"):
        naruto_cursor_mode = False
        last_action_message = "NARUTO CURSOR MODE OFF"

cap.release()
cv2.destroyAllWindows()
landmarker.close()