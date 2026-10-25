
import cv2
import mediapipe as mp
import pyautogui
import time
import os
from datetime import datetime
from collections import deque, Counter

CAMERA_INDEX = 0
COOLDOWN = 1.5
PHOTO_FOLDER = "captured_photos"

os.makedirs(PHOTO_FOLDER, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

gesture_history = deque(maxlen=7)
last_action_time = 0
last_gesture = "None"
action_message = "Show your hand"
photo_count = 0
paused = False


def count_fingers(hand):
    landmarks = hand.landmark
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    count = 0

    for tip, joint in finger_pairs:
        if landmarks[tip].y < landmarks[joint].y:
            count += 1

    return count


def recognize_gesture(fingers):
    gestures = {
        0: "FIST",
        1: "ONE FINGER",
        2: "PEACE SIGN",
        3: "THREE FINGERS",
        4: "OPEN PALM"
    }

    return gestures.get(fingers, "UNKNOWN")


def perform_action(gesture, frame):
    global photo_count, paused

    if gesture == "ONE FINGER":
        pyautogui.press("right")
        return "Next slide"

    elif gesture == "PEACE SIGN":
        pyautogui.press("left")
        return "Previous slide"

    elif gesture == "THREE FINGERS":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            PHOTO_FOLDER, f"photo_{timestamp}.jpg"
        )

        if cv2.imwrite(filename, frame):
            photo_count += 1
            return f"Photo saved: {filename}"

        return "Photo capture failed"

    elif gesture == "FIST":
        pyautogui.press("space")
        paused = not paused

        if paused:
            return "Play / Pause: Paused"

        return "Play / Pause: Resumed"

    elif gesture == "OPEN PALM":
        pyautogui.press("esc")
        return "Escape key pressed"

    return "No action"


if not cap.isOpened():
    print("ERROR: Cannot access webcam.")
    raise SystemExit

print("AI Hand Gesture Control System Started")
print("Press Q in the webcam window to quit.")

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            success, frame = cap.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_gesture = "None"
            current_time = time.time()

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                finger_count = count_fingers(hand)
                current_gesture = recognize_gesture(finger_count)
                gesture_history.append(current_gesture)

                if len(gesture_history) == gesture_history.maxlen:
                    most_common, occurrences = Counter(
                        gesture_history
                    ).most_common(1)[0]

                    if (
                        occurrences >= 6
                        and most_common != last_gesture
                        and current_time - last_action_time >= COOLDOWN
                    ):
                        action_message = perform_action(
                            most_common, frame
                        )
                        last_gesture = most_common
                        last_action_time = current_time

                cv2.putText(
                    frame,
                    f"Fingers: {finger_count}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2
                )

            else:
                gesture_history.clear()
                last_gesture = "None"

            cv2.rectangle(
                frame, (10, 70), (700, 220), (30, 30, 30), -1
            )

            cv2.putText(
                frame,
                f"Gesture: {current_gesture}",
                (25, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Action: {action_message[:55]}",
                (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Photos captured: {photo_count}",
                (25, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "AI Hand Gesture Control System", frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("AI Hand Gesture Control System Closed.")
