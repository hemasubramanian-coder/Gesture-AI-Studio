
import streamlit as st
import cv2
import mediapipe as mp
import time
import os
import ctypes
import platform
import pyautogui
from collections import Counter
from datetime import datetime

st.set_page_config(
    page_title="Gesture AI Studio",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #121b32);
    color: #edf2ff;
}

/* Sidebar visual design */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0a1020 0%,
        #101b31 100%
    );
    border-right: 1px solid rgba(117, 148, 205, 0.22);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
    padding-bottom: 1rem;
}

[data-testid="stSidebar"] h1 {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.7px;
    line-height: 1.3;
    margin-bottom: 0.4rem;
    color: #f3f6ff !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #8293b2 !important;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #9db1d4 !important;
    font-size: 0.73rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0.35rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
    background: rgba(25, 40, 67, 0.48);
    border: 1px solid rgba(111, 139, 188, 0.12);
    border-radius: 11px;
    padding: 0.48rem 0.65rem;
    margin: 0;
    transition: background 0.2s ease,
                border-color 0.2s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div > label:hover {
    background: rgba(49, 74, 118, 0.48);
    border-color: rgba(113, 155, 255, 0.5);
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div > label:has(input:checked) {
    background: linear-gradient(
        100deg,
        rgba(54, 91, 160, 0.55),
        rgba(32, 53, 89, 0.65)
    );
    border-color: rgba(113, 155, 255, 0.65);
    box-shadow: inset 3px 0 0 #80a8ff;
}

[data-testid="stSidebar"] [data-testid="stRadio"] p {
    color: #e4edff !important;
    font-weight: 600;
    font-size: 0.91rem;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(125, 151, 196, 0.22);
    margin-top: 1.5rem;
    margin-bottom: 1.4rem;
}

[data-testid="stSidebar"] h3 {
    color: #edf3ff !important;
    font-size: 0.88rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.055em;
    margin-bottom: 0.85rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #c6d5ef;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

/* Main heading */
h1, h2, h3 {
    color: #f2f5ff !important;
}

p, label, span {
    color: #c6d2e8;
}

/* Buttons */
div.stButton > button {
    border-radius: 10px;
    border: 1px solid #354a70;
    background: #1b2b49;
    color: white;
    min-height: 42px;
}

div.stButton > button:hover {
    background: #3159a6;
    border-color: #6b93ff;
    color: white;
}

/* Dashboard metric cards */
.metric-card {
    background: linear-gradient(
        135deg,
        #1c2c49 0%,
        #111d33 100%
    );
    padding: 24px 22px;
    border-radius: 18px;
    border: 1px solid rgba(112, 153, 231, 0.32);
    min-height: 145px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    transition: border-color 0.2s ease,
                transform 0.2s ease,
                box-shadow 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(125, 169, 255, 0.7);
    box-shadow: 0 12px 30px rgba(35, 75, 145, 0.2);
    transform: translateY(-3px);
}

.metric-label {
    color: #9eb4da;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.metric-value {
    color: #f5f7ff;
    font-size: 34px;
    font-weight: 800;
    margin-top: 14px;
    line-height: 1.25;
    overflow-wrap: anywhere;
}

.metric-description {
    color: #8da6d2;
    font-size: 12px;
    margin-top: 8px;
}

.small-muted {
    color: #94a7c8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

PHOTO_FOLDER = "captured_photos"
os.makedirs(PHOTO_FOLDER, exist_ok=True)

HOLD_SECONDS = 0.8
ACTION_COOLDOWN = 1.3

defaults = {
    "running": False,
    "counts": Counter(),
    "history": [],
    "last_gesture": "Waiting",
    "camera_index": 0,
    "camera": None,
    "navigation": "Dashboard",
    "action_candidate": None,
    "action_since": 0.0,
    "action_fired": False,
    "last_action_time": 0.0,
    "action_message": "No action performed yet",
    "actions_enabled": True
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to_live_detection():
    st.session_state.navigation = "Live Detection"


def go_to_gesture_guide():
    st.session_state.navigation = "Gesture Guide"


def release_camera():
    camera = st.session_state.get("camera")

    if camera is not None:
        camera.release()

    st.session_state.camera = None
    st.session_state.running = False
    st.session_state.action_candidate = None
    st.session_state.action_fired = False


def detect_gesture(landmarks):
    p = landmarks.landmark

    fingers = [
        p[8].y < p[6].y,
        p[12].y < p[10].y,
        p[16].y < p[14].y,
        p[20].y < p[18].y
    ]

    count = sum(fingers)

    thumb_up = p[4].y < p[3].y and count == 0
    thumb_down = p[4].y > p[3].y and count == 0

    if thumb_up:
        return "THUMBS UP"

    if thumb_down:
        return "THUMBS DOWN"

    if count == 4:
        return "OPEN PALM"

    if count == 1:
        return "ONE FINGER"

    if count == 2:
        return "PEACE SIGN"

    if count == 3:
        return "THREE FINGERS"

    if count == 0:
        return "FIST"

    return "HAND DETECTED"


def press_windows_key(key_code):
    if platform.system() != "Windows":
        raise OSError("System controls require Windows.")

    user32 = ctypes.windll.user32
    user32.keybd_event(key_code, 0, 0, 0)
    user32.keybd_event(key_code, 0, 2, 0)


def perform_gesture_action(gesture, photo_frame):
    try:
        if gesture == "ONE FINGER":
            pyautogui.press("right")
            return "Next slide command sent"

        elif gesture == "PEACE SIGN":
            pyautogui.press("left")
            return "Previous slide command sent"

        elif gesture == "THREE FINGERS":
            press_windows_key(0xAF)
            return "System volume increased"

        elif gesture == "THUMBS DOWN":
            press_windows_key(0xAD)
            return "System mute toggled"

        elif gesture == "OPEN PALM":
            press_windows_key(0xB3)
            return "Media play/pause command sent"

        elif gesture == "THUMBS UP":
            filename = datetime.now().strftime(
                "photo_%Y%m%d_%H%M%S_%f.jpg"
            )

            filepath = os.path.join(
                PHOTO_FOLDER, filename
            )

            success = cv2.imwrite(filepath, photo_frame)

            if success:
                return f"Photo saved: {filepath}"

            return "Photo capture failed"

        return "No action assigned"

    except Exception as error:
        return f"Action failed: {error}"


def record_action(gesture, action):
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Gesture": gesture,
        "Action": action
    })


with st.sidebar:
    st.markdown("# ✋ Gesture AI")
    st.caption("SMART VISION CONTROL")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Live Detection",
            "Gesture Guide",
            "Activity History",
            "Settings"
        ],
        key="navigation"
    )

    st.divider()
    st.markdown("### SYSTEM STATUS")

    st.write("🟢 Dashboard: Online")

    st.write(
        "🟢 Camera: Running"
        if st.session_state.running
        else "📷 Camera: Stopped"
    )

    st.write(
        "🟢 Gesture actions: Enabled"
        if st.session_state.actions_enabled
        else "⏸️ Gesture actions: Disabled"
    )

    st.caption("AI Hand Gesture Control System")


st.title("✋ Gesture AI Studio")
st.markdown("### Intelligent hand tracking and gesture control")
st.caption("Real-time computer vision workspace")


if page != "Live Detection" and st.session_state.running:
    release_camera()


if page == "Dashboard":
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">GESTURES DETECTED</div>
                <div class="metric-value">
                    {sum(st.session_state.counts.values())}
                </div>
                <div class="metric-description">
                    Total recognized gestures
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">UNIQUE GESTURES</div>
                <div class="metric-value">
                    {len(st.session_state.counts)}
                </div>
                <div class="metric-description">
                    Different gestures recognized
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">LATEST GESTURE</div>
                <div class="metric-value" style="font-size:26px;">
                    {st.session_state.last_gesture}
                </div>
                <div class="metric-description">
                    Most recently detected
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()
    st.subheader("Quick Actions")

    a, b = st.columns(2)

    with a:
        st.button(
            "Open Live Detection",
            use_container_width=True,
            on_click=go_to_live_detection
        )

    with b:
        st.button(
            "View Gesture Guide",
            use_container_width=True,
            on_click=go_to_gesture_guide
        )

    st.info(
        "Open Live Detection to start your webcam "
        "and try the six gesture controls."
    )

    st.subheader("Gesture Controls")

    controls = [
        ("☝️ ONE FINGER", "Next PowerPoint slide"),
        ("✌️ PEACE SIGN", "Previous PowerPoint slide"),
        ("🤟 THREE FINGERS", "Increase system volume"),
        ("👎 THUMBS DOWN", "Mute / Unmute"),
        ("🖐️ OPEN PALM", "Play / Pause media"),
        ("👍 THUMBS UP", "Capture and save photo")
    ]

    for name, action in controls:
        st.write(f"**{name}** — {action}")


elif page == "Live Detection":
    st.subheader("Live Camera Workspace")
    st.write(
        "Keep your hand clearly visible in front of the camera."
    )

    st.warning(
        "System controls send keyboard/media commands to Windows. "
        "Keep the intended application active when using slide controls."
    )

    camera_index = st.selectbox(
        "Camera",
        [0, 1, 2],
        index=st.session_state.camera_index
    )

    start_col, stop_col = st.columns(2)

    with start_col:
        start = st.button(
            "▶ Start Camera",
            use_container_width=True
        )

    with stop_col:
        stop = st.button(
            "■ Stop Camera",
            use_container_width=True
        )

    if start:
        release_camera()
        camera = cv2.VideoCapture(camera_index)

        if camera.isOpened():
            st.session_state.camera = camera
            st.session_state.camera_index = camera_index
            st.session_state.running = True
            st.session_state.action_candidate = None
            st.session_state.action_fired = False
        else:
            camera.release()
            st.error(
                "Camera open aagala. Close other camera apps "
                "and try again."
            )

    if stop:
        release_camera()
        st.rerun()

    @st.fragment(run_every=0.1)
    def live_camera():
        if not st.session_state.running:
            st.info("Press Start Camera to begin live detection.")
            return

        camera = st.session_state.camera

        if camera is None or not camera.isOpened():
            release_camera()
            st.error("Camera unavailable. Start it again.")
            return

        ok, frame = camera.read()

        if not ok:
            release_camera()
            st.error("Unable to read camera frame.")
            return

        frame = cv2.flip(frame, 1)
        photo_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gesture = "No hand detected"

        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        ) as hands:
            result = hands.process(rgb)

        if result.multi_hand_landmarks:
            drawing = mp.solutions.drawing_utils

            for hand_landmarks in result.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS
                )

                gesture = detect_gesture(hand_landmarks)

        if gesture != "No hand detected":
            if gesture != st.session_state.last_gesture:
                st.session_state.counts[gesture] += 1

                st.session_state.history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Gesture": gesture,
                    "Action": "Detected"
                })

            st.session_state.last_gesture = gesture

        now = time.monotonic()

        if gesture == "No hand detected":
            st.session_state.action_candidate = None
            st.session_state.action_fired = False

        elif gesture != st.session_state.action_candidate:
            st.session_state.action_candidate = gesture
            st.session_state.action_since = now
            st.session_state.action_fired = False

        elif (
            st.session_state.actions_enabled
            and not st.session_state.action_fired
            and now - st.session_state.action_since >= HOLD_SECONDS
            and now - st.session_state.last_action_time
            >= ACTION_COOLDOWN
        ):
            action = perform_gesture_action(
                gesture,
                photo_frame
            )

            st.session_state.action_message = action
            st.session_state.last_action_time = now
            st.session_state.action_fired = True

            record_action(gesture, action)

        st.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

        st.success("Camera is running")
        st.markdown(f"### Detected Gesture: `{gesture}`")

        st.info(
            f"Last action: {st.session_state.action_message}"
        )

        if not st.session_state.actions_enabled:
            st.warning(
                "Gesture actions are disabled. "
                "Enable them in Settings."
            )

    live_camera()


elif page == "Gesture Guide":
    st.subheader("Supported Gesture Guide")

    guides = [
        ("☝️", "ONE FINGER",
         "Raise your index finger. Next slide."),
        ("✌️", "PEACE SIGN",
         "Raise your index and middle fingers. Previous slide."),
        ("🤟", "THREE FINGERS",
         "Raise three fingers. Increase system volume."),
        ("👎", "THUMBS DOWN",
         "Point your thumb downward. Toggle system mute."),
        ("🖐️", "OPEN PALM",
         "Open your hand. Play or pause media."),
        ("👍", "THUMBS UP",
         "Raise your thumb. Save a webcam photo.")
    ]

    for icon, name, description in guides:
        with st.container(border=True):
            st.markdown(f"### {icon} {name}")
            st.write(description)
            st.caption(
                "Hold the gesture steadily for about 0.8 seconds."
            )


elif page == "Activity History":
    st.subheader("Gesture Activity History")

    if st.session_state.history:
        st.dataframe(
            list(reversed(st.session_state.history)),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Gesture Counts")

        if st.session_state.counts:
            st.bar_chart(
                dict(st.session_state.counts)
            )

        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.counts.clear()
            st.session_state.last_gesture = "Waiting"
            st.session_state.action_message = (
                "No action performed yet"
            )
            st.rerun()

    else:
        st.info(
            "No gesture activity recorded yet. "
            "Start live detection first."
        )


elif page == "Settings":
    st.subheader("Application Settings")

    st.checkbox(
        "Enable gesture actions",
        key="actions_enabled"
    )

    st.write("**Camera:** Select the camera in Live Detection.")
    st.write("**Detection:** MediaPipe hand landmark tracking.")
    st.write("**Theme:** Professional dark interface.")
    st.write(f"**Photo folder:** {os.path.abspath(PHOTO_FOLDER)}")
    st.write(f"**Gesture hold time:** {HOLD_SECONDS} seconds")
    st.write(f"**Action cooldown:** {ACTION_COOLDOWN} seconds")

    st.warning(
        "Windows system controls require Windows. "
        "Keep the intended application active for slide controls."
    )

    if st.button("Reset Gesture Statistics"):
        st.session_state.counts.clear()
        st.session_state.history = []
        st.session_state.last_gesture = "Waiting"
        st.session_state.action_message = (
            "No action performed yet"
        )
        st.success("Statistics reset.")


st.divider()
st.caption(
    "Gesture AI Studio • AI Hand Gesture Control System"
)
