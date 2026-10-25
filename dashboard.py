
import streamlit as st
import cv2
import mediapipe as mp
import av
import threading
from collections import Counter
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(
    page_title="Gesture AI Studio",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #121b32);
    color: #edf2ff;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1020, #101b31);
}
h1, h2, h3 {
    color: #f2f5ff !important;
}
.metric-card {
    background: linear-gradient(135deg, #1c2c49, #111d33);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(112,153,231,.32);
    min-height: 125px;
}
.metric-label {
    color: #9eb4da;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}
.metric-value {
    color: #f5f7ff;
    font-size: 30px;
    font-weight: 800;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

GESTURES = {
    "ONE FINGER": ("☝️", "Next slide"),
    "PEACE SIGN": ("✌️", "Previous slide"),
    "THREE FINGERS": ("🤟", "Increase volume"),
    "THUMBS DOWN": ("👎", "Mute / unmute"),
    "OPEN PALM": ("🖐️", "Play / pause media"),
    "THUMBS UP": ("👍", "Capture gesture event"),
}

if "history" not in st.session_state:
    st.session_state.history = []

if "counts" not in st.session_state:
    st.session_state.counts = Counter()

if "latest" not in st.session_state:
    st.session_state.latest = "Waiting"


class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = "No hand detected"
        self.last_seen = None
        self.counts = Counter()
        self.history = []

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.drawing = mp.solutions.drawing_utils

    @staticmethod
    def detect_gesture(hand_landmarks):
        p = hand_landmarks.landmark

        fingers = [
            p[8].y < p[6].y,
            p[12].y < p[10].y,
            p[16].y < p[14].y,
            p[20].y < p[18].y,
        ]

        count = sum(fingers)

        if count == 4:
            return "OPEN PALM"
        if count == 1:
            return "ONE FINGER"
        if count == 2:
            return "PEACE SIGN"
        if count == 3:
            return "THREE FINGERS"
        if count == 0:
            thumb_tip = p[4]
            thumb_joint = p[3]

            if thumb_tip.y < thumb_joint.y:
                return "THUMBS UP"
            return "THUMBS DOWN"

        return "HAND DETECTED"

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        gesture = "No hand detected"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                )
                gesture = self.detect_gesture(hand_landmarks)

        with self.lock:
            self.latest = gesture

            if gesture != "No hand detected" and gesture != self.last_seen:
                action = GESTURES.get(
                    gesture, ("", "Gesture detected")
                )[1]

                self.counts[gesture] += 1
                self.history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Gesture": gesture,
                    "Action": action,
                })

                self.last_seen = gesture

            elif gesture == "No hand detected":
                self.last_seen = None

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def snapshot(self):
        with self.lock:
            return (
                self.latest,
                self.counts.copy(),
                list(self.history),
            )


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
            "Settings",
        ],
    )

    st.divider()
    st.markdown("### SYSTEM STATUS")
    st.write("🟢 Dashboard: Online")
    st.caption("Camera access requires browser permission.")
    st.caption("Cloud mode displays detected gestures and actions.")

st.title("✋ Gesture AI Studio")
st.markdown("### Intelligent hand tracking and gesture recognition")
st.caption("Real-time computer vision workspace")

if page == "Dashboard":
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">GESTURES DETECTED</div>'
            f'<div class="metric-value">'
            f'{sum(st.session_state.counts.values())}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">UNIQUE GESTURES</div>'
            f'<div class="metric-value">'
            f'{len(st.session_state.counts)}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">LATEST GESTURE</div>'
            f'<div class="metric-value">'
            f'{st.session_state.latest}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Gesture Controls")

    for name, (emoji, action) in GESTURES.items():
        st.write(f"**{emoji} {name}**")
        st.write(f"Action: {action}")

    st.info("Open Live Detection and start the camera.")

elif page == "Live Detection":
    st.subheader("Live Camera Workspace")
    st.write("Click START and allow camera access in your browser.")

    st.warning(
        "Hosted mode detects gestures and displays the mapped action. "
        "It cannot directly press keys or control media on your laptop."
    )

    ctx = webrtc_streamer(
        key="gesture-ai-live",
        video_processor_factory=GestureProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
        rtc_configuration=RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        }),
    )

    if ctx.video_processor:
        gesture, counts, history = ctx.video_processor.snapshot()

        st.session_state.latest = gesture
        st.session_state.counts = counts
        st.session_state.history = history

        st.markdown(f"### Detected Gesture: `{gesture}`")

        if gesture in GESTURES:
            st.success(
                f"{GESTURES[gesture][0]} Action: "
                f"{GESTURES[gesture][1]}"
            )
        else:
            st.info("Show a supported gesture to the camera.")
    else:
        st.info("Camera is waiting to start.")

elif page == "Gesture Guide":
    st.subheader("Supported Gesture Guide")

    for name, (emoji, action) in GESTURES.items():
        with st.container(border=True):
            st.markdown(f"### {emoji} {name}")
            st.write(f"**Action:** {action}")
            st.caption(
                "Hold your hand steadily in front of the camera."
            )

elif page == "Activity History":
    st.subheader("Gesture Activity History")

    if st.session_state.history:
        st.dataframe(
            list(reversed(st.session_state.history)),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Gesture Counts")
        st.bar_chart(dict(st.session_state.counts))

        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.counts = Counter()
            st.session_state.latest = "Waiting"
            st.rerun()
    else:
        st.info(
            "No gesture activity recorded yet. "
            "Start Live Detection first."
        )

elif page == "Settings":
    st.subheader("Application Settings")
    st.write("**Detection:** MediaPipe hand landmark tracking")
    st.write("**Camera:** Browser camera using WebRTC")
    st.write("**Theme:** Professional dark interface")

    st.warning(
        "Keyboard and media actions are displayed as mapped commands. "
        "A hosted cloud app cannot directly control your laptop."
    )

st.divider()
st.caption("Gesture AI Studio • AI Hand Gesture Recognition System")
