import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Try WebRTC (optional)
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except:
    WEBRTC_AVAILABLE = False

# Page Config
st.set_page_config(page_title="FireGuard AI Pro", layout="wide")
st.title("🔥 Intelligent Fire Detection System")

# Load Model (CPU forced for stability)
@st.cache_resource
def load_model():
    model = YOLO("best_Final_model.pt")
    model.to("cpu")
    return model

model = load_model()

# Sidebar
st.sidebar.header("Control Panel")
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
area_threshold = st.sidebar.slider("Red Alert Threshold", 1000, 30000, 4000)

tab1, tab2 = st.tabs(["📷 Snapshot / Upload", "🎥 Live Monitor (Beta)"])

# ---------------- SNAPSHOT / UPLOAD (MAIN FEATURE) ----------------
with tab1:
    st.subheader("Upload or Capture Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    camera_img = st.camera_input("Or capture from camera")

    img = None

    if uploaded_file:
        img = Image.open(uploaded_file)
        img = np.array(img)

    elif camera_img:
        file_bytes = np.frombuffer(camera_img.getvalue(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        if img is None:
            st.error("Image processing failed.")
        else:
            img = img.astype(np.uint8)

            try:
                results = model.predict(img, conf=conf_threshold, verbose=False)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)

                        if area > area_threshold:
                            color, label = (0, 0, 255), "🔥 FIRE DETECTED"
                        else:
                            color, label = (0, 255, 255), "⚠️ SMALL FLAME"

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(img, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

            except Exception as e:
                st.error(f"Inference failed: {e}")

# ---------------- LIVE MONITOR (SAFE VERSION) ----------------
with tab2:
    st.subheader("Live Detection (Experimental)")

    if not WEBRTC_AVAILABLE:
        st.warning("Live video not supported in this environment.")
    else:
        try:
            class VideoProcessor(VideoProcessorBase):
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")

                    img = img.astype(np.uint8)

                    results = model.predict(img, conf=conf_threshold, verbose=False)

                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 2)

                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            RTC_CONFIG = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            webrtc_streamer(
                key="fire",
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )

        except Exception as e:
            st.error("Live video failed. Use snapshot mode instead.")
