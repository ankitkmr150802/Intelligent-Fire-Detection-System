import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# WebRTC and AV handling
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Page Configuration
st.set_page_config(page_title="FireGuard AI Pro", layout="wide")
st.title("🔥 Intelligent Fire Detection System")

# --- Model Loading (Optimized for Streamlit Cloud) ---
@st.cache_resource
def load_model():
    # Model name must exactly match your filename in GitHub
    model = YOLO("best_Final_model.pt")
    model.to("cpu")  # Forced CPU for cloud stability
    return model

model = load_model()

# --- Sidebar ---
st.sidebar.header("⚙️ Control Panel")
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.4)
area_threshold = st.sidebar.slider("Red Alert Threshold (px)", 1000, 30000, 4000)

tab1, tab2 = st.tabs(["📷 Snapshot / Upload", "🎥 Live Monitor (Beta)"])

# ---------------- TAB 1: SNAPSHOT / UPLOAD ----------------
with tab1:
    st.subheader("High-Res Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    with col2:
        camera_img = st.camera_input("Take a quick photo")

    input_img = None
    if uploaded_file:
        input_img = Image.open(uploaded_file).convert("RGB")
        input_img = np.array(input_img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    elif camera_img:
        file_bytes = np.frombuffer(camera_img.getvalue(), np.uint8)
        input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if input_img is not None:
        # Inference
        results = model.predict(input_img, conf=conf_threshold, verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                # Logic: Red for big fire, Yellow for small
                color, label = (0, 255, 255), "WARNING: FLAME"
                if area > area_threshold:
                    color, label = (0, 0, 255), "🚨 RED ALERT: FIRE"
                
                cv2.rectangle(input_img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(input_img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), 
                 caption="Detection Result", use_container_width=True)

# ---------------- TAB 2: LIVE MONITOR ----------------
with tab2:
    st.subheader("Real-Time Detection Feed")
    
    if not WEBRTC_AVAILABLE:
        st.error("Missing libraries (streamlit-webrtc or av). Check requirements.txt")
    else:
        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.frame_skip = 0 # To reduce CPU load

            def recv(self, frame):
                self.frame_skip += 1
                img = frame.to_ndarray(format="bgr24")

                # Process every 3rd frame to prevent "Connection Timeout"
                if self.frame_skip % 3 == 0:
                    results = model.predict(img, conf=conf_threshold, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # Simple box for live performance
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(img, "Fire", (x1, y1-5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Multi-STUN configuration for better connectivity
        RTC_CONFIG = RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun.services.mozilla.com"]}
            ]}
        )

        webrtc_streamer(
            key="fire-detection-live",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

st.sidebar.markdown("---")
st.sidebar.info("Tip: If Live Feed lags, try Snapshot mode for higher accuracy.")
