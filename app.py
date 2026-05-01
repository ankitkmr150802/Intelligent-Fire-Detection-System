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

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Make sure 'best_Final_model.pt' is in the same folder
    model = YOLO("best_Final_model.pt")
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
        # Prediction
        results = model.predict(input_img, conf=conf_threshold, verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                if area > area_threshold:
                    color, label = (0, 0, 255), "RED ALERT: FIRE"
                else:
                    color, label = (0, 255, 255), "WARNING: FLAME"
                
                cv2.rectangle(input_img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(input_img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), 
                 caption="Detection Result", use_container_width=True)

# ---------------- TAB 2: LIVE MONITOR ----------------
with tab2:
    if not WEBRTC_AVAILABLE:
        st.error("Missing libraries. Check requirements.txt")
    else:
        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.frame_skip = 0

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                self.frame_skip += 1
                
                # Inference only every 3rd frame to prevent lag on Cloud CPU
                if self.frame_skip % 3 == 0:
                    results = model.predict(img, conf=conf_threshold, verbose=False, imgsz=320)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            current_area = (x2 - x1) * (y2 - y1)
                            
                            color, label = ((0, 0, 255), "FIRE!") if current_area > area_threshold else ((0, 255, 255), "Flame")
                            
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="fire-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
