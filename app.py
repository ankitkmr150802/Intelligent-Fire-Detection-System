import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av


# Page Configuration with latest 2026 standards
st.set_page_config(page_title="FireGuard AI Pro", layout="wide")
st.title("Intelligent Fire Detection System")

# Load Model
@st.cache_resource
def load_model():
    return YOLO('best_Final_model.pt')

model = load_model()

# Sidebar Control Panel
st.sidebar.header("Control Panel")
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.40)
area_threshold = st.sidebar.slider("Red Alert Threshold", 1000, 30000, 4000)

# Implementation of Dual-Mode Tabs
tab1, tab2 = st.tabs(["Real-Time Monitor", "Snapshot Analysis"])

# --- TAB 1: REAL-TIME DETECTION (Optimized for 2026) ---
with tab1:
    st.subheader("Live Video Feed")
    
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Efficient inference
            results = self.model.predict(img, conf=conf_threshold, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Alert Logic based on fire size
                    if area > area_threshold:
                        color, label = (0, 0, 255), "RED ALERT: FIRE"
                    else:
                        color, label = (0, 255, 255), "WARNING: SMALL FLAME"
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # STUN configuration for web deployment
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="fire-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# --- TAB 2: SNAPSHOT ANALYSIS (Warning Free) ---
with tab2:
    st.subheader("High-Resolution Snapshot Analysis")
    img_file_buffer = st.camera_input("Take a photo for detailed check")
    
    if img_file_buffer is not None:
        file_bytes = np.frombuffer(img_file_buffer.getvalue(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # High resolution prediction
        results = model.predict(img, conf=conf_threshold, verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                color = (0, 0, 255) if area > area_threshold else (0, 255, 255)
                label = "FIRE DETECTED" if area > area_threshold else "WARNING"
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Fixed: Replaced use_container_width with width='stretch' as per latest API
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width='stretch')
