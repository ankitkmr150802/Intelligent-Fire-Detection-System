import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Page Config
st.set_page_config(page_title="FireGuard AI", layout="centered")
st.title("🔥 Intelligent Fire Detection System")
st.markdown("Real-time monitoring using YOLO11")

# Load Model
@st.cache_resource
def load_model():
    return YOLO('best_Final_model.pt')

model = load_model()

# Sidebar for Settings
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
area_threshold = st.sidebar.slider("Red Alert Area Threshold", 1000, 20000, 8000)

# Camera Input (Works on Mobile/Laptop)
img_file_buffer = st.camera_input("Take a picture or use live feed")

if img_file_buffer is not None:
    # Convert buffer to image
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Inference
    results = model.predict(cv2_img, conf=conf_threshold, verbose=False)
    
    # Process Results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            area = w * h
            
            if area < area_threshold:
                label, color = "WARNING: SMALL FLAME", (255, 255, 0) # Yellow (RGB)
            else:
                label, color = "ALERT: FIRE!", (255, 0, 0) # Red (RGB)
            
            # Draw on Image
            cv2.rectangle(cv2_img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(cv2_img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display Result
    st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption="Detection Result")