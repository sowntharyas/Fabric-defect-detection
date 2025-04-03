import streamlit as st
import cv2
import numpy as np
import torch
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

# Load YOLO model
MODEL_PATH = r"C:\Users\DELL\Documents\Final fabric\best copy.pt"
model = YOLO(MODEL_PATH)

st.set_page_config(layout="wide", page_title="Fabric Defect Detection")
st.title("🧵 Fabric Defect Detection Dashboard")

# Initialize session state for real-time updates
if "defect_data" not in st.session_state:
    st.session_state.defect_data = {
        "total_images": 0,
        "defect_count": 0,
        "non_defect_count": 0,
        "defect_summary": {}
    }

st.sidebar.header("Options")
option = st.sidebar.radio("Choose an option:", ["Single Image", "Multiple Images", "Upload Video", "Live Camera", "Dashboard"])

def detect_defects(image):
    results = model(image)
    output_image = results[0].plot()

    defect_summary = {}
    for box in results[0].boxes:
        cls = int(box.cls[0])
        defect_name = model.names[cls]
        defect_summary[defect_name] = defect_summary.get(defect_name, 0) + 1

    return output_image, defect_summary

def update_dashboard(total_images, defect_summary):
    st.session_state.defect_data["total_images"] += total_images
    defect_count = sum(defect_summary.values())
    st.session_state.defect_data["defect_count"] += defect_count
    st.session_state.defect_data["non_defect_count"] = max(0, st.session_state.defect_data["total_images"] - st.session_state.defect_data["defect_count"])

    for defect, count in defect_summary.items():
        st.session_state.defect_data["defect_summary"][defect] = st.session_state.defect_data["defect_summary"].get(defect, 0) + count

    st.rerun()  # Auto-refresh dashboard

def process_single_image():
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_image, defect_summary = detect_defects(img)
        
        st.image(output_image, caption="Processed Image", use_container_width=True)
        update_dashboard(1, defect_summary)

def process_multiple_images():
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        total_defects = {}
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output_image, img_defect_summary = detect_defects(img)

            for defect, count in img_defect_summary.items():
                total_defects[defect] = total_defects.get(defect, 0) + count

            st.image(output_image, caption="Processed Image", use_container_width=True)

        update_dashboard(len(uploaded_files), total_defects)

def process_uploaded_video():
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        st.video(uploaded_video)

        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video_path)
        total_frames, total_defects = 0, {}

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frame, frame_defect_summary = detect_defects(frame_rgb)

            for defect, count in frame_defect_summary.items():
                total_defects[defect] = total_defects.get(defect, 0) + count

            total_frames += 1
            stframe.image(output_frame, channels="RGB", use_container_width=True)
            update_dashboard(1, frame_defect_summary)

        cap.release()

def live_camera_detection():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    total_frames, total_defects = 0, {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frame, frame_defect_summary = detect_defects(frame_rgb)

        for defect, count in frame_defect_summary.items():
            total_defects[defect] = total_defects.get(defect, 0) + count

        total_frames += 1
        stframe.image(output_frame, channels="RGB", use_container_width=True)
        update_dashboard(1, frame_defect_summary)

    cap.release()

def display_dashboard():
    st.header("📊 Real-Time Fabric Defect Dashboard")
    
    st.metric(label="📷 Total Images Processed", value=st.session_state.defect_data["total_images"])
    st.metric(label="⚠️ Total Defects Detected", value=st.session_state.defect_data["defect_count"])
    st.metric(label="✅ Non-Defective Images", value=st.session_state.defect_data["non_defect_count"])

    defect_summary = st.session_state.defect_data["defect_summary"]
    if not defect_summary:
        defect_summary = {"No Defects Detected": 0}

    defect_summary_df = pd.DataFrame(
        list(defect_summary.items()),
        columns=["🛠️ Defect Type", "🔢 Count"]
    )
    st.subheader("Defect Breakdown")
    st.table(defect_summary_df)

    if st.session_state.defect_data["defect_count"] > 0:
        pie_chart = px.pie(
            names=list(st.session_state.defect_data["defect_summary"].keys()),
            values=list(st.session_state.defect_data["defect_summary"].values()),
            title="⚖️ Defect Distribution"
        )
        st.plotly_chart(pie_chart)

        bar_chart = px.bar(
            x=list(st.session_state.defect_data["defect_summary"].keys()),
            y=list(st.session_state.defect_data["defect_summary"].values()),
            title="📊 Defect Count by Type",
            labels={"x": "Defect Type", "y": "Count"}
        )
        st.plotly_chart(bar_chart)
    else:
        st.warning("No defects detected yet. Start processing images or videos!")

if option == "Single Image":
    process_single_image()
elif option == "Multiple Images":
    process_multiple_images()
elif option == "Upload Video":
    process_uploaded_video()
elif option == "Live Camera":
    live_camera_detection()
elif option == "Dashboard":
    display_dashboard()
