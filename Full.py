import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ultralytics import YOLO
from collections import Counter
import streamlit_webrtc
import av
from typing import List, NamedTuple

# Load YOLOv8 model (Ensure correct model path)
MODEL_PATH = "best copy.pt"
try:
    model = YOLO(MODEL_PATH)
    st.success(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Initialize session state for tracking defect statistics
if "defect_count" not in st.session_state:
    st.session_state.defect_count = 0
if "total_count" not in st.session_state:
    st.session_state.total_count = 0
if "defect_summary" not in st.session_state:
    st.session_state.defect_summary = Counter()
if "no_defect_count" not in st.session_state:
    st.session_state.no_defect_count = 0
if "camera_allowed" not in st.session_state:
    st.session_state.camera_allowed = False

# Function to process images
def process_image(image):
    if model is None:
        st.error("Model not loaded properly")
        return image, []
        
    results = model(image)  # Run YOLOv8 on image
    defect_types = []
    for r in results:
        im_array = r.plot()  # Annotate image
        defect_types = [model.names[int(box.cls[0])] for box in r.boxes]

    # Update defect count
    st.session_state.total_count += 1
    if defect_types:
        st.session_state.defect_count += 1
        st.session_state.defect_summary.update(defect_types)
    else:
        st.session_state.no_defect_count += 1

    return im_array, defect_types

# Function to process video frame by frame
def process_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()  # Placeholder to display video frames in real-time

    defect_count, total_count, no_defect_count = 0, 0, 0
    defect_types = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        results = model(frame)  # Run YOLO on frame
        frame_defects = []
        for r in results:
            im_array = r.plot()  # Annotate frame
            for box in r.boxes:
                class_id = int(box.cls[0])  
                defect_name = model.names[class_id]  
                frame_defects.append(defect_name)
                defect_types.append(defect_name)  

        total_count += 1
        if frame_defects:
            defect_count += 1
        else:
            no_defect_count += 1

        # Display processed frame in real-time
        stframe.image(im_array, channels="BGR", use_column_width=True)

    cap.release()

    # Update session state
    st.session_state.defect_count += defect_count
    st.session_state.total_count += total_count
    st.session_state.no_defect_count += no_defect_count
    st.session_state.defect_summary.update(defect_types)

# VideoProcessor class for live camera
class VideoProcessor:
    def __init__(self):
        self.defect_types = []

    def recv(self, frame):
        if model is None:
            return frame  # Return unprocessed frame if model failed to load
        
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame with YOLO
        try:
            # Optional: Resize for faster processing (uncomment if needed)
            # img = cv2.resize(img, (320, 320))
            
            results = model(img)
            frame_defects = []
            
            for r in results:
                img = r.plot()  # Annotate image with bounding boxes
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    defect_name = model.names[class_id]
                    frame_defects.append(defect_name)
                    self.defect_types.append(defect_name)
            
            # Update session state
            st.session_state.total_count += 1
            if frame_defects:
                st.session_state.defect_count += 1
                st.session_state.defect_summary.update(frame_defects)
            else:
                st.session_state.no_defect_count += 1
            
            # Add defect labels to the frame for real-time feedback
            if frame_defects:
                cv2.putText(img, f"Defects: {', '.join(frame_defects)}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(img, "No Defects", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        except Exception as e:
            st.error(f"Error processing frame: {e}")
            return frame
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Function to use WebRTC for live camera access
def process_live_camera():
    st.subheader("📸 Live Camera Defect Detection")
    
    st.info("📱 This feature requires camera access. Please allow permissions when prompted by your browser.")
    
    # Placeholder for real-time defect summary
    defect_status = st.empty()
    
    # Create WebRTC streamer
    webrtc_ctx = streamlit_webrtc.webrtc_streamer(
        key="fabric-defect-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.session_state.camera_allowed = True
        st.success("✅ Camera connected successfully! Detecting fabric defects in real-time.")
        
        # Display real-time defect summary
        processor = webrtc_ctx.video_processor
        if processor and processor.defect_types:
            defect_status.error(f"⚠️ Defects detected: {', '.join(set(processor.defect_types[-10:]))}")
        else:
            defect_status.success("✅ No defects detected yet")
    else:
        if st.session_state.camera_allowed:
            st.warning("Camera disconnected.")
            defect_status.empty()
        else:
            st.warning("Waiting for camera permission...")

# Function to process multiple images
def process_multiple_images(uploaded_files):
    if not uploaded_files:
        st.error("No image files uploaded.")
        return
    
    st.success(f"Found {len(uploaded_files)} images to process.")
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Process each image
    results_container = st.expander("Detailed Results", expanded=True)
    
    with results_container:
        for i, uploaded_file in enumerate(uploaded_files):
            # Read image
            image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
            if image is None:
                st.warning(f"Could not read image: {uploaded_file.name}")
                continue
            
            # Process image
            processed_img, defects = process_image(image)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(processed_img, channels="BGR", caption=f"File: {uploaded_file.name}")
            with col2:
                if defects:
                    st.error(f"⚠️ Defects found: {', '.join(defects)}")
                else:
                    st.success("✅ No defects detected")
            
            st.markdown("---")
            
            # Update progress bar
            progress_bar.progress((i + 1) / len(uploaded_files))
    
    st.success(f"Completed processing {len(uploaded_files)} images.")

# Function to display dashboard statistics
def display_dashboard():
    if st.session_state.total_count > 0:
        defect_percentage = (st.session_state.defect_count / st.session_state.total_count) * 100
        no_defect_percentage = (st.session_state.no_defect_count / st.session_state.total_count) * 100
    else:
        defect_percentage = 0
        no_defect_percentage = 0

    st.subheader("📊 Defect Analysis Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images Processed", st.session_state.total_count)
    col2.metric("Defective Images", st.session_state.defect_count)
    col3.metric("Non-Defective Images", st.session_state.no_defect_count)
    col4.metric("Defect Rate", f"{defect_percentage:.2f}%")

    # Pie chart showing defect vs no defect percentages
    if st.session_state.total_count > 0:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.pie(
            [defect_percentage, no_defect_percentage], 
            labels=["Defective", "Non-Defective"], 
            autopct="%1.1f%%", 
            colors=["red", "green"],
            explode=(0.1, 0),  # Explode the defect slice for emphasis
            shadow=True,
            startangle=90
        )
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.set_title("Defect vs Non-Defect Distribution", fontsize=16, pad=20)
        st.pyplot(fig1)
    else:
        st.info("No data available for visualization. Process some images first.")

    # Bar chart
    if st.session_state.total_count > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        categories = ["Defective", "Non-Defective"]
        values = [defect_percentage, no_defect_percentage]
        colors = ["red", "green"]
        
        bars = ax2.bar(categories, values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)
        
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Percentage (%)", fontsize=12)
        ax2.set_title("Image Quality Distribution", fontsize=16, pad=20)
        st.pyplot(fig2)

    # Display defect summary
    st.subheader("📌 Defect Type Summary")
    defect_summary = st.session_state.defect_summary.most_common()
    if defect_summary:
        df_summary = pd.DataFrame(defect_summary, columns=["Defect Type", "Count"])
        
        # Add percentage column
        total_defects = sum(count for _, count in defect_summary)
        df_summary["Percentage"] = (df_summary["Count"] / total_defects * 100).round(2).astype(str) + '%'
        
        st.table(df_summary)
        
        # Horizontal bar chart for defect types
        fig3, ax3 = plt.subplots(figsize=(10, max(4, len(defect_summary) * 0.5)))
        defect_types = [defect for defect, _ in defect_summary]
        defect_counts = [count for _, count in defect_summary]
        
        # Sort by count
        sorted_indices = np.argsort(defect_counts)
        defect_types = [defect_types[i] for i in sorted_indices]
        defect_counts = [defect_counts[i] for i in sorted_indices]
        
        y_pos = np.arange(len(defect_types))
        ax3.barh(y_pos, defect_counts, color='salmon')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(defect_types)
        ax3.invert_yaxis()  # Labels read top-to-bottom
        ax3.set_xlabel('Count')
        ax3.set_title('Defect Types Distribution')
        
        # Add count labels
        for i, v in enumerate(defect_counts):
            ax3.text(v + 0.1, i, str(v), va='center')
            
        st.pyplot(fig3)
    else:
        st.success("✅ No defects detected in any processed images!")
        
        # Create a simple visualization for the no-defect case
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.bar(["Non-Defective"], [100], color='green', alpha=0.7)
        ax4.set_ylim(0, 100)
        ax4.set_ylabel("Percentage (%)")
        ax4.set_title("100% Defect-Free Images")
        ax4.text(0, 50, "Perfect Quality!", ha='center', va='center', fontsize=20, rotation=0, color='darkgreen')
        st.pyplot(fig4)

# Streamlit UI
st.title("🔍 Fabric Defect Detection using YOLOv8")

st.sidebar.title("📋 Setup Requirements")
st.sidebar.info("""
This app uses YOLOv8 for fabric defect detection. 
Make sure to install the following packages:
""")

tab1, tab2 = st.tabs(["Detection", "Dashboard"])

with tab1:
    option = st.sidebar.radio("Choose Input Type", ("Single Image", "Multiple Images", "Video", "Live Camera"))

    if option == "Single Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            image = np.array(cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1))
            processed_image, defects = process_image(image)
            st.image(processed_image, channels="BGR", use_container_width=True)
            if defects:
                st.error(f"⚠️ Defects detected: {', '.join(defects)}")
            else:
                st.success("✅ No defects detected in this image!")
    
    elif option == "Multiple Images":
        st.info("Select multiple images to upload (hold Ctrl/Cmd to select multiple files)")
        uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files and st.button("Process Selected Images"):
            with st.spinner("Processing multiple images..."):
                process_multiple_images(uploaded_files)

    elif option == "Video":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_video.read())
                process_video_stream(temp_file.name)

    elif option == "Live Camera":
        process_live_camera()

with tab2:
    display_dashboard()