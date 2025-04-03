import streamlit as st
import cv2
import numpy as np
import os
import time
import tempfile
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

# Load YOLO Model
model = YOLO("best copy.pt")  # Replace with your trained model path

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Initialize dashboard data
if 'defect_counts' not in st.session_state:
    st.session_state.defect_counts = {}
    st.session_state.total_images = 0
    st.session_state.total_defective = 0
    st.session_state.total_non_defective = 0

# Login Function
def login():
    st.session_state.logged_in = True

def logout():
    st.session_state.logged_in = False

def detect_defects(image):
    results = model(image)
    annotated_image = results[0].plot()  # Get image with annotations
    defect_counts = {}
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            label = result.names[cls]
            defect_counts[label] = defect_counts.get(label, 0) + 1
    
    return annotated_image, defect_counts

def update_dashboard(defect_counts):
    for defect, count in defect_counts.items():
        if defect in st.session_state.defect_counts:
            st.session_state.defect_counts[defect] += count
        else:
            st.session_state.defect_counts[defect] = count
    
    st.session_state.total_images += 1
    if defect_counts:
        st.session_state.total_defective += 1
    else:
        st.session_state.total_non_defective += 1

# Sidebar for Login & Logout
st.sidebar.title("User Authentication")
if not st.session_state.logged_in:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == "admin" and password == "password":  # Replace with a secure method
            login()
            st.sidebar.success("Logged in successfully!")
        else:
            st.sidebar.error("Invalid credentials")
else:
    st.sidebar.button("Logout", on_click=logout)

# If not logged in, stop execution
if not st.session_state.logged_in:
    st.stop()

# Dashboard Page
st.title("Fabric Defect Detection Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Images Processed", st.session_state.total_images)
col2.metric("Defective Images", st.session_state.total_defective)
col3.metric("Non-Defective Images", st.session_state.total_non_defective)

defect_df = pd.DataFrame(st.session_state.defect_counts.items(), columns=["Defect Name", "Count"])
if not defect_df.empty:
    st.subheader("Defect Summary Table")
    st.dataframe(defect_df)
    pie_chart = px.pie(defect_df, names="Defect Name", values="Count", title="Defective Percentage")
    st.plotly_chart(pie_chart)
    bar_chart = px.bar(defect_df, x="Defect Name", y="Count", title="Defect Count", color="Defect Name")
    st.plotly_chart(bar_chart)

# File Upload for Images & Videos
st.sidebar.subheader("Upload Files")
option = st.sidebar.radio("Select Type", ["Single Image", "Multiple Images", "Video Upload", "Live Camera"])

if option == "Single Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        annotated_image, defects = detect_defects(image)
        update_dashboard(defects)
        st.image(annotated_image, caption="Detected Defects", use_container_width=True)

elif option == "Multiple Images":
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            annotated_image, defects = detect_defects(image)
            update_dashboard(defects)
            st.image(annotated_image, caption=f"Detected Defects - {uploaded_file.name}", use_container_width=True)

elif option == "Video Upload":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_video.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())
        cap = cv2.VideoCapture(temp_video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame, defects = detect_defects(frame)
            update_dashboard(defects)
            st.image(annotated_frame, caption="Detected Defects in Video", use_container_width=True)
        cap.release()

elif option == "Live Camera":
    st.subheader("Live Camera Feed")
    cap = cv2.VideoCapture(0)  # Use 1 for external camera, or change for mobile IP camera
    frame_window = st.image([])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, defects = detect_defects(frame)
        update_dashboard(defects)
        frame_window.image(annotated_frame, channels="BGR")
        time.sleep(0.05)  # Adjust for smooth streaming
    cap.release()
