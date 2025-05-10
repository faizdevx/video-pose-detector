import streamlit as st
import tempfile
import os
from process_video import process_video

st.set_page_config(page_title="Pose + YOLO Video Processor", layout="centered")
st.title("📹 Pose + YOLOv8 Video Processor")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    st.info("⏳ Processing your video...")
    
    try:
        processed_path = process_video(temp_input_path, output_path)
        st.success("✅ Video processing complete!")
        with open(processed_path, "rb") as f:
            st.download_button("📥 Download Processed Video", f, file_name="processed_output.mp4")
    except Exception as e:
        st.error(f"Error: {str(e)}")
