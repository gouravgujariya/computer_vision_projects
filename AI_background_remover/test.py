import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# Load RMBG-1.4 model for background removal
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

# Function to remove background from an image
def remove_bg(image):
    image = image.convert("RGB")

    # Get segmented mask and processed image
    mask = pipe(image, return_mask=True)  # Get segmentation mask
    processed_image = pipe(image)  # Get image with mask applied

    return processed_image

# Function to process video (frame-by-frame background removal)
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = "output_video.mp4"
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height), True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        bg_removed = remove_bg(pil_img)
        
        # Convert back to BGR (no transparency for video)
        frame_out = cv2.cvtColor(np.array(bg_removed.convert("RGB")), cv2.COLOR_RGB2BGR)
        out.write(frame_out)

    cap.release()
    out.release()
    return out_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ AI Background Removal & Replacement")

    with gr.Tab("Image"):
        image_input = gr.Image(label="Upload an Image", type="pil")
        image_output = gr.Image(label="Background Removed Image", type="pil")
        remove_btn = gr.Button("Remove Background")
        remove_btn.click(remove_bg, inputs=image_input, outputs=image_output)

    with gr.Tab("Video"):
        video_input = gr.Video(label="Upload a Video")
        video_output = gr.Video(label="Processed Video")
        video_btn = gr.Button("Process Video")
        video_btn.click(process_video, inputs=video_input, outputs=video_output)

demo.launch()
