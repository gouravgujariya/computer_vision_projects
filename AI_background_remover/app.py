import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from pandas
# Load BiRefNet model (for background segmentation)
device = "cuda" if torch.cuda.is_available() else "cpu"
birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(device)

# Preprocessing pipeline for images
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to remove background from an image
def remove_bg(image):
    image = image.convert("RGB")
    orig_size = image.size
    input_tensor = transform_image(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()
    mask = preds[0].squeeze().numpy()

    # Convert mask to proper format
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, orig_size)

    # Apply mask to image
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    image_np[:, :, 3] = mask  # Set alpha channel

    return Image.fromarray(image_np)

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
        out.write(cv2.cvtColor(np.array(bg_removed), cv2.COLOR_RGBA2BGRA))

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
