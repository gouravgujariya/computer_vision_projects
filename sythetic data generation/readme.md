# Low-Light Image Segmentation with Swin Transformer

This repository provides a **low-light image segmentation** pipeline using **Swin Transformer** for classification-based segmentation and **histogram equalization** for low-light enhancement. The model runs efficiently on **CPU** and is deployed using **Gradio** for real-time inference.

## 🚀 Features
- **Low-light image enhancement** using OpenCV.
- **Swin Transformer-based segmentation** (CPU-optimized).
- **Gradio web interface** for real-time image processing.
- **Lightweight deployment** (no GPU required).

## 🛠️ Installation
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## 🔥 Usage
Run the application with:

```bash
python app.py
```

Then, open the provided Gradio link in your browser and upload an image for processing.

## 📦 Dependencies
- `torch`
- `gradio`
- `opencv-python`
- `numpy`
- `pillow`
- `transformers`

## 🎯 Model & Approach
1. **Low-Light Enhancement**: Uses histogram equalization to improve visibility in dark images.
2. **Swin Transformer**: Applies Swin-Tiny for classification-based segmentation.
3. **Gradio Interface**: Allows users to upload images and visualize results instantly.

## 📸 Example
Upload a low-light image, and the model will enhance and segment the clothing in real time.

## 🔗 Showcase Boost
"Synthetic Data + Vision Transformers for Low-Light Image Segmentation!"

