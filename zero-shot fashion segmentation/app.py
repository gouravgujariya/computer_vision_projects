import gradio as gr
import cv2
import requests
import os

from ultralytics import YOLO

file_urls = ["https://www.dropbox.com/scl/fi/i582tgw95r0f8i8cssmx0/images.jpg?rlkey=fa3d74yaj0bh941jo67n0elns&dl=0"
    # 'https://www.dropbox.com/scl/fi/z4tnnills03s1o4evbqpl/download.jpg?rlkey=gmh63kexnnjcva6ahzo2wfmsd&dl=0'
]

def download_file(url, save_name):
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

for i, url in enumerate(file_urls):
    if 'mp4' in file_urls[i]:
        download_file(
            file_urls[i],
            f"video.mp4"
        )
    else:
        download_file(
            file_urls[i],
            f"image_{i}.jpg"
        )

model = YOLO('best.pt')
path  = [['images.jpg'],['images_1.png'],['image/i1.png'],['image/i2.png'],['image/i3.png']]
#          ,['image/i4.png'],['image/i5.png'],['image/i6.png'],['image/i7.png'],['image/i8.png'],['image/i9.png'],['image/i10.png'],['image/i11.png'],
#          ['image/i12.png'],['image/i13.png'],['image/i14.png'],['image/i15.png'],['image/i16.png'],['image/i17.png'],['image/i18.png'],['image/i19.png'],
#          ['image/i20.png'],['image/i21.png'],['image/i22.png'],['image/i23.png'],['image/i24.png'],['image/i25.png'],['image/i26.png'],['image/i27.png'],['image/i28.png']]
# video_path = [['video.mp4']]

def show_preds_image(image_path):
    image = cv2.imread(image_path)
    outputs = model.predict(source=image_path)
    results = outputs[0].cpu().numpy()
    for i, det in enumerate(results.boxes.xyxy):
        cv2.rectangle(
            image,
            (int(det[0]), int(det[1])),
            (int(det[2]), int(det[3])),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

inputs_image = [
    gr.components.Image(type="filepath", label="Input Image"),
]
outputs_image = [
    gr.components.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=inputs_image,
    outputs=outputs_image,
    title="Airport Luggage Weapon Detector app",
    examples=path,
    cache_examples=False,
)

def show_preds_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_copy = frame.copy()
            outputs = model.predict(source=frame)
            results = outputs[0].cpu().numpy()
            for i, det in enumerate(results.boxes.xyxy):
                cv2.rectangle(
                    frame_copy,
                    (int(det[0]), int(det[1])),
                    (int(det[2]), int(det[3])),
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
            yield cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

inputs_video = [
    gr.components.Video(),

]
outputs_video = [
    gr.components.Image(),
]
interface_video = gr.Interface(
    fn=show_preds_video,
    inputs=inputs_video,
    outputs=outputs_video,
    title="Airport Luggage Weapon Detector",
    cache_examples=False,
)

MORE = """ ## TRY Other Models
        ![imagea](image_path 'image/i5.png')
        ### Instant Image: 4k images in 5 Second -> https://huggingface.co/spaces/KingNish/Instant-Image
        """

gr.Markdown(MORE)

gr.TabbedInterface(
    [interface_image, interface_video],
    tab_names=['Image inference', 'Video inference']
).queue().launch()