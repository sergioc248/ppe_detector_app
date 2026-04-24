from __future__ import annotations

from collections import Counter
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path(__file__).with_name("train-3_best.onnx")


@st.cache_resource
def load_model(model_path: Path) -> YOLO:
    return YOLO(str(model_path), task="detect")


def summarize_detections(result) -> list[dict[str, str | int | float]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = result.names
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    counts = Counter(int(class_id) for class_id in class_ids)

    summary = []
    for class_id, count in sorted(counts.items(), key=lambda item: names[item[0]]):
        matching_confidences = [
            confidence
            for detected_class_id, confidence in zip(class_ids, confidences, strict=True)
            if int(detected_class_id) == class_id
        ]
        summary.append(
            {
                "class": names[class_id],
                "count": count,
                "best_confidence": round(max(matching_confidences), 3),
            }
        )

    return summary


st.set_page_config(page_title="PPE Detector", layout="wide")

st.title("PPE Detector")
st.caption("Streamlit app backed by the exported `train-3_best.onnx` detector.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.stop()

with st.sidebar:
    st.header("Inference Settings")
    confidence_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("IoU threshold", 0.05, 0.95, 0.45, 0.05)
    image_size = st.select_slider("Image size", options=[320, 480, 640, 768, 960], value=640)
    st.info("This app runs the ONNX model through ONNX Runtime on CPU for compatibility.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.write("Upload a PPE image to run detection.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
model = load_model(MODEL_PATH)

with st.spinner("Running detection..."):
    results = model.predict(
        image,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=image_size,
        device="cpu",
        verbose=False,
    )

result = results[0]
annotated_image = Image.fromarray(result.plot()[..., ::-1])
summary = summarize_detections(result)

left, right = st.columns([1, 1])

with left:
    st.subheader("Original")
    st.image(image, use_container_width=True)

with right:
    st.subheader("Detections")
    st.image(annotated_image, use_container_width=True)

metric_cols = st.columns(3)
metric_cols[0].metric("Detections", len(result.boxes) if result.boxes is not None else 0)
metric_cols[1].metric("Classes found", len(summary))
metric_cols[2].metric("Model", MODEL_PATH.name)

st.subheader("Detection Summary")
if summary:
    st.dataframe(summary, use_container_width=True, hide_index=True)
else:
    st.write("No detections found for the current thresholds.")
