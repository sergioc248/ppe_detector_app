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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg-a: #f7f3e9;
        --bg-b: #dff3ee;
        --ink: #18222f;
        --muted: #4b5a6b;
        --card: rgba(255, 255, 255, 0.78);
        --line: rgba(24, 34, 47, 0.14);
        --accent: #0f766e;
        --accent-soft: #d1fae5;
    }

    .stApp {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ink);
        background:
            radial-gradient(circle at 8% 10%, rgba(255, 190, 129, 0.40) 0%, rgba(255, 190, 129, 0.0) 34%),
            radial-gradient(circle at 92% 12%, rgba(45, 212, 191, 0.28) 0%, rgba(45, 212, 191, 0.0) 38%),
            linear-gradient(125deg, var(--bg-a) 0%, var(--bg-b) 100%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.4rem;
        max-width: 1180px;
    }

    .hero {
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1.1rem;
        background: linear-gradient(160deg, rgba(255, 255, 255, 0.84), rgba(255, 255, 255, 0.68));
        backdrop-filter: blur(4px);
        box-shadow: 0 14px 32px rgba(24, 34, 47, 0.08);
        animation: rise-in 550ms ease-out both;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(1.8rem, 2.6vw, 2.55rem);
        letter-spacing: -0.03em;
        line-height: 1.08;
    }

    .hero p {
        margin: 0.55rem 0 0;
        color: var(--muted);
        font-size: 1rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(24, 34, 47, 0.12);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.92) 0%, rgba(242, 252, 249, 0.92) 100%);
    }

    [data-testid="stSidebar"] * {
        font-family: 'Space Grotesk', sans-serif;
    }

    .panel {
        border: 1px solid var(--line);
        border-radius: 18px;
        background: var(--card);
        padding: 0.95rem 1rem 0.8rem;
        box-shadow: 0 8px 22px rgba(24, 34, 47, 0.07);
        animation: rise-in 500ms ease-out both;
    }

    .panel h3 {
        margin-top: 0;
        letter-spacing: -0.01em;
    }

    [data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.7rem 0.85rem;
        background: rgba(255, 255, 255, 0.80);
        box-shadow: 0 8px 18px rgba(24, 34, 47, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 700;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(15, 118, 110, 0.30);
        color: #0b5f57;
        background: var(--accent-soft);
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .mono {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.88rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(24, 34, 47, 0.06);
    }

    @keyframes rise-in {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (max-width: 820px) {
        .block-container {
            padding-top: 0.85rem;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        .hero {
            padding: 1rem;
            border-radius: 16px;
        }

        .panel {
            padding: 0.85rem;
            border-radius: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="pill">Safety Vision System</div>
      <h1>PPE Detector</h1>
      <p>Upload an image to detect personal protective equipment with your ONNX model.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

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
    st.markdown(
        f"""
        <div class="panel">
            <div class="pill">Ready</div>
            <p style="margin:0.2rem 0 0.2rem; color: var(--muted);">
                Upload a PPE image to run detection. Model in use:
                <span class="mono">{MODEL_PATH.name}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Original")
    st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Detections")
    st.image(annotated_image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

metric_cols = st.columns(3)
metric_cols[0].metric("Detections", len(result.boxes) if result.boxes is not None else 0)
metric_cols[1].metric("Classes found", len(summary))
metric_cols[2].metric("Model", MODEL_PATH.name)

st.markdown('<div class="panel" style="margin-top: 0.85rem;">', unsafe_allow_html=True)
st.subheader("Detection Summary")
if summary:
    st.dataframe(summary, use_container_width=True, hide_index=True)
else:
    st.write("No detections found for the current thresholds.")
st.markdown("</div>", unsafe_allow_html=True)
