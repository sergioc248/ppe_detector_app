from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
    ULTRALYTICS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - deployment dependency guard
    YOLO = None  # type: ignore[assignment]
    ULTRALYTICS_IMPORT_ERROR = exc


MODEL_PATH = Path(__file__).with_name("train-3_best_float16.tflite")
DEFAULT_IMGSZ = 512


@st.cache_resource
def load_model(model_path: Path):
    if YOLO is None:
        raise RuntimeError("Ultralytics no esta disponible en este entorno.") from ULTRALYTICS_IMPORT_ERROR
    return YOLO(str(model_path), task="detect")


def extract_detections(result, img_width: int, img_height: int) -> list[dict[str, str | int | float]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = result.names
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    xyxy = boxes.xyxy.tolist()

    detections = []
    for idx, (class_id, confidence, coords) in enumerate(zip(class_ids, confidences, xyxy, strict=True), start=1):
        rx1, ry1, rx2, ry2 = coords
        
        # FIX: Algunos modelos devuelven las coordenadas ya normalizadas [0, 1] 
        # en lugar de absolutas. Si estan normalizadas, las reescalamos al ancho/alto.
        if abs(rx2) <= 2.0 and abs(ry2) <= 2.0:
            x1, x2 = rx1 * img_width, rx2 * img_width
            y1, y2 = ry1 * img_height, ry2 * img_height
        else:
            x1, x2 = rx1, rx2
            y1, y2 = ry1, ry2
            
        # Nos aseguramos de que x1<x2 y y1<y2 por si las cajas salen volteadas
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        
        detections.append(
            {
                "id_caja": idx,
                "clase": names[int(class_id)],
                "probabilidad": round(float(confidence), 4),
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1),
            }
        )

    return detections


def get_color(class_name: str) -> str:
    palette = [
        "#ef4444", "#f97316", "#f59e0b", "#eab308", "#84cc16", 
        "#22c55e", "#10b981", "#14b8a6", "#06b6d4", "#0ea5e9", 
        "#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#d946ef", 
        "#ec4899", "#f43f5e"
    ]
    idx = sum(ord(c) for c in class_name) % len(palette)
    return palette[idx]


def render_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        label = f"{d['clase']} {d['probabilidad']:.2f}"
        color = get_color(d["clase"])
        
        # Caja
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Etiqueta
        text_w = len(label) * 6 + 8
        text_h = 16
        draw.rectangle([x1, max(0, y1 - text_h), x1 + text_w, max(0, y1)], fill=color)
        draw.text((x1 + 3, max(0, y1 - text_h)), label, fill="white")
        
    return annotated


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
      <div class="pill">Sistema de Vision de Seguridad</div>
      <h1>PPE Detector</h1>
      <p>Sube una imagen o usa tu camara para detectar equipo de proteccion personal en tiempo real con tu modelo TFLite.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"No se encontro el archivo del modelo: `{MODEL_PATH}`")
    st.stop()

if YOLO is None:
    st.error("Ultralytics no esta instalado correctamente en este despliegue.")
    st.info(
        "Instala las dependencias de requirements.txt y vuelve a desplegar. "
        "Si sigue fallando, revisa los logs de compilacion para ver el error de instalacion de ultralytics."
    )
    if ULTRALYTICS_IMPORT_ERROR is not None:
        st.code(str(ULTRALYTICS_IMPORT_ERROR), language="text")
    st.stop()

with st.sidebar:
    st.header("Configuracion de inferencia")
    confidence_threshold = st.slider("Umbral de confianza", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("Umbral IoU", 0.05, 0.95, 0.45, 0.05)
    st.info(
        "Esta app ejecuta el modelo TFLite en CPU. "
        "El tamano de la imagen se escala automaticamente a 512px de forma interna para la inferencia."
    )
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: var(--muted); font-size: 0.85rem;">
            Desarrollado por <b>Sergio Cuadros</b> &copy; 2026
        </div>
        """,
        unsafe_allow_html=True,
    )

input_method = st.radio("Metodo de entrada", ["📸 Usar camara", "📁 Subir archivo"], horizontal=True)

if input_method == "📁 Subir archivo":
    uploaded_file = st.file_uploader(
        "Sube una imagen",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )
else:
    uploaded_file = st.camera_input("Toma una foto con tu camara")

if uploaded_file is None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="pill">Listo</div>
            <p style="margin:0.2rem 0 0.2rem; color: var(--muted);">
                Sube una imagen o toma una foto para ejecutar la deteccion. Modelo en uso:
                <span class="mono">{MODEL_PATH.name}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
model = load_model(MODEL_PATH)
results = None

with st.spinner("Ejecutando deteccion..."):
    try:
        results = model.predict(
            image,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=DEFAULT_IMGSZ,
            device="cpu",
            verbose=False,
        )
    except Exception as exc:
        st.error("La inferencia TFLite fallo. Revisa la compatibilidad del modelo y los logs del despliegue.")
        st.code(f"{type(exc).__name__}: {exc}", language="text")
        st.stop()

if results is None or len(results) == 0:
    st.error("El modelo no devolvio resultados de inferencia.")
    st.stop()

assert results is not None
result = results[0]
detections = extract_detections(result, image.width, image.height)
annotated_image = render_boxes(image, detections)

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Imagen original")
    st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Detecciones")
    st.image(annotated_image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

metric_cols = st.columns(3)
metric_cols[0].metric("Detecciones", len(result.boxes) if result.boxes is not None else 0)
metric_cols[1].metric("Clases detectadas", len({row["clase"] for row in detections}))
metric_cols[2].metric("Modelo", MODEL_PATH.name)

st.markdown('<div class="panel" style="margin-top: 0.85rem;">', unsafe_allow_html=True)
st.subheader("Todas las cajas detectadas y sus probabilidades")
if detections:
    st.dataframe(
        detections,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id_caja": st.column_config.NumberColumn("ID Caja", format="%d"),
            "clase": st.column_config.TextColumn("Clase de PPE"),
            "probabilidad": st.column_config.ProgressColumn(
                "Probabilidad",
                help="Nivel de confianza del modelo",
                format="%f",
                min_value=0.0,
                max_value=1.0,
            ),
            "x1": st.column_config.NumberColumn("X1", format="%d px"),
            "y1": st.column_config.NumberColumn("Y1", format="%d px"),
            "x2": st.column_config.NumberColumn("X2", format="%d px"),
            "y2": st.column_config.NumberColumn("Y2", format="%d px"),
        },
    )
else:
    st.write("No se encontraron detecciones con los umbrales actuales.")
st.markdown("</div>", unsafe_allow_html=True)
