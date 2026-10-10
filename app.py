import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "best_model.keras"))
CLASS_PATH = Path(os.environ.get("CLASS_PATH", BASE_DIR / "class_names.json"))
MODEL_URL = (
    "https://huggingface.co/gdgkahgd/leaflens-plant-disease-model/"
    "resolve/main/best_model.keras"
)

st.set_page_config(
    page_title="LeafLens | Plant Health AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --green:#216b4b; --deep:#123c2b; --mint:#e9f5ed; --ink:#17251e; --muted:#68766e; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f5faf6 0%,#ffffff 42%); color:var(--ink); }
.block-container { max-width:1180px; padding-top:2rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#f4f7f3 !important; border-right:1px solid #dce9df; }
/* Sidebar: force readable contrast across Streamlit's nested elements/themes. */
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] div, [data-testid="stSidebar"] li,
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stCaption {
  color:#20352a !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong {
  color:#123c2b !important;
}
[data-testid="stSidebar"] code {
  color:#ffffff !important; background:#123c2b !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
  background:#ffffff !important; border:1px solid #dce9df !important;
  border-radius:12px !important;
}
[data-testid="stSidebar"] [data-testid="stToggle"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
  color:#20352a !important;
}
[data-testid="stHeader"] { background:rgba(255,255,255,.94) !important; }
[data-testid="stToolbar"] { right:1rem; }
.hero { padding:2.2rem 2.3rem; border-radius:26px; color:white; background:linear-gradient(125deg,#123c2b 0%,#216b4b 65%,#39865e 100%); box-shadow:0 16px 45px rgba(20,75,48,.14); margin-bottom:1.3rem; }
.eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.74rem; font-weight:700; color:#c3e6ce; }
.hero h1 { font-family:'Manrope',sans-serif; color:white; font-size:clamp(2.1rem,4vw,3.3rem); line-height:1.08; letter-spacing:-.04em; margin:.55rem 0 .85rem; }
.hero p { color:#e1f0e5; max-width:700px; font-size:1.04rem; line-height:1.7; margin:0; }
.pill { display:inline-block; border:1px solid rgba(255,255,255,.28); border-radius:99px; padding:.35rem .7rem; margin-top:1.2rem; font-size:.78rem; color:#f0faf2; }
.section-title { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.2rem; margin:.5rem 0 .25rem; }
.subtle { color:#68766e; font-size:.93rem; line-height:1.6; }
.panel { background:white; border:1px solid #e1ebe3; border-radius:20px; padding:1.25rem 1.35rem; box-shadow:0 6px 24px rgba(24,60,39,.045); }
.result-label { color:#607268; font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; }
.result-name { font-family:'Manrope',sans-serif; color:#174a32; font-size:1.45rem; font-weight:800; line-height:1.25; margin:.45rem 0; }
.confidence { font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; color:#216b4b; }
.small-note { color:#6d7b72; font-size:.82rem; line-height:1.5; }
footer {visibility:hidden;}
[data-testid="stMetric"] { background:white; border:1px solid #e1ebe3; border-radius:16px; padding:1rem; }
.stButton>button { border-radius:12px; font-weight:700; }
body, [data-testid="stAppViewContainer"] { color:#17251e !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] p { color:#526258; }
[data-testid="stFileUploader"] section { background:#ffffff !important; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color:#263b2f !important; }
@media (max-width: 760px) {
  .block-container { padding:1rem 1rem 2rem; }
  .hero { padding:1.5rem 1.25rem; border-radius:20px; }
  .hero h1 { font-size:2.1rem; }
  .hero p { font-size:.98rem; }
}
[data-testid="stFileUploader"] { background:white; border:1px dashed #a9c9b2; border-radius:16px; padding:.65rem; }
hr { border-color:#e4eee6; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Downloading the trained model…")
def ensure_model():
    """Download the model from Hugging Face if it is not already available."""
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0:
        return str(MODEL_PATH)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MODEL_PATH.with_name(MODEL_PATH.name + ".download")

    try:
        with requests.get(MODEL_URL, stream=True, timeout=(20, 300)) as response:
            response.raise_for_status()
            with temp_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)

        if not temp_path.is_file() or temp_path.stat().st_size == 0:
            raise ValueError("The downloaded model file is empty.")
        temp_path.replace(MODEL_PATH)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return str(MODEL_PATH)


@st.cache_resource(show_spinner="Loading the trained model…")
def load_assets():
    if not CLASS_PATH.is_file():
        raise FileNotFoundError(
            "class_names.json is missing from the GitHub repository."
        )
    model = keras.models.load_model(ensure_model(), compile=False)
    with CLASS_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

def gradcam_overlay(model, image_array, class_index):
    """Best-effort Grad-CAM for the nested CNN backbone used by this model."""
    submodels = [layer for layer in model.layers if isinstance(layer, keras.Model)]
    if not submodels:
        raise ValueError("No nested feature-extractor model was found.")
    backbone = max(submodels, key=lambda layer: len(layer.layers))
    backbone_index = model.layers.index(backbone)
    pre_layers = [layer for layer in model.layers[:backbone_index]
                  if not isinstance(layer, keras.layers.InputLayer)]
    post_layers = list(model.layers[backbone_index + 1:])
    x = tf.convert_to_tensor(image_array[None].astype("float32"))
    for layer in pre_layers:
        x = layer(x)
    with tf.GradientTape() as tape:
        features = backbone(x, training=False)
        tape.watch(features)
        z = features
        for layer in post_layers[:-1]:
            z = layer(z)
        final_layer = post_layers[-1]
        if not hasattr(final_layer, "kernel"):
            raise ValueError("The final classification layer is not a Dense layer.")
        logits = tf.matmul(z, final_layer.kernel) + final_layer.bias
        score = logits[:, class_index]
    gradients = tape.gradient(score, features)
    if gradients is None:
        raise ValueError("Could not calculate gradients for this model.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))[0]
    cam = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
    cam = np.clip(Image.fromarray(np.uint8(cam * 255)).resize((224, 224)), 0, 255).astype(np.float32) / 255
    import cv2
    heat = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip(0.58 * image_array + 0.42 * heat, 0, 255))

if not CLASS_PATH.is_file():
    st.error("class_names.json is missing. Upload it to GitHub beside app.py.")
    st.stop()

try:
    model, metadata = load_assets()
except Exception as exc:
    st.error("The model could not be loaded. Check the deployed dependency versions and model files.")
    st.exception(exc)
    st.stop()

class_names = metadata["class_names"]

with st.sidebar:
    st.markdown("## 🌿 LeafLens")
    st.caption("Plant health screening powered by computer vision")
    st.divider()
    st.markdown("**Model details**")
    st.write(f"Architecture: `{metadata.get('selected_model', 'Trained model')}`")
    st.write(f"Recognised classes: **{metadata.get('num_classes', len(class_names))}**")
    st.write("Input image: **224 × 224 px**")
    st.divider()
    show_gradcam = st.toggle("Show visual explanation (Grad-CAM)", value=True)
    st.markdown("**Supported categories**")
    with st.expander("Browse all 38 classes"):
        for class_name in class_names:
            st.write(f"• {class_name}")
    st.caption("Research prototype · Not a medical or agronomic diagnosis")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-powered crop screening</div>
  <h1>A clearer picture of<br>plant health.</h1>
  <p>Upload a leaf photo to get a model prediction, confidence score, and optional visual explanation. Designed as a research prototype for early plant-disease screening.</p>
  <div class="pill">EfficientNetB0&nbsp; · &nbsp;38 classes&nbsp; · &nbsp;Image classification</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, .95], gap="large")
with left:
    st.markdown('<div class="section-title">Analyse a leaf</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">For best results, use a sharp, well-lit photo showing one leaf clearly. The model may be less reliable on field images or unfamiliar crops.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded:
        try:
            original_image = Image.open(uploaded).convert("RGB")
            st.image(original_image, caption="Image selected for analysis", use_container_width=True)
        except Exception:
            st.error("This file could not be opened as an image.")
            original_image = None
    else:
        original_image = None
        st.markdown('<div class="panel"><div class="section-title">Before you upload</div><p class="subtle">• Photograph one leaf in focus<br>• Avoid heavy shadows and motion blur<br>• Keep the leaf large in the frame<br>• Use the result as a screening aid, not a final diagnosis</p></div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Analysis results</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Your prediction will appear here after you select an image.</div>', unsafe_allow_html=True)
    if original_image is not None:
        with st.spinner("Analysing image…"):
            resized = original_image.resize((224, 224), Image.Resampling.BILINEAR)
            image_array = np.asarray(resized).astype("float32")
            try:
                probabilities = np.asarray(model.predict(image_array[None, ...], verbose=0)[0]).reshape(-1)
            except Exception as exc:
                st.error("Prediction failed. Check that the model and input configuration match.")
                st.exception(exc)
                st.stop()
        if len(probabilities) != len(class_names):
            st.error(f"Model returned {len(probabilities)} outputs but metadata contains {len(class_names)} class names.")
            st.stop()
        top_indices = np.argsort(probabilities)[::-1][:3]
        best_index = int(top_indices[0])
        confidence = float(probabilities[best_index])
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">Top prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-name">{class_names[best_index]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence">{confidence * 100:.1f}%</div><div class="small-note">Model confidence score — not a guarantee of correctness.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if confidence < 0.50:
            st.warning("Low confidence. The image may be ambiguous or outside the model's training distribution. Do not rely on this prediction alone.")
        st.markdown("#### Top 3 model predictions")
        chart_data = pd.DataFrame({"Class": [class_names[int(i)] for i in top_indices], "Score": [float(probabilities[int(i)]) for i in top_indices]})
        st.bar_chart(chart_data.set_index("Class"), horizontal=True)
        if show_gradcam:
            st.markdown("#### Visual explanation")
            try:
                overlay = gradcam_overlay(model, image_array, best_index)
                st.image(overlay, caption="Grad-CAM overlay: highlighted regions influenced the prediction. This is an explanation aid, not proof that the model is correct.", use_container_width=True)
            except Exception as exc:
                st.info(f"Grad-CAM is unavailable for this model configuration. Prediction is still available. ({str(exc)[:140]})")
        st.download_button("Download prediction summary (CSV)", data=chart_data.to_csv(index=False).encode("utf-8"), file_name="plant_disease_prediction.csv", mime="text/csv", use_container_width=True)
    else:
        st.markdown('<div class="panel"><div class="result-label">Waiting for image</div><div class="result-name">Your results will appear here</div><p class="subtle">The app returns the three highest-scoring classes once an image is uploaded.</p></div>', unsafe_allow_html=True)

st.divider()
metric = metadata.get("metrics", {}).get("test", {})
if metric:
    st.markdown("### Model evaluation snapshot")
    st.markdown('<p class="subtle">Reported metrics from the saved experiment metadata on the in-domain PlantVillage test split. These are not a promise of accuracy on user-uploaded field photos.</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy", f"{metric.get('accuracy', 0) * 100:.2f}%")
    m2.metric("Test macro precision", f"{metric.get('precision', 0) * 100:.2f}%")
    m3.metric("Test macro F1", f"{metric.get('f1', 0) * 100:.2f}%")

st.markdown('<p class="small-note">LeafLens is an educational research prototype. The classifier recognises only the listed crop/disease classes and may perform less reliably on real-world field photographs. Confirm important decisions with a qualified agricultural specialist.</p>', unsafe_allow_html=True)
