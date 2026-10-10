import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import html
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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Figtree:wght@400;500;600;700&display=swap');

:root {
  --paper:#f2f5f1;
  --surface:#ffffff;
  --ink:#0f2a1d;
  --ink-2:#3d5446;
  --muted:#6b7d72;
  --line:#dbe4dc;
  --forest:#12281d;
  --leaf:#2c7a4f;
  --leaf-soft:#e3f0e7;
  --amber:#b7791f;
  --amber-soft:#fbf0d9;
  --radius:14px;
}

html, body, [class*="css"] { font-family:'Figtree',system-ui,sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:1200px; padding-top:1.6rem; padding-bottom:3rem; }
footer { visibility:hidden; }
hr { border-color:var(--line) !important; }
[data-testid="stHeader"] { background:transparent !important; }
[data-testid="stAppViewContainer"], body { color:var(--ink); }
[data-testid="stMarkdownContainer"] p { color:var(--ink-2); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background:var(--forest) !important; border-right:0; }
[data-testid="stSidebar"] * { color:#d6e5db !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color:#ffffff !important; font-weight:600; }
[data-testid="stSidebar"] h2 { font-family:'Fraunces',serif; font-size:1.7rem; letter-spacing:-.01em; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.12) !important; }
[data-testid="stSidebar"] code {
  background:rgba(255,255,255,.12) !important; color:#ffffff !important;
  border-radius:6px; padding:.1rem .4rem;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#9fb8a8 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] {
  background:rgba(255,255,255,.05) !important;
  border:1px solid rgba(255,255,255,.12) !important;
  border-radius:var(--radius) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background:rgba(255,255,255,.06); }
[data-testid="stSidebar"] [data-testid="stExpander"] p { font-size:.88rem; margin-bottom:.15rem; }

/* ---------- Masthead ---------- */
.masthead {
  position:relative; overflow:hidden;
  background:var(--forest); color:#fff;
  border-radius:22px; padding:2.6rem 2.8rem 2.4rem; margin-bottom:1.8rem;
}
.masthead::before {
  content:""; position:absolute; inset:0;
  background:
    radial-gradient(1200px 400px at 105% -20%, rgba(80,170,115,.35), transparent 60%),
    repeating-linear-gradient(115deg, rgba(255,255,255,.035) 0 1px, transparent 1px 34px);
}
.masthead-inner { position:relative; max-width:720px; }
.brand { display:flex; align-items:center; gap:.6rem; font-weight:600; color:#bfe0cb; font-size:.95rem; }
.brand-mark {
  width:30px; height:30px; border-radius:9px; background:#2c7a4f;
  display:grid; place-items:center; font-size:1rem;
}
.masthead h1 {
  font-family:'Fraunces',serif; font-weight:600; color:#fff;
  font-size:clamp(2.2rem,4.4vw,3.5rem); line-height:1.06; letter-spacing:-.025em;
  margin:1.3rem 0 1rem; padding:0;
}
.masthead p { color:#cfe3d6 !important; font-size:1.05rem; line-height:1.65; margin:0; max-width:620px; }
.facts { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.6rem; }
.fact {
  border:1px solid rgba(255,255,255,.2); background:rgba(255,255,255,.06);
  color:#e6f2ea; border-radius:999px; padding:.32rem .8rem; font-size:.82rem; font-weight:500;
}

/* ---------- Typography helpers ---------- */
.section-title { font-family:'Fraunces',serif; font-weight:600; font-size:1.55rem; color:var(--ink); letter-spacing:-.01em; margin:.2rem 0 .35rem; }
.subtle { color:var(--muted); font-size:.95rem; line-height:1.6; margin-bottom:1rem; }
.small-note { color:var(--muted); font-size:.82rem; line-height:1.55; }
.sub-heading { font-family:'Fraunces',serif; font-weight:600; font-size:1.2rem; color:var(--ink); margin:1.6rem 0 .6rem; }

/* ---------- Panels ---------- */
.panel {
  background:var(--surface); border:1px solid var(--line);
  border-radius:var(--radius); padding:1.4rem 1.5rem;
}
.panel .section-title { font-size:1.25rem; }
.checklist { list-style:none; padding:0; margin:.6rem 0 0; }
.checklist li { position:relative; padding:.45rem 0 .45rem 1.7rem; color:var(--ink-2); font-size:.95rem; border-top:1px solid var(--line); }
.checklist li:first-child { border-top:0; }
.checklist li::before {
  content:""; position:absolute; left:.1rem; top:.85rem; width:.55rem; height:.55rem;
  border-radius:50% 0 50% 0; background:var(--leaf); transform:rotate(-20deg);
}
.empty {
  border:1.5px dashed #b9cbbd; border-radius:var(--radius); padding:2.2rem 1.6rem;
  text-align:center; background:rgba(255,255,255,.55);
}
.empty .glyph { font-size:2rem; margin-bottom:.4rem; }
.empty .title { font-family:'Fraunces',serif; font-size:1.3rem; font-weight:600; color:var(--ink); }
.empty p { color:var(--muted) !important; font-size:.93rem; margin:.4rem auto 0; max-width:340px; line-height:1.55; }

/* ---------- Result card ---------- */
.result {
  background:var(--surface); border:1px solid var(--line); border-radius:var(--radius);
  padding:1.5rem 1.6rem; display:flex; align-items:center; justify-content:space-between; gap:1.4rem;
  border-left:5px solid var(--leaf);
}
.result.low { border-left-color:var(--amber); }
.result-label { color:var(--muted); font-size:.85rem; font-weight:600; }
.result-crop { color:var(--leaf); font-weight:600; font-size:.98rem; margin-top:.5rem; }
.result.low .result-crop { color:var(--amber); }
.result-name { font-family:'Fraunces',serif; color:var(--ink); font-size:1.85rem; font-weight:600; line-height:1.15; letter-spacing:-.015em; margin:.15rem 0 .5rem; }
.ring {
  --p:0; --c:var(--leaf);
  flex:0 0 auto; width:104px; height:104px; border-radius:50%;
  background:conic-gradient(var(--c) calc(var(--p) * 1%), #e4ebe5 0);
  display:grid; place-items:center; position:relative;
}
.ring::before { content:""; position:absolute; inset:9px; background:var(--surface); border-radius:50%; }
.ring span { position:relative; font-family:'Fraunces',serif; font-weight:600; font-size:1.3rem; color:var(--ink); }
.result.low .ring { --c:var(--amber); }
.confidence-note { color:var(--muted); font-size:.8rem; line-height:1.45; max-width:230px; }

/* ---------- Top-3 bars ---------- */
.bars { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:.4rem 1.3rem; }
.bar-row { padding:.85rem 0; border-top:1px solid var(--line); }
.bar-row:first-child { border-top:0; }
.bar-top { display:flex; justify-content:space-between; gap:1rem; font-size:.93rem; margin-bottom:.45rem; }
.bar-name { color:var(--ink); font-weight:500; }
.bar-name small { color:var(--muted); font-weight:400; margin-left:.35rem; }
.bar-val { color:var(--ink-2); font-variant-numeric:tabular-nums; font-weight:600; }
.track { height:8px; border-radius:99px; background:#e8eee9; overflow:hidden; }
.fill { height:100%; border-radius:99px; background:#9bc4aa; }
.bar-row:first-child .fill { background:var(--leaf); }

/* ---------- Streamlit widgets ---------- */
[data-testid="stFileUploader"] { background:transparent; padding:0; margin-bottom:1rem; }
[data-testid="stFileUploader"] section {
  background:var(--surface) !important; border:1.5px dashed #a8c3ae !important;
  border-radius:var(--radius) !important; padding:1.3rem !important;
}
[data-testid="stFileUploader"] section:hover { border-color:var(--leaf) !important; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color:var(--ink-2) !important; }
[data-testid="stFileUploader"] button { border-radius:10px; }
.stButton>button, .stDownloadButton>button {
  border-radius:11px; font-weight:600; border:1px solid var(--ink);
  background:var(--ink); color:#fff; padding:.6rem 1rem; transition:background .15s;
}
.stButton>button:hover, .stDownloadButton>button:hover { background:var(--leaf); border-color:var(--leaf); color:#fff; }
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible { outline:3px solid #8fd0a9; outline-offset:2px; }
[data-testid="stImage"] img { border-radius:var(--radius); border:1px solid var(--line); }
[data-testid="stImage"] [data-testid="stImageCaption"], [data-testid="stCaptionContainer"] { color:var(--muted); }
[data-testid="stAlert"] { border-radius:var(--radius); }
[data-testid="stMetric"] {
  background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:1.1rem 1.3rem;
}
[data-testid="stMetricLabel"] p { color:var(--muted) !important; font-weight:500; }
[data-testid="stMetricValue"] { font-family:'Fraunces',serif; font-weight:600; color:var(--ink); }

@media (max-width: 760px) {
  .block-container { padding:1rem 1rem 2rem; }
  .masthead { padding:1.7rem 1.4rem; border-radius:18px; }
  .result { flex-direction:column-reverse; align-items:flex-start; }
}
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
</style>
""", unsafe_allow_html=True)


def split_label(raw):
    """Display-only: turn 'Tomato___Late_blight' into ('Tomato', 'Late blight')."""
    if "___" in raw:
        crop, condition = raw.split("___", 1)
    else:
        crop, condition = "", raw
    clean = lambda s: html.escape(s.replace("_", " ").replace("  ", " ").strip())
    return clean(crop), clean(condition)


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
<div class="masthead"><div class="masthead-inner">
<div class="brand"><span class="brand-mark">🌿</span>LeafLens</div>
<h1>A clearer picture of plant health.</h1>
<p>Upload a leaf photo to get a model prediction, confidence score, and optional visual explanation. Designed as a research prototype for early plant-disease screening.</p>
<div class="facts"><span class="fact">EfficientNetB0</span><span class="fact">38 classes</span><span class="fact">Image classification</span></div>
</div></div>
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
        st.markdown('<div class="panel"><div class="section-title">Before you upload</div><ul class="checklist"><li>Photograph one leaf in focus</li><li>Avoid heavy shadows and motion blur</li><li>Keep the leaf large in the frame</li><li>Use the result as a screening aid, not a final diagnosis</li></ul></div>', unsafe_allow_html=True)

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
        crop, condition = split_label(class_names[best_index])
        low_class = " low" if confidence < 0.50 else ""
        crop_html = f'<div class="result-crop">{crop}</div>' if crop else ""
        st.markdown(
            f'<div class="result{low_class}"><div>'
            f'<div class="result-label">Top prediction</div>{crop_html}'
            f'<div class="result-name">{condition}</div>'
            f'<div class="confidence-note">Model confidence score — not a guarantee of correctness.</div>'
            f'</div><div class="ring" style="--p:{confidence * 100:.1f}"><span>{confidence * 100:.1f}%</span></div></div>',
            unsafe_allow_html=True,
        )
        if confidence < 0.50:
            st.warning("Low confidence. The image may be ambiguous or outside the model's training distribution. Do not rely on this prediction alone.")
        st.markdown('<div class="sub-heading">Top 3 model predictions</div>', unsafe_allow_html=True)
        chart_data = pd.DataFrame({"Class": [class_names[int(i)] for i in top_indices], "Score": [float(probabilities[int(i)]) for i in top_indices]})
        rows = ""
        for i in top_indices:
            c, d = split_label(class_names[int(i)])
            score = float(probabilities[int(i)])
            tag = f"<small>{c}</small>" if c else ""
            rows += (
                f'<div class="bar-row"><div class="bar-top"><span class="bar-name">{d}{tag}</span>'
                f'<span class="bar-val">{score * 100:.1f}%</span></div>'
                f'<div class="track"><div class="fill" style="width:{max(score * 100, 0.8):.1f}%"></div></div></div>'
            )
        st.markdown(f'<div class="bars">{rows}</div>', unsafe_allow_html=True)
        if show_gradcam:
            st.markdown('<div class="sub-heading">Visual explanation</div>', unsafe_allow_html=True)
            try:
                overlay = gradcam_overlay(model, image_array, best_index)
                st.image(overlay, caption="Grad-CAM overlay: highlighted regions influenced the prediction. This is an explanation aid, not proof that the model is correct.", use_container_width=True)
            except Exception as exc:
                st.info(f"Grad-CAM is unavailable for this model configuration. Prediction is still available. ({str(exc)[:140]})")
        st.download_button("Download prediction summary (CSV)", data=chart_data.to_csv(index=False).encode("utf-8"), file_name="plant_disease_prediction.csv", mime="text/csv", use_container_width=True)
    else:
        st.markdown('<div class="empty"><div class="glyph">🍃</div><div class="title">Your results will appear here</div><p>The app returns the three highest-scoring classes once an image is uploaded.</p></div>', unsafe_allow_html=True)

st.divider()
metric = metadata.get("metrics", {}).get("test", {})
if metric:
    st.markdown('<div class="section-title">Model evaluation snapshot</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Reported metrics from the saved experiment metadata on the in-domain PlantVillage test split. These are not a promise of accuracy on user-uploaded field photos.</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Test accuracy", f"{metric.get('accuracy', 0) * 100:.2f}%")
    m2.metric("Test macro precision", f"{metric.get('precision', 0) * 100:.2f}%")
    m3.metric("Test macro F1", f"{metric.get('f1', 0) * 100:.2f}%")

st.markdown('<p class="small-note" style="margin-top:1.4rem">LeafLens is an educational research prototype. The classifier recognises only the listed crop/disease classes and may perform less reliably on real-world field photographs. Confirm important decisions with a qualified agricultural specialist.</p>', unsafe_allow_html=True)
