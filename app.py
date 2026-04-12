import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChiliScan — Deteksi Penyakit Daun Cabai",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Root & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 2rem;
}

.app-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    letter-spacing: -0.03em;
    margin: 0;
    background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 50%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
}

@keyframes shimmer {
    to { background-position: 200% center; }
}

.app-header p {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 1.05rem;
    color: #8b949e;
    margin-top: 0.5rem;
    letter-spacing: 0.02em;
}

.badge {
    display: inline-block;
    background: rgba(255, 107, 53, 0.12);
    border: 1px solid rgba(255, 107, 53, 0.3);
    color: #ff6b35;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}

/* ── Upload & Image Panel ── */
.upload-panel {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 1.5rem;
}

.upload-panel h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #c9d1d9;
    margin-bottom: 1rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Model Card ── */
.model-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
    transition: border-color 0.3s ease;
}

.model-card:hover {
    border-color: rgba(255, 107, 53, 0.4);
}

.model-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #ff6b35;
    margin-bottom: 0.25rem;
}

.disease-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: #f0f6fc;
    text-transform: capitalize;
    margin: 0.4rem 0;
    line-height: 1.2;
}

.confidence-pct {
    font-size: 2.4rem;
    font-weight: 300;
    font-family: 'DM Sans', sans-serif;
    color: #ff6b35;
    letter-spacing: -0.04em;
    margin: 0.5rem 0 1rem;
}

.confidence-pct span {
    font-size: 1.1rem;
    color: #8b949e;
    letter-spacing: 0;
}

/* ── Progress bar override ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #ff6b35, #f7c59f) !important;
    border-radius: 999px !important;
}

.stProgress > div > div > div {
    background: #21262d !important;
    border-radius: 999px !important;
    height: 8px !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #ff6b35, #e85d2a);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 2rem;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 24px rgba(255, 107, 53, 0.25);
    margin-top: 1rem;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(255, 107, 53, 0.4);
}

.stButton > button:active {
    transform: translateY(0);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0d1117;
    border: 2px dashed #30363d;
    border-radius: 12px;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(255, 107, 53, 0.5);
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 2rem 0;
}

/* ── Result section title ── */
.result-section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 1.2rem;
}

/* ── Healthy chip ── */
.tag-healthy {
    display: inline-block;
    background: rgba(46, 160, 67, 0.15);
    border: 1px solid rgba(46, 160, 67, 0.4);
    color: #3fb950;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
}

.tag-disease {
    display: inline-block;
    background: rgba(248, 81, 73, 0.12);
    border: 1px solid rgba(248, 81, 73, 0.35);
    color: #f85149;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    color: #484f58;
    font-size: 0.8rem;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD CLASS MAPPING
# ─────────────────────────────────────────────
@st.cache_data
def load_class_mapping(path: str = "chili_class_mapping.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
#  LOAD MODELS (cached — hanya sekali saat startup)
# ─────────────────────────────────────────────
@st.cache_resource
def load_all_models(num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── DenseNet-121 ──
    densenet = models.densenet121(weights=None)
    densenet.classifier = torch.nn.Linear(
        densenet.classifier.in_features, num_classes
    )
    densenet.load_state_dict(
        torch.load("DenseNet_121_final.pth", map_location=device)
    )
    densenet.to(device).eval()

    # ── ResNet-50 ──
    resnet = models.resnet50(weights=None)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
    resnet.load_state_dict(
        torch.load("ResNet_50_final.pth", map_location=device)
    )
    resnet.to(device).eval()

    # ── EfficientNet-B0 ──
    efficientnet = models.efficientnet_b0(weights=None)
    efficientnet.classifier[1] = torch.nn.Linear(
        efficientnet.classifier[1].in_features, num_classes
    )
    efficientnet.load_state_dict(
        torch.load("EfficientNet_B0_final.pth", map_location=device)
    )
    efficientnet.to(device).eval()

    return {
        "DenseNet-121": densenet,
        "ResNet-50": resnet,
        "EfficientNet-B0": efficientnet,
    }, device


# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Konversi PIL Image → tensor siap inferensi."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    img_rgb = image.convert("RGB")
    tensor = transform(img_rgb)
    return tensor.unsqueeze(0)  # tambah dimensi batch → [1, C, H, W]


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
def run_inference(model, tensor: torch.Tensor, device) -> tuple[str, float, np.ndarray]:
    """Jalankan inferensi dan kembalikan (class_idx, confidence, all_probs)."""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    return class_idx, confidence, probs


# ─────────────────────────────────────────────
#  HELPER: render satu kartu model
# ─────────────────────────────────────────────
def render_model_card(col, model_label: str, class_name: str, confidence: float):
    is_healthy = "healthy" in class_name.lower()
    status_tag = (
        '<span class="tag-healthy">✔ Sehat</span>'
        if is_healthy
        else '<span class="tag-disease">⚠ Penyakit</span>'
    )
    with col:
        st.markdown(f"""
        <div class="model-card">
            <div class="model-name">{model_label}</div>
            {status_tag}
            <div class="disease-label">{class_name.replace("_", " ").title()}</div>
            <div class="confidence-pct">{confidence*100:.1f}<span>%</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(confidence)


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown("""
    <div class="app-header">
        <div class="badge">🌶️ AI-Powered Plant Pathology</div>
        <h1>ChiliScan</h1>
        <p>Identifikasi penyakit daun cabai secara otomatis menggunakan tiga model deep learning.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load class mapping ──
    try:
        class_mapping = load_class_mapping()
    except FileNotFoundError:
        st.error("❌ File `chili_class_mapping.json` tidak ditemukan. Pastikan file ada di direktori yang sama dengan `app.py`.")
        st.stop()

    num_classes = len(class_mapping)

    # ── Load models ──
    try:
        models_dict, device = load_all_models(num_classes)
    except FileNotFoundError as e:
        st.error(f"❌ File model tidak ditemukan: {e}. Pastikan ketiga file `.pth` ada di direktori yang sama.")
        st.stop()

    # ── Layout: left panel (upload) | right panel (results) ──
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
        st.markdown('<h3>📁 Upload Gambar Daun</h3>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="Pilih gambar daun cabai",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Gambar yang diupload")

        analyze_btn = st.button("🔬 Analisis Trio Model", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Results panel ──
    with right_col:
        if not uploaded_file:
            st.markdown("""
            <div style="
                height: 380px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                color: #484f58;
                border: 1px dashed #21262d;
                border-radius: 16px;
                text-align: center;
                padding: 2rem;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🍃</div>
                <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; color: #8b949e; margin-bottom: 0.5rem;">
                    Belum ada gambar yang diupload
                </div>
                <div style="font-size: 0.85rem; color: #484f58; max-width: 300px;">
                    Upload gambar daun cabai di panel kiri, lalu tekan tombol Analisis.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif not analyze_btn:
            st.markdown("""
            <div style="
                height: 380px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                color: #484f58;
                border: 1px dashed #21262d;
                border-radius: 16px;
                text-align: center;
                padding: 2rem;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
                <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; color: #8b949e; margin-bottom: 0.5rem;">
                    Gambar siap dianalisis
                </div>
                <div style="font-size: 0.85rem; color: #484f58; max-width: 300px;">
                    Tekan tombol <strong style="color: #ff6b35;">Analisis Trio Model</strong> untuk menjalankan inferensi ketiga model.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Run inference ──
            image = Image.open(uploaded_file)
            tensor = preprocess_image(image)

            results = {}
            with st.spinner("Menjalankan inferensi pada tiga model..."):
                for model_name, model in models_dict.items():
                    idx, conf, probs = run_inference(model, tensor, device)
                    class_name = class_mapping[str(idx)]
                    results[model_name] = {
                        "class_name": class_name,
                        "confidence": conf,
                        "probs": probs,
                    }

            # ── Render result cards ──
            st.markdown('<div class="result-section-title">Hasil Prediksi — Trio Model</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3, gap="medium")
            model_names = list(results.keys())

            render_model_card(c1, model_names[0], results[model_names[0]]["class_name"], results[model_names[0]]["confidence"])
            render_model_card(c2, model_names[1], results[model_names[1]]["class_name"], results[model_names[1]]["confidence"])
            render_model_card(c3, model_names[2], results[model_names[2]]["class_name"], results[model_names[2]]["confidence"])

            # ── Voting summary ──
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            votes = [r["class_name"] for r in results.values()]
            majority = max(set(votes), key=votes.count)
            vote_count = votes.count(majority)

            st.markdown(f"""
            <div style="
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 16px;
                padding: 1.2rem 1.5rem;
                display: flex;
                align-items: center;
                gap: 1rem;
            ">
                <div style="font-size: 2rem;">🗳️</div>
                <div>
                    <div style="font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #8b949e; margin-bottom: 0.25rem;">
                        Konsensus Model ({vote_count}/3 suara)
                    </div>
                    <div style="font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 800; color: #f0f6fc; text-transform: capitalize;">
                        {majority.replace("_", " ").title()}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div class="footer">
        ChiliScan · DenseNet-121 · ResNet-50 · EfficientNet-B0 · Powered by PyTorch & Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()