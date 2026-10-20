"""
CSFNet-2: Real-Time Multispectral (RGBT) Semantic Segmentation Web App
Powered exclusively by the best model: CSFNet-2 (STDC2 Backbone, ImageNet Pretrained, 19.36M Params).
"""

import os
import io
import time
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import streamlit as st

from configs.default_config import get_default_config
from models.csfnet import build_csfnet
from utils.visualization import colorize_mask, CLASS_COLORS, CLASS_NAMES

# ------------------------------------------------------------------------------
# Page Configuration & Aesthetics
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CSFNet-2 RGBT Segmentation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #50E3C2, #4A90E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #8E9AAF;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background: #181920;
        border: 1px solid #2B2D3A;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #50E3C2;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #8E9AAF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .class-badge {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Model Initialization (Cached)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CSFNet-2 weights...")
def get_model(device_str: str):
    """Instantiate and load CSFNet-2 best checkpoint."""
    device = torch.device(device_str)
    model = build_csfnet(
        backbone="STDCNet1446",
        pretrained=False,
        num_classes=9,
        dataset="MFNet"
    ).to(device)

    # Checkpoint resolution priority
    candidates = [
        "runs/csfnet2_pretrained/best_model.pth",
        "results/csfnet2_best_model.pth"
    ]
    ckpt_path = next((p for p in candidates if os.path.exists(p)), None)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        status = f"Loaded `{ckpt_path}`"
    else:
        status = "No checkpoint found. Running with initialized weights."

    model.eval()
    return model, status


def apply_thermal_colormap(gray_arr: np.ndarray, colormap_name: str = "inferno") -> np.ndarray:
    """Transform 1-channel thermal radiation to RGB pseudo-color."""
    norm = np.clip(gray_arr.astype(np.float32) / 255.0, 0.0, 1.0)
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps[colormap_name]
    except Exception:
        try:
            cmap = cm.get_cmap(colormap_name)
        except Exception:
            return np.stack([gray_arr] * 3, axis=-1)
    colored = cmap(norm)[:, :, :3]
    return (colored * 255).astype(np.uint8)


# ------------------------------------------------------------------------------
# Sidebar Settings
# ------------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Model Settings")
st.sidebar.markdown("**Active Architecture:**")
st.sidebar.info("🚀 **CSFNet-2** (STDC2 Backbone)\n* 19.36M Parameters\n* 49.39% Test mIoU\n* ~72 FPS Real-Time")

device_choice = "cuda" if torch.cuda.is_available() else "cpu"
device = st.sidebar.selectbox("Inference Device:", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"], index=0)
if device == "cuda":
    st.sidebar.caption(f"⚡ GPU: {torch.cuda.get_device_name(0)}")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎨 Visual Controls")
overlay_alpha = st.sidebar.slider("Mask Overlay Opacity:", min_value=0.1, max_value=0.9, value=0.55, step=0.05)
thermal_palette = st.sidebar.selectbox("Thermal Palette:", ["inferno", "turbo", "plasma", "magma", "viridis", "gray"], index=0)

# Load configuration and model
cfg = get_default_config()
model, load_msg = get_model(device)
st.sidebar.caption(f"Status: {load_msg}")

# ------------------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">Multimodal RGBT Semantic Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time scene parsing using paired Visible RGB + Thermal Infrared imaging with CSFNet-2.</div>', unsafe_allow_html=True)

# Tabs: App vs About
tab_app, tab_benchmark = st.tabs(["🚀 Run Segmentation", "📊 Model Performance"])

with tab_app:
    # Upload Input Zone
    st.markdown("### 1. Upload Multispectral Input")
    
    up_mode = st.radio(
        "Upload Format:",
        ["Single 4-Channel RGBT Image (.png)", "Two Separate Files (RGB Color + Thermal Grayscale)"],
        horizontal=True
    )

    rgb_array, thermal_array = None, None
    sample_name = "Uploaded_Scene"

    if up_mode == "Single 4-Channel RGBT Image (.png)":
        st.caption("Upload a single `.png` where channels 0, 1, 2 are Visible RGB and channel 3 is Thermal infrared.")
        f_single = st.file_uploader("Upload 4-Channel RGBT Image (.png)", type=["png"])
        if f_single:
            img_4ch = Image.open(f_single)
            if img_4ch.mode != "RGBA":
                st.error("Uploaded image must have 4 channels (RGBA format).")
            else:
                arr_4ch = np.array(img_4ch.resize((640, 480)))
                rgb_array = arr_4ch[:, :, :3]
                thermal_array = arr_4ch[:, :, 3]
                sample_name = os.path.splitext(f_single.name)[0]

    else:
        st.caption("Upload separate RGB and Thermal image files.")
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            f_rgb = st.file_uploader("1. Visible Camera (RGB Color Image)", type=["png", "jpg", "jpeg"])
        with col_up2:
            f_th = st.file_uploader("2. Thermal Camera (Grayscale / LWIR Image)", type=["png", "jpg", "jpeg"])

        if f_rgb and f_th:
            rgb_img = Image.open(f_rgb).convert("RGB").resize((640, 480))
            th_img = Image.open(f_th).convert("L").resize((640, 480))
            rgb_array = np.array(rgb_img)
            thermal_array = np.array(th_img)
            sample_name = os.path.splitext(f_rgb.name)[0]

    # --------------------------------------------------------------------------
    # Inference & Results
    # --------------------------------------------------------------------------
    if rgb_array is not None and thermal_array is not None:
        st.markdown("---")
        st.markdown("### 2. Segmentation Results")

        # Preprocessing Tensors
        pil_rgb = Image.fromarray(rgb_array, mode="RGB")
        pil_th = Image.fromarray(thermal_array, mode="L")

        t_rgb = TF.to_tensor(pil_rgb)
        t_rgb = TF.normalize(t_rgb, mean=list(cfg.dataset.rgb_mean), std=list(cfg.dataset.rgb_std)).unsqueeze(0).to(device)

        t_th = TF.to_tensor(pil_th)
        t_th = TF.normalize(t_th, mean=list(cfg.dataset.thermal_mean), std=list(cfg.dataset.thermal_std)).unsqueeze(0).to(device)

        # Inference Timing
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            output_logits = model(t_rgb, t_th)
            pred_class_mask = torch.argmax(output_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if device == "cuda":
            torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        fps_val = 1000.0 / duration_ms if duration_ms > 0 else 0

        # Postprocessing: Colorize & Overlay
        colored_mask = colorize_mask(pred_class_mask)
        blended_overlay = (rgb_array * (1.0 - overlay_alpha) + colored_mask * overlay_alpha).astype(np.uint8)
        colored_thermal = apply_thermal_colormap(thermal_array, thermal_palette)

        # Metrics Strip
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{duration_ms:.1f} ms</div><div class="stat-label">Latency</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{fps_val:.1f} FPS</div><div class="stat-label">Inference Speed</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="stat-box"><div class="stat-number">480 × 640</div><div class="stat-label">Resolution</div></div>', unsafe_allow_html=True)
        with s4:
            detected_classes = [CLASS_NAMES[c] for c in np.unique(pred_class_mask) if c != 0]
            st.markdown(f'<div class="stat-box"><div class="stat-number">{len(detected_classes)}</div><div class="stat-label">Objects Found</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-Side Images Display
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.image(blended_overlay, caption=f"✨ Blended Overlay (Alpha = {overlay_alpha:.2f})", use_column_width=True)
            st.image(rgb_array, caption="📷 Visible RGB Input", use_column_width=True)

        with col_res2:
            st.image(colored_mask, caption="🎯 Predicted Semantic Mask (9 Classes)", use_column_width=True)
            st.image(colored_thermal, caption=f"🌡️ Thermal Infrared ({thermal_palette.capitalize()} Colormap)", use_column_width=True)

        # Download Buttons
        st.markdown("### 3. Export Predictions")
        d_col1, d_col2 = st.columns(2)

        # Buffer for blended overlay
        buf_blend = io.BytesIO()
        Image.fromarray(blended_overlay).save(buf_blend, format="PNG")
        with d_col1:
            st.download_button(
                label="⬇️ Download Overlay Image (PNG)",
                data=buf_blend.getvalue(),
                file_name=f"{sample_name}_csfnet2_overlay.png",
                mime="image/png"
            )

        # Buffer for colored mask
        buf_mask = io.BytesIO()
        Image.fromarray(colored_mask).save(buf_mask, format="PNG")
        with d_col2:
            st.download_button(
                label="⬇️ Download Semantic Mask (PNG)",
                data=buf_mask.getvalue(),
                file_name=f"{sample_name}_csfnet2_mask.png",
                mime="image/png"
            )

        # Detected Classes Legend
        st.markdown("---")
        st.markdown("### 🏷️ Detected Classes & Distribution")
        
        total_px = pred_class_mask.size
        leg_cols = st.columns(len(CLASS_NAMES))
        for i, c_name in enumerate(CLASS_NAMES):
            c_rgb = CLASS_COLORS[i]
            hex_c = f"#{c_rgb[0]:02x}{c_rgb[1]:02x}{c_rgb[2]:02x}"
            px_count = np.sum(pred_class_mask == i)
            pct = (px_count / total_px) * 100
            is_present = px_count > 0

            with leg_cols[i]:
                opacity_style = "opacity: 1.0;" if is_present else "opacity: 0.35;"
                st.markdown(
                    f"<div style='background-color: {hex_c}; {opacity_style} text-align: center; color: white; padding: 6px; border-radius: 6px; font-weight: bold; font-size: 11px;'>"
                    f"{c_name}</div>"
                    f"<div style='text-align: center; font-size: 12px; margin-top: 4px; color: {'#50E3C2' if is_present else '#777'};'>{pct:.1f}%</div>",
                    unsafe_allow_html=True
                )

# ------------------------------------------------------------------------------
# Tab 2: Benchmark Details
# ------------------------------------------------------------------------------
with tab_benchmark:
    st.markdown("### 🏆 CSFNet-2 Benchmark Performance")
    st.markdown("Empirical test results on the **MFNet** dataset using our trained model on an **NVIDIA GeForce RTX 3050 Laptop GPU**:")

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("""
        #### Overall Performance
        * **Overall Test mIoU:** **49.39%** *(+10.93% over baseline)*
        * **Foreground mIoU:** **43.41%** *(+12.12% over baseline)*
        * **Pixel Accuracy:** **97.16%**
        * **Real-Time Speed:** **~72.7 FPS** (13.7 ms latency)
        * **Parameters:** **19.36M**
        """)

    with t_col2:
        st.markdown("""
        #### Day vs Night Sensitivity
        * ☀️ **Daytime mIoU (`test_day`):** **49.79%** (Car IoU: 83.51%)
        * 🌙 **Nighttime mIoU (`test_night`):** **47.64%** (Person IoU: **70.56%**)
        
        > **Day/Night Gap:** Only **2.15%**, demonstrating robust illumination invariance. Under zero-light conditions, thermal infrared radiation ensures pedestrians remain clearly detectable.
        """)
