import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from main import UNet

# ------------------
# LOAD MODEL
# ------------------
@st.cache_resource
def load_model():
    model = UNet()

    checkpoint = torch.load(
        "checkpoints/best-epoch=099-val_loss=0.0167.ckpt",
        map_location="cpu"
    )

    state_dict = {
        k.replace("unet.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("unet.")
    }

    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# ------------------
# TRANSFORM
# ------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------
# MASK FUNCTION (ثابت للـ demo)
# ------------------
def apply_mask(x):
    mask = torch.ones_like(x)
    _, _, h, w = x.shape

   
    mask[:, :, h//4:h//2, w//4:w//2] = 0

    return x * mask

# ------------------
# UI
# ------------------
st.set_page_config(page_title="Face Inpainting AI", layout="centered")

st.title("🧠 Face Inpainting AI")
st.write("Upload an image and let the model reconstruct missing parts.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # ------------------
    # ORIGINAL
    # ------------------
    img = Image.open(uploaded_file).convert("RGB")

    # ------------------
    # PREPROCESS
    # ------------------
    x = transform(img).unsqueeze(0)

    # ------------------
    # APPLY MASK
    # ------------------
    x_masked = apply_mask(x)

    # ------------------
    # INFERENCE
    # ------------------
    with torch.no_grad():
        output = model(x_masked)

    # ------------------
    # POSTPROCESS OUTPUT
    # ------------------
    output = output[0].detach().cpu().permute(1, 2, 0).numpy()
    output = (output + 1) / 2
    output = np.clip(output, 0, 1)

    masked_img = x_masked[0].detach().cpu().permute(1, 2, 0).numpy()
    masked_img = (masked_img + 1) / 2
    masked_img = np.clip(masked_img, 0, 1)

    # ------------------
    # DISPLAY
    # ------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(masked_img, caption="Masked Input", use_container_width=True)

    with col2:
        st.image(output, caption="Predicted", use_container_width=True)

    with col3:
        st.image(img, caption="Ground Truth", use_container_width=True)