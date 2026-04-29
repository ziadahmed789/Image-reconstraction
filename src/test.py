import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===============================
# CONFIG — change these paths!
# ===============================
CHECKPOINT_PATH = "checkpoints/best-epoch=099-val_loss=0.0167.ckpt"
TEST_IMAGES_DIR = "data/processed/train"
OUTPUT_DIR      = "test_outputs"
IMG_SIZE        = 128
MASK_RATIO      = 0.25
NUM_IMAGES      = 8

# ===============================
# SSIM IMPLEMENTATION ✅
# ===============================
def gaussian_kernel(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel / kernel.sum()

def ssim(pred, target, window_size=11):
    """
    Compute SSIM between pred and target.
    Inputs: [B, C, H, W] tensors in range [-1, 1]
    Returns: scalar SSIM value (higher = better, max = 1.0)
    """
    # Convert from [-1, 1] to [0, 1]
    pred   = (pred + 1) / 2
    target = (target + 1) / 2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    kernel = gaussian_kernel(window_size).to(pred.device)
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size)

    mu1 = F.conv2d(pred,   kernel, padding=window_size//2, groups=pred.shape[1])
    mu2 = F.conv2d(target, kernel, padding=window_size//2, groups=pred.shape[1])

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred,     kernel, padding=window_size//2, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size//2, groups=pred.shape[1]) - mu2_sq
    sigma12   = F.conv2d(pred * target,   kernel, padding=window_size//2, groups=pred.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()

# ===============================
# UNET (same as training)
# ===============================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.tanh(self.out(d1))

# ===============================
# MASK GENERATOR
# ===============================
class MaskGenerator:
    def __init__(self, mask_ratio=0.25):
        self.mask_ratio = mask_ratio

    def __call__(self, img):
        _, H, W = img.shape
        mask = torch.ones(1, H, W)
        hole_h = int(H * self.mask_ratio)
        hole_w = int(W * self.mask_ratio)
        top  = torch.randint(0, H - hole_h, (1,)).item()
        left = torch.randint(0, W - hole_w, (1,)).item()
        mask[:, top:top+hole_h, left:left+hole_w] = 0
        return mask

# ===============================
# LOAD MODEL
# ===============================
def load_model(checkpoint_path):
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {
        k.replace("unet.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("unet.")
    }
    model = UNet()
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully!")
    return model

# ===============================
# COMPUTE ALL METRICS ✅
# ===============================
def compute_metrics(pred, target):
    # L1
    l1 = F.l1_loss(pred, target).item()
    # PSNR
    mse = F.mse_loss(pred, target).item()
    psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item() if mse > 0 else 100.0
    # Accuracy
    accuracy = (1 - l1) * 100
    # SSIM ✅
    ssim_score = ssim(pred, target)
    return l1, psnr, accuracy, ssim_score

# ===============================
# TEST SINGLE IMAGE
# ===============================
def test_single_image(model, image_path, mask_generator, transform, device):
    img = Image.open(image_path).convert("RGB")
    y = transform(img).unsqueeze(0).to(device)
    mask = mask_generator(y.squeeze(0)).unsqueeze(0).to(device)
    x = y * mask

    with torch.no_grad():
        pred = model(x.to(device))

    inv_mask = 1 - mask
    l1, psnr, accuracy, ssim_score = compute_metrics(
        pred * inv_mask,
        y * inv_mask
    )

    return x.squeeze(0), pred.squeeze(0), y.squeeze(0), l1, psnr, accuracy, ssim_score

# ===============================
# SSIM RATING HELPER
# ===============================
def ssim_rating(score):
    if score >= 0.98: return "🏆 Excellent"
    if score >= 0.95: return "✅ Very Good"
    if score >= 0.90: return "👍 Good"
    if score >= 0.80: return "⚠️  Fair"
    return "❌ Poor"

# ===============================
# MAIN
# ===============================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Auto-find checkpoint
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        checkpoints = list(Path("checkpoints").glob("best-*.ckpt"))
        if not checkpoints:
            checkpoints = list(Path("checkpoints").glob("*.ckpt"))
        if checkpoints:
            ckpt_path = sorted(checkpoints)[-1]
            print(f"⚠️  Auto-found checkpoint: {ckpt_path}")
        else:
            print("❌ No checkpoint found!")
            return

    model = load_model(str(ckpt_path)).to(device)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    mask_generator = MaskGenerator(MASK_RATIO)

    image_paths = [
        p for p in Path(TEST_IMAGES_DIR).glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ][:NUM_IMAGES]

    if not image_paths:
        print(f"❌ No images found in {TEST_IMAGES_DIR}")
        return

    print(f"🖼️  Testing on {len(image_paths)} images...\n")
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    all_l1, all_psnr, all_acc, all_ssim = [], [], [], []

    fig, axes = plt.subplots(3, len(image_paths), figsize=(len(image_paths) * 3, 9))

    for i, img_path in enumerate(image_paths):
        x, pred, y, l1, psnr, accuracy, ssim_score = test_single_image(
            model, img_path, mask_generator, transform, device
        )

        all_l1.append(l1)
        all_psnr.append(psnr)
        all_acc.append(accuracy)
        all_ssim.append(ssim_score)

        print(
            f"Image {i+1}: "
            f"L1={l1:.4f} | "
            f"PSNR={psnr:.2f}dB | "
            f"Accuracy={accuracy:.2f}% | "
            f"SSIM={ssim_score:.4f} {ssim_rating(ssim_score)}"
        )

        for ax, img_tensor, title in zip(
            axes[:, i],
            [x, pred, y],
            ["Masked", "Predicted", "Ground Truth"]
        ):
            img_np = (img_tensor.cpu().permute(1,2,0).numpy() * 0.5 + 0.5).clip(0,1)
            ax.imshow(img_np)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    avg_ssim = sum(all_ssim) / len(all_ssim)
    avg_acc  = sum(all_acc)  / len(all_acc)
    avg_psnr = sum(all_psnr) / len(all_psnr)

    plt.suptitle(
        f"Avg Accuracy: {avg_acc:.2f}% | "
        f"Avg PSNR: {avg_psnr:.2f}dB | "
        f"Avg SSIM: {avg_ssim:.4f} {ssim_rating(avg_ssim)}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/test_results.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    # ===============================
    # FINAL SUMMARY
    # ===============================
    print("\n" + "="*55)
    print("📊 FINAL TEST SUMMARY")
    print("="*55)
    print(f"✅ Images Tested   : {len(image_paths)}")
    print(f"✅ Avg L1 Loss     : {sum(all_l1)/len(all_l1):.4f}")
    print(f"✅ Avg PSNR        : {avg_psnr:.2f} dB")
    print(f"✅ Avg Accuracy    : {avg_acc:.2f}%")
    print(f"✅ Best Accuracy   : {max(all_acc):.2f}%")
    print(f"✅ Worst Accuracy  : {min(all_acc):.2f}%")
    print(f"✅ Avg SSIM        : {avg_ssim:.4f}  {ssim_rating(avg_ssim)}")
    print(f"✅ Best SSIM       : {max(all_ssim):.4f}")
    print(f"✅ Worst SSIM      : {min(all_ssim):.4f}")
    print("="*55)
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()