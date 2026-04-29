import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from pytorch_lightning.callbacks import ModelCheckpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===============================
# PERCEPTUAL LOSS
# ===============================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, x, y):
        self.vgg = self.vgg.to(x.device)
        x = (x + 1) / 2
        y = (y + 1) / 2
        return F.l1_loss(self.vgg(x), self.vgg(y))

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
        top = torch.randint(0, H - hole_h, (1,)).item()
        left = torch.randint(0, W - hole_w, (1,)).item()
        mask[:, top:top+hole_h, left:left+hole_w] = 0
        return mask

# ===============================
# DATASET
# ===============================
class CelebaDataset(Dataset):
    def __init__(self, data_dir, mask_generator, img_size=128):
        self.paths = [
            p for p in Path(data_dir).glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ]
        self.mask_generator = mask_generator
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        y = self.transform(img)
        mask = self.mask_generator(y)
        x = y * mask
        return x, y

# ===============================
# REAL UNET
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
# VISUALIZATION CALLBACK
# ===============================
class VisualizationCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 5 != 0:
            return
        pl_module.eval()
        batch = next(iter(trainer.train_dataloader))
        x, y = batch
        x, y = x[:4].to(pl_module.device), y[:4].to(pl_module.device)
        with torch.no_grad():
            pred = pl_module(x)

        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for i in range(4):
            for ax, img, title in zip(
                axes[:, i],
                [x[i], pred[i], y[i]],
                ["Masked", "Predicted", "Ground Truth"]
            ):
                img = (img.cpu().permute(1,2,0).numpy() * 0.5 + 0.5).clip(0,1)
                ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")

        Path("outputs").mkdir(exist_ok=True)
        plt.savefig(f"outputs/epoch_{trainer.current_epoch:03d}.png")
        plt.close()
        pl_module.train()

# ===============================
# LIGHTNING MODEL
# ===============================
class InpaintingModel(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.unet = UNet()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.lambda_perc = 0.1
        self.lr = lr

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        l1   = self.l1_loss(y_pred, y)
        perc = self.perceptual_loss(y_pred, y)
        loss = l1 + self.lambda_perc * perc
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        l1   = self.l1_loss(y_pred, y)
        perc = self.perceptual_loss(y_pred, y)
        loss = l1 + self.lambda_perc * perc
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        # ✅ LR scheduler for better quality
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
        return [optimizer], [scheduler]

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    DATA_DIR   = "data/processed/train"
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    LR         = 1e-4
    IMG_SIZE   = 128   # ✅ keep high quality
    NUM_SAMPLES = 10000  # ✅ 10k = ~14 min/epoch = ~24hrs total

    full_dataset = CelebaDataset(DATA_DIR, MaskGenerator(), img_size=IMG_SIZE)

    # ✅ Use 10k samples for 24hr target
    dataset = torch.utils.data.Subset(full_dataset, range(NUM_SAMPLES))

    train_size = int(0.9 * len(dataset))  # 9k train
    val_size   = len(dataset) - train_size  # 1k val
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2      # ✅ 2 workers = faster but not laggy
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2
    )

    model = InpaintingModel(lr=LR)

    # ✅ CHECKPOINTS
    best_checkpoint = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        verbose=True,
    )

    periodic_checkpoint = ModelCheckpoint(
        dirpath="checkpoints/periodic/",
        filename="epoch-{epoch:03d}",
        every_n_epochs=10,
        save_top_k=-1,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        callbacks=[
            VisualizationCallback(),
            best_checkpoint,
            periodic_checkpoint,
        ],
    )

    trainer.fit(model, train_loader, val_loader)

# ✅ TO RESUME IF MAC CRASHES:
# trainer.fit(model, train_loader, val_loader,
#             ckpt_path="checkpoints/periodic/epoch-060.ckpt")