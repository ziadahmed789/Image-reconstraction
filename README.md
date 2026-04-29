# 🧠 Face Image Inpainting using U-Net

A deep learning project for reconstructing missing regions in facial images using a **U-Net architecture** trained with **PyTorch Lightning** on the **CelebA dataset**. The model learns to fill randomly masked patches while preserving facial structure, visual consistency, and perceptual realism.

An interactive **Streamlit demo** is included so you can test the model on your own face images.

---

## 📸 Sample Output

> Results after 95 epochs of training — Masked Input | Predicted Reconstruction | Ground Truth

![Epoch 95 Output](results/epoch_095.png)

> Test results across 8 unseen images:

![Test Results](test_outputs/test_results.png)

---

## 📊 Test Results

| Metric | Score |
|---|---|
| Average L1 Loss | 0.0066 |
| Average PSNR | 28.18 dB |
| Average Accuracy | 99.34% |
| Best Accuracy | 99.77% |
| Worst Accuracy | 98.85% |

---

## 📁 Project Structure

```
Image-reconstraction/
│
├── src/
│   ├── main.py                 # U-Net architecture, dataset, training loop
│   ├── app.py                  # Streamlit inference demo
│   ├── test.py                 # Evaluation script
│   ├── mask_generator.py       # Mask generation utilities
│   └── preprocess_celeba.py    # CelebA preprocessing script
│
├── data/
│   └── processed/              # Preprocessed dataset splits
│       └── train/
│
├── results/                    # Sample outputs saved every 5 epochs
├── test_outputs/               # Test evaluation results
├── outputs/                    # Training visualization outputs
├── checkpoints/                # Saved model checkpoints
├── lightning_logs/             # PyTorch Lightning training logs
├── unet.ckpt                   # Trained model checkpoint
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

This project uses the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) — a large-scale face attributes dataset with over 200,000 celebrity images.

All images are resized to **128×128 pixels** during training.

| Split | Images Used |
|---|---|
| Training | 9,000 |
| Validation | 1,000 |
| **Total** | **10,000** |

> ⚠️ Dataset files are **not included** in this repository due to size. Download CelebA from the [official source](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), place it in `data/`, and run the preprocessing script before training.

---

## ⚙️ Preprocessing

The preprocessing pipeline handles:

- Loading raw CelebA images from disk
- Resizing all images to **128×128**
- Verifying image integrity (skipping corrupted files)
- Saving processed images to `data/processed/train/`

Run it with:

```bash
python src/preprocess_celeba.py
```

---

## 🏗️ Model Architecture

The model is based on the **U-Net** architecture — a symmetric encoder-decoder network with skip connections designed for image-to-image tasks.

```
Input (Masked Face 128×128)
        │
   ┌────▼────┐
   │ Encoder  │  Conv → BN → ReLU blocks at 64, 128, 256 channels
   │          │  MaxPool downsampling at each stage
   └────┬────┘
        │
   ┌────▼────┐
   │Bottleneck│  512-channel feature representation
   └────┬────┘
        │
   ┌────▼────┐
   │ Decoder  │  ConvTranspose upsampling at 256, 128, 64 channels
   │          │  Skip connections concatenated at each stage
   └────┬────┘
        │
   Output (Reconstructed Face 128×128)
```

**Loss Function:**
- **L1 Loss** — pixel-wise reconstruction accuracy
- **Perceptual Loss** (VGG16 features, λ=0.1) — ensures visual and semantic realism

---

## 🏋️ Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 128×128 |
| Batch Size | 32 |
| Epochs | 100 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| LR Scheduler | StepLR (×0.5 every 30 epochs) |
| Precision | bf16-mixed |
| Accelerator | Auto (MPS / CUDA / CPU) |
| Mask Size | 25% of image (random location) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ziadahmed789/Image-reconstraction.git
cd Image-reconstraction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires **Python 3.8+**

### 3. Prepare the dataset

Download CelebA, place it in `data/`, then run:

```bash
python src/preprocess_celeba.py
```

### 4. Train the model

```bash
python src/main.py
```

Training logs are saved automatically to `lightning_logs/`. Visual outputs are saved to `outputs/` every 5 epochs.

### 5. Evaluate the model

```bash
python src/test.py
```

---

## 🖥️ Streamlit Demo

Run the interactive inpainting app locally:

```bash
cd src
streamlit run app.py
```

> ⚠️ **Note:** The app must be run from inside the `src/` directory due to the local import of `main.py`. It also expects the checkpoint at `checkpoints/best-epoch=099-val_loss=0.0167.ckpt`. If you are using a different checkpoint, update the path in `app.py` accordingly.

Upload any face image → the app applies a fixed center mask and displays the **Masked Input**, **Model Reconstruction**, and **Original** side by side.

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/)
- [torchvision](https://pytorch.org/vision/)
- [Streamlit](https://streamlit.io/)
- [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

---

## 📄 Citation

If you use this project in your research, please cite it as:

```
Z. Ahmed et al., "Face Image Inpainting using U-Net," GitHub, 2025.
Available: https://github.com/ziadahmed789/Image-reconstraction
```

---

## 📬 Contact

**Ziad Ahmed**
GitHub: [@ziadahmed789](https://github.com/ziadahmed789)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
