# 🌊 Marine Debris Detection using U-Net

This project applies deep learning (U-Net architecture) to detect marine debris from satellite or drone imagery. It supports image segmentation for environmental monitoring using datasets like MARIDA and NASA's marine debris archive.

## 🧠 Model Overview

- **Architecture**: U-Net (custom implementation in TensorFlow/Keras)
- **Input Size**: 256x256 RGB images
- **Output**: Binary segmentation mask (debris vs. non-debris)

## 📂 Project Structure

Marine-Debris-Detection/
│
├── train_images/ # Training images (RGB)
├── train_masks/ # Corresponding binary masks
├── marine_debris_unet.h5 # Saved U-Net model
├── marine_debris_detection.py # Inference script
├── train_unet.py # Training script
├── requirements.txt # Dependencies
└── README.md # Project documentation


## 📥 Dataset Links

1. **[MARIDA: Marine Debris Archive](https://doi.org/10.5281/zenodo.5151941)**
2. **[NASA Marine Debris Detection](https://github.com/NASA-IMPACT/marine_debris_ML)**
3. **[Seaclear Dataset (Underwater)](https://www.nature.com/articles/s41597-024-03759-2)**

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/Marine-Debris-Detection.git
cd Marine-Debris-Detection
pip install -r requirements.txt

🏋️‍♂️ Training the U-Net Model

python train_unet.py

Ensure you have the following folder structure for training:

train_images/
    debris1.jpg
    debris2.jpg
train_masks/
    debris1_mask.png
    debris2_mask.png

    📌 Images and masks must be size-matched and preprocessed to 256x256.

🔍 Running Inference

python marine_debris_detection.py

Edit marine_debris_detection.py to update the image path and model path before running.
📊 Evaluation (optional)

You can include IoU, Dice score, and precision/recall metrics if you use additional evaluation scripts.
🛠 Dependencies

tensorflow
numpy
opencv-python
matplotlib

Install with:

pip install tensorflow numpy opencv-python matplotlib

🙌 Credits

    Sentinel-2 Imagery via Copernicus Open Access Hub

    MARIDA Dataset – Zenodo

    NASA IMPACT – Marine Debris ML Research

Author Name: Otutu Anslem
Github: https://github.com/Otutu11

📜 License

MIT License – free to use, adapt, and distribute with attribution.


