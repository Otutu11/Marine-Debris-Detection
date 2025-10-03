import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Step 1: Load and preprocess image ===
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # normalize
    return img[np.newaxis, ...]

# === Step 2: Load trained model ===
def load_segmentation_model(model_path='marine_debris_unet.h5'):
    return load_model(model_path)

# === Step 3: Predict mask ===
def predict_debris(model, preprocessed_image):
    pred_mask = model.predict(preprocessed_image)
    pred_mask = (pred_mask[0] > 0.5).astype(np.uint8)
    return pred_mask

# === Step 4: Visualize ===
def visualize_results(image_path, pred_mask):
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (256, 256))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Detected Marine Debris')
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# === Step 5: Run the pipeline ===
if __name__ == "__main__":
    image_path = 'sample_ocean_image.jpg'  # replace with your image
    model_path = 'marine_debris_unet.h5'   # trained U-Net model

    image = load_and_preprocess_image(image_path)
    model = load_segmentation_model(model_path)
    pred_mask = predict_debris(model, image)
    visualize_results(image_path, pred_mask)
