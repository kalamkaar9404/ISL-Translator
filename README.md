#  ISL-Net: Real-Time Indian Sign Language Recognition
### A Hybrid Computer Vision Pipeline using MediaPipe & ResNet18

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)
![Accuracy](https://img.shields.io/badge/Accuracy-99.9%25-brightgreen)

##  Project Novelty: The "Smart-Crop" Pipeline

Most Sign Language recognition systems suffer from two major problems:
1.  **Background Noise:** Standard CNNs get confused by messy backgrounds (furniture, clothes, lighting).
2.  **Information Loss:** Landmark-only approaches (like LSTM on coordinates) lose visual context like finger crossing or texture.

**Our Solution? A Hybrid "Smart-Crop" Architecture.**

Instead of feeding the raw frame to the AI, we implement a two-stage pipeline:
1.  **Detector (MediaPipe):** Acts as a lightweight "sentry" to locate the hand and extract a dynamic **Region of Interest (ROI)**.
2.  **Classifier (ResNet18):** A heavy-duty convolutional network that receives *only* the cropped, normalized hand image.

**Result:** The model becomes **Background Invariant**. It doesn't matter if you are in a dark room or outside; the ResNet only sees the hand.

---

##  Performance
- **Dataset:** Indian Sign Language (Kaggle)
- **Classes:** 36 (0-9, A-Z)
- **Test Accuracy:** **99.95%** (on Validation Set)
- **Inference Speed:** Real-time (~30 FPS on GPU)

---

##  Tech Stack
- **Core Framework:** PyTorch (Transfer Learning on ResNet18)
- **Hand Detection:** Google MediaPipe
- **Image Processing:** OpenCV & Albumentations (Data Augmentation)
- **Visualization:** Matplotlib & Seaborn (Confusion Matrices)
- **Environment:** Google Colab (T4 GPU)

---

##  Step-by-Step Implementation

### 1. Prerequisites
You need a Kaggle account to download the dataset.
1. Go to your Kaggle Account -> Settings -> API.
2. Click **"Create New API Token"** to download `kaggle.json`.

### 2. Setup (Google Colab)
This project is optimized for Google Colab. Upload your notebook and run the following setup cell to fix version conflicts (NumPy 2.0 vs PyTorch):

```python
# Run this first to ensure environment stability
!pip uninstall -y numpy
!pip install "numpy<2.0" --force-reinstall
!pip install scikit-learn --upgrade mediapipe kaggle
# RESTART RUNTIME AFTER INSTALLATION
