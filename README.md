# 🚪 Door Handle Classification – Inference Only

A lightweight Python package to detect and classify **door knobs and handles** from a live USB camera feed using:

- ✅ **TensorFlow Lite** for classification (MobileNetV2)
- ✅ **PyTorch Faster R-CNN** for object detection
- ⚡ Optimized for edge deployment (Raspberry Pi 5, Jetson Nano, etc.)

---

## 📁 Folder Structure

```
door_handle_inference/
├── classify_door_handle.py        # Real-time inference script
├── door_classifier.tflite         # Pre-trained MobileNetV2 classifier (TFLite)
├── requirements.txt               # Installation requirements
```

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 On Raspberry Pi or Jetson, ensure `tflite-runtime` is installed instead of full TensorFlow.

### 2️⃣ Run Real-Time Prediction from Camera

```bash
python predict_realtime.py --webcam
```

> ✅ Press `Q` to exit the camera feed window.

---

## 🧠 Model Used

- Architecture: **MobileNetV2** (transfer learning)
- Input: `128x128 RGB` image
- Classes: `['handle', 'knob']`
- Format: TensorFlow Lite (`.tflite`)

---

## ⚙️ Dependencies (see `requirements.txt`)

- `tflite-runtime`
- `torch`, `torchvision`
- `opencv-python`
- `numpy`

---


## 🧼 Cleanup
```bash
rm -rf __pycache__ *.log *.png
```
