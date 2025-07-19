# predict_realtime.py (updated to use TensorFlow Lite)
import cv2
import numpy as np
import argparse
import torch
from torchvision import models
from torchvision.transforms import functional as F
import tensorflow as tf
tflite = tf.lite

# Arguments
parser = argparse.ArgumentParser(description="Door Handle Detection and Classification")
parser.add_argument('--video', type=str, help="Path to video file")
parser.add_argument('--webcam', action='store_true', help="Use webcam instead of video file")
parser.add_argument('--image', type=str, help="Path to a single image file")
args = parser.parse_args()

# Load TFLite model
tflite_model_path = 'door_handle_gru_model.tflite'
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 128
labels = ['knob', 'handle']

# Load object detector (Faster R-CNN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.to(device).eval()

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def detect_doorhandle(image):
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = faster_rcnn(image_tensor)[0]
    handles = []
    for bbox, score in zip(outputs['boxes'], outputs['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, bbox)
            handles.append((x1, y1, x2, y2))
    return handles

def predict_tflite_model(crop):
    image = sharpen_image(crop)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    label = labels[np.argmax(preds)]
    confidence = np.max(preds)
    return label, confidence

# Image prediction
if args.image:
    print(f"Using image file: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print("Error: Could not load image.")
        exit()

    handles = detect_doorhandle(image)
    for (x1, y1, x2, y2) in handles:
        crop = image[y1:y2, x1:x2]
        label, confidence = predict_tflite_model(crop)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Prediction on Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Image prediction completed.")
    exit()

# Video/Webcam prediction
if args.video:
    print(f"Using video file: {args.video}")
    cap = cv2.VideoCapture(args.video)
elif args.webcam or not args.video:
    print("Using webcam for prediction...")
    cap = cv2.VideoCapture(0)
    cv2.waitKey(2000)
else:
    raise ValueError("Specify --video path, --image path, or use --webcam")

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    handles = detect_doorhandle(frame)
    for (x1, y1, x2, y2) in handles:
        crop = frame[y1:y2, x1:x2]
        label, confidence = predict_tflite_model(crop)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Door Handle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Door handle detection and classification completed.")
exit()