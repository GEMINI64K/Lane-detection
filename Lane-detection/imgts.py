import cv2
import numpy as np

# Configuration for Raspberry Pi 5 optimization
CONFIG = {
    'model_config': 'ssd_mobilenet_v1_coco.pbtxt',
    'model_weights': 'frozen_inference_graph.pb',
    'input_size': (300, 300),  # MobileNet SSD typically uses 300x300 input
    'confidence_threshold': 0.5,  # Minimum confidence for detection
    'frame_skip': 2,  # Process every 2nd frame to reduce load
    'display_size': (640, 480)  # Display resolution
}

# COCO class labels (subset relevant to traffic)
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Traffic-related classes to highlight
TRAFFIC_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']

def load_model():
    """Load the MobileNet SSD model using OpenCV DNN."""
    try:
        net = cv2.dnn.readNetFromTensorflow(CONFIG['model_weights'], CONFIG['model_config'])
        print("Model loaded successfully.")
        return net
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_frame(frame):
    """Preprocess the frame for MobileNet SSD input."""
    # Downscale frame to match model input size
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=CONFIG['input_size'], mean=(127.5, 127.5, 127.5),
        swapRB=True, crop=False
    )
    return blob

def detect_objects(net, frame):
    """Run object detection on the frame and return detections."""
    blob = preprocess_frame(frame)
    net.setInput(blob)
    detections = net.forward()
    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes and labels for detected objects."""
    height, width = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIG['confidence_threshold']:
            class_id = int(detections[0, 0, i, 1])
            class_name = COCO_CLASSES[class_id]

            # Only process traffic-related classes
            if class_name not in TRAFFIC_CLASSES:
                continue

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure bounding box stays within frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(width - 1, endX), min(height - 1, endY)

            # Draw bounding box and label
            label = f"{class_name}: {confidence:.2f}"
            color = (0, 255, 0) if class_name in ['car', 'truck', 'bus'] else (0, 0, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, startY - 10 if startY - 10 > 10 else startY + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def main():
    # Load the model
    net = load_model()
    if net is None:
        return

    # Initialize video capture
    capture = cv2.VideoCapture('vid1.mp4')  # Replace with 0 for live camera
    if not capture.isOpened():
        print("Error: Could not open video file.")
        return

    # Set resolution for better performance
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Skip frames to reduce processing load
        frame_count += 1
        if frame_count % CONFIG['frame_skip'] != 0:
            continue

        # Detect objects
        detections = detect_objects(net, frame)

        # Draw detections on the frame
        result = draw_detections(frame, detections)

        # Resize for display
        rsimg = cv2.resize(result, CONFIG['display_size'])

        # Display the result
        cv2.imshow('Traffic Identification', rsimg)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()