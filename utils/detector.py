"""
Object detection module for the home surveillance system.
"""

import cv2
import torch
from torchvision.models import detection
from config import CONFIG, COCO_CLASSES


class ObjectDetector:
    """Handles object detection using a pretrained model."""

    def __init__(self):
        """Initialize the object detector with a pre-trained model."""
        print("Loading object detection model...")
        # Load a pre-trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = detection.__dict__[CONFIG['model_name']](pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def detect_objects(self, frame):
        """
        Detect objects in the given frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detection dictionaries, each containing box, confidence, class_id, class_name
        """
        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(rgb_frame.transpose(2, 0, 1)).float().div(255.0)
        image_tensor = image_tensor.to(self.device)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Extract results
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Filter by confidence threshold
        mask = scores >= CONFIG['confidence_threshold']
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        results = []
        for box, score, label in zip(boxes, scores, labels):
            results.append({
                'box': box.astype(int).tolist(),  # [x1, y1, x2, y2]
                'confidence': float(score),
                'class_id': int(label),
                'class_name': COCO_CLASSES[label]
            })

        return results

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: Original BGR image
            detections: List of detection dictionaries

        Returns:
            Frame with bounding boxes and labels drawn
        """
        result_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"

            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1),
                          (0, 255, 0), -1)

            # Draw label text
            cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                        2)

        return result_frame