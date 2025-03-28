"""
Object detection module for the home surveillance system.
"""

import os
import cv2
import torch
import numpy as np
import json
from torchvision.models import detection
from config import CONFIG, COCO_CLASSES


class ObjectDetector:
    """Handles object detection using a pretrained model."""

    def __init__(self, initialize_full=True):
        """Initialize the object detector with either a pre-trained or custom model.
        
        Args:
            initialize_full: If True, initialize the full model for detection.
                            If False, just load class names for UI purposes.
        """
        print("Loading object detection model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load class labels based on model type
        if CONFIG['use_custom_model']:
            self.load_custom_model(initialize_full)
        else:
            self.load_torchvision_model(initialize_full)
        
        if initialize_full:
            print(f"Model loaded on {self.device}")
        
    def load_torchvision_model(self, initialize_full=True):
        """Load a pre-trained model from torchvision.
        
        Args:
            initialize_full: If True, initialize the full model for detection.
                            If False, just set class names for UI purposes.
        """
        if initialize_full:
            self.model = detection.__dict__[CONFIG['model_name']](pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        
        self.class_names = COCO_CLASSES
        self.model_type = 'torchvision'
        
    def load_custom_model(self, initialize_full=True):
        """Load a custom model from file.
        
        Args:
            initialize_full: If True, initialize the full model for detection.
                            If False, just load class names for UI purposes.
        """
        model_path = CONFIG['custom_model_path']
        model_type = CONFIG['custom_model_type'].lower()
        
        if not os.path.exists(model_path):
            print(f"Error: Custom model file not found at {model_path}")
            print("Falling back to torchvision model")
            self.load_torchvision_model(initialize_full)
            return
            
        try:
            # Load model based on format if doing full initialization
            if initialize_full:
                if model_type == 'torchscript':
                    self.model = torch.jit.load(model_path, map_location=self.device)
                    self.model.eval()
                    self.model_type = 'torchscript'
                elif model_type == 'onnx':
                    # ONNX model requires onnxruntime
                    try:
                        import onnxruntime as ort
                        self.ort_session = ort.InferenceSession(model_path)
                        self.model_type = 'onnx'
                    except ImportError:
                        print("Error: onnxruntime not installed. Please install with: pip install onnxruntime")
                        print("Falling back to torchvision model")
                        self.load_torchvision_model(initialize_full)
                        return
                else:
                    print(f"Error: Unsupported model type: {model_type}")
                    print("Falling back to torchvision model")
                    self.load_torchvision_model(initialize_full)
                    return
            else:
                # Just set the model type for UI purposes
                self.model_type = model_type
                
            # Load custom labels if provided
            if CONFIG['custom_model_labels_path'] and os.path.exists(CONFIG['custom_model_labels_path']):
                try:
                    with open(CONFIG['custom_model_labels_path'], 'r') as f:
                        # Try to load as JSON
                        try:
                            labels_data = json.load(f)
                            if isinstance(labels_data, list):
                                self.class_names = labels_data
                            elif isinstance(labels_data, dict):
                                # If it's a dict, we expect a key 'labels' or 'classes'
                                if 'labels' in labels_data:
                                    self.class_names = labels_data['labels']
                                elif 'classes' in labels_data:
                                    self.class_names = labels_data['classes']
                                elif 'names' in labels_data:
                                    self.class_names = labels_data['names']
                                else:
                                    # Use dict keys if they're integers or try values
                                    try:
                                        # Try to convert keys to ints and sort
                                        id_to_name = {int(k): v for k, v in labels_data.items() if k.isdigit()}
                                        max_id = max(id_to_name.keys())
                                        self.class_names = ['__background__'] + [id_to_name.get(i, f"class_{i}") for i in range(1, max_id + 1)]
                                    except (ValueError, AttributeError):
                                        # If keys aren't all integers, use values
                                        self.class_names = ['__background__'] + list(labels_data.values())
                        except json.JSONDecodeError:
                            # If not JSON, try to load as text file with one class per line
                            f.seek(0)  # Go back to start of file
                            lines = f.read().splitlines()
                            self.class_names = ['__background__'] + [line.strip() for line in lines if line.strip()]
                except Exception as e:
                    print(f"Error loading custom labels: {e}")
                    self.class_names = [f"class_{i}" for i in range(100)]  # Default to generic class names
            else:
                # If no labels file, use COCO classes as default
                self.class_names = COCO_CLASSES
                
            if initialize_full:
                print(f"Loaded custom {model_type} model from {model_path}")
            if hasattr(self, 'class_names'):
                print(f"Using {len(self.class_names)} class labels")
                
        except Exception as e:
            print(f"Error loading custom model: {e}")
            print("Falling back to torchvision model")
            self.load_torchvision_model(initialize_full)

    def preprocess_image(self, frame):
        """Preprocess image for model input."""
        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Handle specific preprocessing based on model type
        if self.model_type == 'onnx':
            # ONNX models often require specific preprocessing
            # Resize to expected input size (many models use 640x640)
            input_height, input_width = 640, 640  # Default size, adjust based on model
            
            # Store original dimensions for scaling boxes back
            self.orig_height, self.orig_width = frame.shape[:2]
            
            # Resize image
            resized = cv2.resize(rgb_frame, (input_width, input_height))
            
            # Normalize pixel values to 0-1
            normalized = resized.astype(np.float32) / 255.0
            
            # Transpose to channel-first format (NCHW)
            transposed = normalized.transpose(2, 0, 1)
            
            # Add batch dimension
            batched = np.expand_dims(transposed, 0)
            
            return batched
        else:
            # Standard PyTorch preprocessing
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(rgb_frame.transpose(2, 0, 1)).float().div(255.0)
            image_tensor = image_tensor.to(self.device)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor

    def detect_objects(self, frame):
        """
        Detect objects in the given frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detection dictionaries, each containing box, confidence, class_id, class_name
        """
        # Preprocess the image based on model type
        input_data = self.preprocess_image(frame)
        
        results = []
        
        try:
            if self.model_type == 'onnx':
                # Get model input name
                input_name = self.ort_session.get_inputs()[0].name
                
                # Run inference
                outputs = self.ort_session.run(None, {input_name: input_data})
                
                # Parse ONNX model outputs (format varies by model)
                # Here's a generic approach for YOLO-style models
                if len(outputs) == 1:
                    # Most YOLO models output a single tensor with [batch, num_detections, 5+num_classes]
                    detections = outputs[0]
                    
                    for i in range(detections.shape[1]):
                        # Format is usually [x, y, w, h, confidence, class_prob_1, class_prob_2, ...]
                        if detections[0, i, 4] >= CONFIG['confidence_threshold']:
                            # Get box coordinates
                            cx, cy, w, h = detections[0, i, 0:4]
                            
                            # Convert center coordinates to corners
                            x1 = int((cx - w/2) * self.orig_width)
                            y1 = int((cy - h/2) * self.orig_height)
                            x2 = int((cx + w/2) * self.orig_width)
                            y2 = int((cy + h/2) * self.orig_height)
                            
                            # Get class ID (may vary by model format)
                            class_scores = detections[0, i, 5:]
                            class_id = int(np.argmax(class_scores))
                            confidence = float(detections[0, i, 4] * class_scores[class_id])
                            
                            # Get class name
                            class_name = self.class_names[class_id + 1] if class_id + 1 < len(self.class_names) else f"class_{class_id}"
                            
                            # Filter by enabled objects if list is not empty
                            if CONFIG['enabled_objects'] and class_name not in CONFIG['enabled_objects']:
                                continue
                                
                            results.append({
                                'box': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': class_name
                            })
                else:
                    # Handle other ONNX model output formats
                    # This is a simplified example - real implementation depends on model
                    boxes = outputs[0][0]
                    scores = outputs[1][0]
                    labels = outputs[2][0]
                    
                    for box, score, label in zip(boxes, scores, labels):
                        if score >= CONFIG['confidence_threshold']:
                            # Convert box to pixel coordinates
                            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in 
                                            zip(box, [self.orig_width, self.orig_height, 
                                                    self.orig_width, self.orig_height])]
                            
                            # Get class name
                            class_id = int(label)
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                            
                            # Filter by enabled objects if list is not empty
                            if CONFIG['enabled_objects'] and class_name not in CONFIG['enabled_objects']:
                                continue
                                
                            results.append({
                                'box': [x1, y1, x2, y2],
                                'confidence': float(score),
                                'class_id': class_id,
                                'class_name': class_name
                            })
            else:
                # PyTorch models (torchvision or TorchScript)
                with torch.no_grad():
                    predictions = self.model(input_data)
                
                # Extract results
                if isinstance(predictions, dict) or (isinstance(predictions, list) and isinstance(predictions[0], dict)):
                    # Standard torchvision detection model output
                    if isinstance(predictions, list):
                        predictions = predictions[0]  # Use first batch item
                        
                    boxes = predictions['boxes'].cpu().numpy()
                    scores = predictions['scores'].cpu().numpy()
                    labels = predictions['labels'].cpu().numpy()
                    
                    # Filter by confidence threshold
                    mask = scores >= CONFIG['confidence_threshold']
                    boxes = boxes[mask]
                    scores = scores[mask]
                    labels = labels[mask]
                    
                    for box, score, label in zip(boxes, scores, labels):
                        class_id = int(label)
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        # Filter by enabled objects if the list is not empty
                        if CONFIG['enabled_objects'] and class_name not in CONFIG['enabled_objects']:
                            continue
                            
                        results.append({
                            'box': box.astype(int).tolist(),  # [x1, y1, x2, y2]
                            'confidence': float(score),
                            'class_id': class_id,
                            'class_name': class_name
                        })
                else:
                    # Handle other model output formats
                    # This might be a custom TorchScript model with different output format
                    print("Unsupported model output format")
                    
        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
            
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
            class_name = detection['class_name']
            x1, y1, x2, y2 = detection['box']
            label = f"{class_name}: {detection['confidence']:.2f}"
            
            # All detected objects are highlighted with the highlight color
            # (since they're all configured to be detected)
            color = CONFIG['detection_colors']['highlight']

            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1),
                          color, -1)

            # Draw label text (black text on colored background)
            cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                        2)

        return result_frame