"""
Object detection module for the home surveillance system.
"""

import os
import cv2
import torch
import json
import logging
import traceback
from torchvision.models import detection

from config import CONFIG, COCO_CLASSES

logger = logging.getLogger("detector")


class ObjectDetector:
    """Handles object detection using a pretrained model."""

    def __init__(self, initialize_full=True):
        """Initialize the object detector with either a pre-trained or custom model.

        Args:
            initialize_full: If True, initialize the full model for detection.
                            If False, just load class names for UI purposes.
        """
        logger.info("Initializing object detector")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load class labels based on model type
        if CONFIG["use_custom_model"]:
            logger.info("Using custom model configuration")
            self.load_custom_model(initialize_full)
        else:
            logger.info("Using built-in torchvision model")
            self.load_torchvision_model(initialize_full)

        if initialize_full:
            logger.info(f"Model loaded successfully on {self.device}")
            if hasattr(self, "class_names"):
                logger.info(f"Available classes: {len(self.class_names)}")
                logger.debug(f"Class names: {self.class_names[:20]}...")

            # Log enabled objects configuration
            if CONFIG["enabled_objects"]:
                logger.info(f"Enabled objects: {CONFIG['enabled_objects']}")
            else:
                logger.info("All objects enabled for detection")

    def load_torchvision_model(self, initialize_full=True):
        """Load a pre-trained model from torchvision.

        Args:
            initialize_full: If True, initialize the full model for detection.
                            If False, just set class names for UI purposes.
        """
        try:
            logger.info(f"Loading torchvision model: {CONFIG['model_name']}")

            if initialize_full:
                self.model = detection.__dict__[CONFIG["model_name"]]()
                self.model.to(self.device)
                self.model.eval()
                logger.info(
                    f"Torchvision model {CONFIG['model_name']} loaded successfully"
                )

            self.class_names = COCO_CLASSES
            self.model_type = "torchvision"
            logger.info(f"Using {len(self.class_names)} COCO class names")
        except Exception as e:
            logger.error(f"Error loading torchvision model: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def load_custom_model(self, initialize_full=True):
        """Load a custom YOLO model using ultralytics library.

        Args:
            initialize_full: If True, initialize the full model for detection.
                            If False, just load class names for UI purposes.
        """
        model_path = CONFIG["custom_model_path"]

        logger.info(f"Loading custom YOLO model: {model_path}")

        if not os.path.exists(model_path):
            logger.error(f"Custom model file not found at {model_path}")
            logger.info("Falling back to torchvision model")
            self.load_torchvision_model(initialize_full)
            return

        try:
            # Use ultralytics YOLO for custom models
            if initialize_full:
                try:
                    # Import ultralytics YOLO
                    from ultralytics import YOLO

                    logger.info(f"Loading YOLO model from {model_path}")
                    self.model = YOLO(model_path)
                    self.model_type = "yolo"

                    # Get class names from the model
                    if hasattr(self.model, "names"):
                        self.class_names = list(self.model.names.values())
                        logger.info(
                            f"Using class names from YOLO model: {len(self.class_names)} classes"
                        )
                        logger.debug(f"Class names: {self.class_names[:10]}...")
                    else:
                        # Fall back to COCO classes if model doesn't have names
                        logger.warning(
                            "Model doesn't have class names attribute, using COCO classes"
                        )
                        self.class_names = COCO_CLASSES

                except ImportError:
                    logger.error(
                        "ultralytics not installed. Please install with: pip install ultralytics"
                    )
                    logger.info("Falling back to torchvision model")
                    self.load_torchvision_model(initialize_full)
                    return
                except Exception as e:
                    logger.error(f"Error loading YOLO model: {e}")
                    logger.error(traceback.format_exc())
                    logger.info("Falling back to torchvision model")
                    self.load_torchvision_model(initialize_full)
                    return
            else:
                # Just set the model type for UI purposes
                self.model_type = "yolo"
                logger.info(
                    "Initializing YOLO model for UI purposes only (not loading weights)"
                )

            # Load custom labels if provided and if we couldn't get them from the model
            if (
                (not hasattr(self, "class_names") or not self.class_names)
                and CONFIG["custom_model_labels_path"]
                and os.path.exists(CONFIG["custom_model_labels_path"])
            ):
                logger.info(
                    f"Loading custom labels from {CONFIG['custom_model_labels_path']}"
                )
                try:
                    with open(CONFIG["custom_model_labels_path"], "r") as f:
                        # Try to load as JSON
                        try:
                            labels_data = json.load(f)
                            logger.debug(
                                f"Loaded labels data as JSON: {type(labels_data)}"
                            )

                            if isinstance(labels_data, list):
                                self.class_names = labels_data
                                logger.debug(
                                    f"Using label list directly: {self.class_names[:10]}..."
                                )
                            elif isinstance(labels_data, dict):
                                # If it's a dict, we expect a key 'labels' or 'classes'
                                logger.debug(
                                    f"Label data is dict with keys: {labels_data.keys()}"
                                )
                                if "labels" in labels_data:
                                    self.class_names = labels_data["labels"]
                                    logger.debug("Using 'labels' key from dict")
                                elif "classes" in labels_data:
                                    self.class_names = labels_data["classes"]
                                    logger.debug("Using 'classes' key from dict")
                                elif "names" in labels_data:
                                    self.class_names = labels_data["names"]
                                    logger.debug("Using 'names' key from dict")
                                else:
                                    # Use dict keys if they're integers or try values
                                    try:
                                        # Try to convert keys to ints and sort
                                        logger.debug(
                                            "Attempting to convert dict keys to class indices"
                                        )
                                        id_to_name = {
                                            int(k): v
                                            for k, v in labels_data.items()
                                            if k.isdigit()
                                        }
                                        max_id = max(id_to_name.keys())
                                        logger.debug(
                                            f"Found {len(id_to_name)} class mappings, max index: {max_id}"
                                        )
                                        self.class_names = [
                                            id_to_name.get(i, f"class_{i}")
                                            for i in range(0, max_id + 1)
                                        ]
                                        logger.debug(
                                            f"Created class list: {self.class_names[:10]}..."
                                        )
                                    except (ValueError, AttributeError) as e:
                                        # If keys aren't all integers, use values
                                        logger.debug(
                                            f"Couldn't convert keys to indices: {e}"
                                        )
                                        logger.debug("Using dict values as class names")
                                        self.class_names = list(labels_data.values())
                        except json.JSONDecodeError:
                            # If not JSON, try to load as text file with one class per line
                            logger.debug(
                                "JSON parsing failed, trying to load as text file"
                            )
                            f.seek(0)  # Go back to start of file
                            lines = f.read().splitlines()
                            self.class_names = [
                                line.strip() for line in lines if line.strip()
                            ]
                            logger.debug(
                                f"Loaded {len(self.class_names)} classes from text file"
                            )
                except Exception as e:
                    logger.error(f"Error loading custom labels: {e}")
                    logger.error(traceback.format_exc())
                    logger.info("Using generic class names")
                    self.class_names = [
                        f"class_{i}" for i in range(100)
                    ]  # Default to generic class names
            elif not hasattr(self, "class_names") or not self.class_names:
                # If no labels file, use COCO classes as default
                logger.info("No custom labels file provided, using COCO classes")
                self.class_names = COCO_CLASSES

            if initialize_full:
                logger.info(f"Successfully loaded custom YOLO model from {model_path}")
            if hasattr(self, "class_names"):
                logger.info(f"Using {len(self.class_names)} class labels")
                logger.debug(f"First 10 class labels: {self.class_names[:10]}")

        except Exception as e:
            logger.error(f"Error loading custom model: {e}")
            logger.error(traceback.format_exc())
            logger.info("Falling back to torchvision model")
            self.load_torchvision_model(initialize_full)

    def detect_objects(self, frame):
        """
        Detect objects in the given frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detection dictionaries, each containing box, confidence, class_id, class_name
        """
        logger.debug("Starting object detection")
        results = []

        try:
            if self.model_type == "yolo":
                logger.debug("Running YOLO model inference")

                # YOLO models use the original image directly
                start_time = cv2.getTickCount()

                # Run detection with confidence threshold applied
                # Add verbose=False to suppress the print outputs
                yolo_results = self.model(
                    frame, conf=CONFIG["confidence_threshold"], verbose=False
                )

                end_time = cv2.getTickCount()
                inference_time = (end_time - start_time) / cv2.getTickFrequency()
                logger.debug(
                    f"YOLO inference completed in {inference_time:.4f} seconds"
                )

                # Process results
                for result in yolo_results:
                    boxes = result.boxes
                    logger.debug(f"YOLO found {len(boxes)} detections")

                    # Extract detections
                    for i, box in enumerate(boxes):
                        # Get box coordinates (YOLO returns xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Get confidence
                        confidence = float(box.conf[0].cpu().numpy())

                        # Get class ID
                        class_id = int(box.cls[0].cpu().numpy())

                        # Get class name
                        if class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                        else:
                            class_name = f"class_{class_id}"
                            logger.warning(
                                f"Class ID {class_id} is outside the range of class names (max={len(self.class_names) - 1})"
                            )

                        logger.debug(
                            f"Found object: class={class_name}, confidence={confidence:.4f}, box=[{x1},{y1},{x2},{y2}]"
                        )

                        # Filter by enabled objects if list is not empty
                        if (
                            CONFIG["enabled_objects"]
                            and class_name not in CONFIG["enabled_objects"]
                        ):
                            logger.debug(
                                f"Skipping {class_name} (not in enabled objects list)"
                            )
                            continue

                        results.append(
                            {
                                "box": [x1, y1, x2, y2],
                                "confidence": confidence,
                                "class_id": class_id,
                                "class_name": class_name,
                            }
                        )

            else:
                # For torchvision models, we need to preprocess the image (no change to this part)
                input_data = self.preprocess_image(frame)
                logger.debug("Image preprocessing completed for torchvision model")

                logger.debug(f"Running {self.model_type} model inference")
                # PyTorch models (torchvision)
                start_time = cv2.getTickCount()
                with torch.no_grad():
                    predictions = self.model(input_data)
                end_time = cv2.getTickCount()
                inference_time = (end_time - start_time) / cv2.getTickFrequency()
                logger.debug(f"Inference completed in {inference_time:.4f} seconds")

                # Log the structure of predictions for debugging
                logger.debug(f"Prediction type: {type(predictions)}")
                if isinstance(predictions, tuple) or isinstance(predictions, list):
                    logger.debug(
                        f"Prediction is a collection with {len(predictions)} elements"
                    )
                    for i, pred in enumerate(predictions):
                        logger.debug(f"Element {i} type: {type(pred)}")
                        if isinstance(pred, dict):
                            logger.debug(f"Element {i} keys: {pred.keys()}")
                elif isinstance(predictions, dict):
                    logger.debug(
                        f"Prediction is a dictionary with keys: {predictions.keys()}"
                    )

                # Extract results
                if isinstance(predictions, dict) or (
                    isinstance(predictions, list) and isinstance(predictions[0], dict)
                ):
                    logger.debug(
                        "Processing standard torchvision detection model output"
                    )
                    # Standard torchvision detection model output
                    if isinstance(predictions, list):
                        predictions = predictions[0]  # Use first batch item

                    boxes = predictions["boxes"].cpu().numpy()
                    scores = predictions["scores"].cpu().numpy()
                    labels = predictions["labels"].cpu().numpy()

                    logger.debug(f"Raw predictions: {len(scores)} potential detections")

                    # Filter by confidence threshold
                    mask = scores >= CONFIG["confidence_threshold"]
                    boxes = boxes[mask]
                    scores = scores[mask]
                    labels = labels[mask]

                    logger.debug(
                        f"After confidence filtering: {len(scores)} detections"
                    )

                    for box, score, label in zip(boxes, scores, labels):
                        class_id = int(label)

                        if class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                        else:
                            class_name = f"class_{class_id}"
                            logger.warning(
                                f"Class ID {class_id} is outside the range of class names (max={len(self.class_names) - 1})"
                            )

                        logger.debug(
                            f"Found object: class={class_name}, confidence={score:.4f}, box={box}"
                        )

                        # Filter by enabled objects if the list is not empty
                        if (
                            CONFIG["enabled_objects"]
                            and class_name not in CONFIG["enabled_objects"]
                        ):
                            logger.debug(
                                f"Skipping {class_name} (not in enabled objects list)"
                            )
                            continue

                        results.append(
                            {
                                "box": box.astype(int).tolist(),  # [x1, y1, x2, y2]
                                "confidence": float(score),
                                "class_id": class_id,
                                "class_name": class_name,
                            }
                        )
                else:
                    # Handle other model output formats
                    logger.error("Unsupported model output format")
                    logger.debug(f"Prediction type: {type(predictions)}")
                    if isinstance(predictions, torch.Tensor):
                        logger.debug(f"Tensor shape: {predictions.shape}")

            logger.debug(f"Detection completed, found {len(results)} objects")

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            logger.error(traceback.format_exc())

        return results

    def preprocess_image(self, frame):
        """Preprocess image for torchvision model input."""
        logger.debug("Preprocessing image for model input")
        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        logger.debug(f"Frame shape: {frame.shape}")

        logger.debug("Using PyTorch preprocessing pipeline")
        # Standard PyTorch preprocessing
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(rgb_frame.transpose(2, 0, 1)).float().div(255.0)
        logger.debug(
            f"Image tensor shape: {image_tensor.shape}, value range: {image_tensor.min()}-{image_tensor.max()}"
        )
        image_tensor = image_tensor.to(self.device)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        logger.debug(f"Final input batch shape: {image_tensor.shape}")

        return image_tensor

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
        logger.debug(f"Drawing {len(detections)} detections on frame")

        for current_detection in detections:
            class_name = current_detection["class_name"]
            x1, y1, x2, y2 = current_detection["box"]
            label = f"{class_name}: {current_detection['confidence']:.2f}"
            logger.debug(
                f"Drawing {class_name} at [{x1}, {y1}, {x2}, {y2}] with confidence {current_detection['confidence']:.2f}"
            )

            color = CONFIG["detection_colors"]["highlight"]

            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                result_frame,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0], y1),
                color,
                -1,
            )

            # Draw label text (black text on colored background)
            cv2.putText(
                result_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return result_frame
