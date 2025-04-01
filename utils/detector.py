"""
Object detection module for the home surveillance system.
"""

import os
import cv2
import torch
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
        logger.info("Initializing object detector...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device: {self.device}")

        # Load class labels based on model type
        if CONFIG["use_custom_model"]:
            logger.debug("Using custom model configuration")
            self.load_custom_model(initialize_full)
        else:
            logger.debug("Using built-in torchvision model")
            self.load_torchvision_model(initialize_full)

        if initialize_full:
            logger.debug(f"Model loaded successfully on {self.device}")
            if hasattr(self, "class_names"):
                logger.info(f"Available classes: {len(self.class_names)}")
                logger.debug(f"Class names: {self.class_names[:20]}...")

            # Log enabled objects configuration
            if CONFIG["enabled_objects"]:
                logger.info(f"Enabled objects: {CONFIG['enabled_objects']}")
            else:
                logger.info("All objects enabled for detection")

    def load_torchvision_model(self, initialize_full=True):
        """Load a pre-trained model from torchvision with proper class names extraction."""
        try:
            model_name = CONFIG["model_name"]
            logger.debug(f"Loading torchvision model: {model_name}")

            # Map model names to their respective weights classes
            weights_map = {
                "fasterrcnn_resnet50_fpn": detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                "fasterrcnn_mobilenet_v3_large_fpn": detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                "retinanet_resnet50_fpn": detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT,
                "ssd300_vgg16": detection.SSD300_VGG16_Weights.DEFAULT,
                "ssdlite320_mobilenet_v3_large": detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
            }

            if model_name not in weights_map:
                logger.warning(
                    f"Unknown model name: {model_name}, defaulting to fasterrcnn_resnet50_fpn"
                )
                model_name = "fasterrcnn_resnet50_fpn"

            # Get the weights for this model
            weights = weights_map[model_name]

            # Try to extract class names from weights metadata
            self.class_names = ["__background__"]  # Start with background class

            try:
                if hasattr(weights, "meta"):
                    logger.debug(f"Checking weights metadata: {weights.meta.keys()}")

                    # Different models might store categories differently
                    if "categories" in weights.meta:
                        categories = weights.meta["categories"]

                        if isinstance(categories, list):
                            self.class_names.extend(categories)
                            logger.info(
                                f"Added {len(categories)} classes from weights metadata list"
                            )
                        elif isinstance(categories, dict):
                            # Sort by key if keys are numeric
                            sorted_items = sorted(
                                categories.items(), key=lambda x: int(x[0])
                            )
                            self.class_names.extend([item[1] for item in sorted_items])
                            logger.info(
                                f"Added {len(sorted_items)} classes from sorted dict"
                            )

                    # For SSD models which might use different metadata format
                    elif "class_labels" in weights.meta:
                        class_labels = weights.meta["class_labels"]
                        self.class_names.extend(class_labels)
                        logger.info(
                            f"Added {len(class_labels)} classes from class_labels metadata"
                        )

                    # Look for COCO_CATEGORIES if model uses that
                    elif "COCO_CATEGORIES" in dir(weights.meta):
                        coco_cats = weights.meta.COCO_CATEGORIES
                        self.class_names.extend(coco_cats)
                        logger.info(
                            f"Added {len(coco_cats)} classes from COCO_CATEGORIES"
                        )

                    # Try to import the specific weights module to get categories
                    else:
                        logger.info("Looking for model-specific category definitions")
                        if model_name == "ssd300_vgg16":
                            # For SSD300, try to import the specific label mapping
                            from torchvision.models.detection.ssd import (
                                _COCO_CATEGORIES,
                            )

                            self.class_names = ["__background__"] + list(
                                _COCO_CATEGORIES
                            )
                            logger.info(
                                f"Loaded {len(_COCO_CATEGORIES)} SSD-specific classes"
                            )

                        elif "category_mapping" in weights.meta:
                            # Some models might have this
                            self.class_names = ["__background__"] + list(
                                weights.meta.category_mapping.values()
                            )
                            logger.info("Loaded classes from category_mapping")

                # If we still don't have class names or only background
                if len(self.class_names) <= 1:
                    logger.warning(
                        "Could not extract class names from weights metadata, using COCO classes"
                    )
                    self.class_names = COCO_CLASSES

                logger.debug(
                    f"Final class list contains {len(self.class_names)} classes"
                )
                logger.debug(f"First 10 classes: {self.class_names[:10]}")

            except Exception as e:
                logger.error(f"Error extracting class names: {e}")
                logger.error(traceback.format_exc())
                self.class_names = COCO_CLASSES
                logger.warning("Using default COCO classes as fallback")

            # Initialize the model if needed
            if initialize_full:
                logger.info(f"Initializing {model_name} with pretrained weights")

                # Create model with weights
                if model_name == "fasterrcnn_resnet50_fpn":
                    self.model = detection.fasterrcnn_resnet50_fpn(weights=weights)
                elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
                    self.model = detection.fasterrcnn_mobilenet_v3_large_fpn(
                        weights=weights
                    )
                elif model_name == "retinanet_resnet50_fpn":
                    self.model = detection.retinanet_resnet50_fpn(weights=weights)
                elif model_name == "ssd300_vgg16":
                    self.model = detection.ssd300_vgg16(weights=weights)
                elif model_name == "ssdlite320_mobilenet_v3_large":
                    self.model = detection.ssdlite320_mobilenet_v3_large(
                        weights=weights
                    )

                # Move to device and set to eval mode
                self.model.to(self.device)
                self.model.eval()
                logger.info("Model initialized and set to evaluation mode")

            self.model_type = "torchvision"

        except Exception as e:
            logger.error(f"Error loading torchvision model: {str(e)}")
            logger.error(traceback.format_exc())
            self.class_names = COCO_CLASSES
            self.model_type = "torchvision"
            raise

    def load_custom_model(self, initialize_full=True):
        """Load a custom YOLO model using ultralytics library with improved class name handling."""
        model_path = CONFIG["custom_model_path"]

        if not os.path.exists(model_path):
            logger.error(f"Custom model file not found at {model_path}")
            logger.info("Falling back to torchvision model")
            self.load_torchvision_model(initialize_full)
            return

        try:
            # Set model type for reference
            self.model_type = "yolo"

            # Initialize class_names to None - we'll fill it in order of priority
            self.class_names = None

            # PRIORITY 1: Load the full model and get class names directly from it
            if initialize_full:
                try:
                    from ultralytics import YOLO

                    logger.info(f"Loading YOLO model from {model_path}")
                    self.model = YOLO(model_path)

                    # Get class names from the model's names attribute
                    if hasattr(self.model, "names") and self.model.names:
                        # Check if names is a dict with int keys (typical YOLO format)
                        if isinstance(self.model.names, dict) and all(
                            isinstance(k, int) for k in self.model.names.keys()
                        ):
                            # Convert dict to list ensuring proper index order
                            max_idx = max(self.model.names.keys())
                            class_list = ["unknown"] * (
                                max_idx + 1
                            )  # Initialize with placeholders
                            for idx, name in self.model.names.items():
                                class_list[idx] = name
                            self.class_names = class_list
                        elif isinstance(self.model.names, dict):
                            # If keys aren't all integers, just use values
                            self.class_names = list(self.model.names.values())
                        else:
                            # If it's already a list
                            self.class_names = self.model.names

                        logger.debug(
                            f"Using class names from YOLO model: {len(self.class_names)} classes"
                        )
                        logger.debug(
                            f"Class names from model: {self.class_names[:10]}..."
                        )
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
                    # Continue to try other methods of getting class names

            # PRIORITY 2: Try to load from custom labels file
            if (self.class_names is None or len(self.class_names) == 0) and CONFIG[
                "custom_model_labels_path"
            ]:
                labels_path = CONFIG["custom_model_labels_path"]
                if os.path.exists(labels_path):
                    logger.info(f"Loading custom labels from {labels_path}")
                    try:
                        self.class_names = self._parse_labels_file(labels_path)
                        if self.class_names:
                            logger.info(
                                f"Successfully loaded {len(self.class_names)} classes from {labels_path}"
                            )
                            logger.debug(f"First 10 classes: {self.class_names[:10]}")
                    except Exception as e:
                        logger.error(f"Error loading custom labels file: {e}")
                        self.class_names = None

            # PRIORITY 3: Look for a labels.txt file in the same directory as the model
            if self.class_names is None or len(self.class_names) == 0:
                model_dir = os.path.dirname(model_path)
                for label_filename in [
                    "labels.txt",
                    "classes.txt",
                    "names.txt",
                    "coco.names",
                ]:
                    potential_path = os.path.join(model_dir, label_filename)
                    if os.path.exists(potential_path):
                        logger.info(f"Found potential labels file at {potential_path}")
                        try:
                            self.class_names = self._parse_labels_file(potential_path)
                            if self.class_names:
                                logger.info(
                                    f"Successfully loaded {len(self.class_names)} classes from {potential_path}"
                                )
                                break
                        except Exception as e:
                            logger.error(f"Error loading potential labels file: {e}")
                            continue

            # PRIORITY 4: Look for a *.yaml file in the same directory (YOLO format)
            if self.class_names is None or len(self.class_names) == 0:
                model_dir = os.path.dirname(model_path)
                yaml_files = [f for f in os.listdir(model_dir) if f.endswith(".yaml")]
                for yaml_file in yaml_files:
                    yaml_path = os.path.join(model_dir, yaml_file)
                    try:
                        import yaml

                        with open(yaml_path, "r") as f:
                            yaml_data = yaml.safe_load(f)
                            if isinstance(yaml_data, dict) and "names" in yaml_data:
                                if isinstance(yaml_data["names"], list):
                                    self.class_names = yaml_data["names"]
                                elif isinstance(yaml_data["names"], dict):
                                    # Convert dict to ordered list
                                    max_idx = max(
                                        int(k) for k in yaml_data["names"].keys()
                                    )
                                    class_list = ["unknown"] * (max_idx + 1)
                                    for idx, name in yaml_data["names"].items():
                                        class_list[int(idx)] = name
                                    self.class_names = class_list
                                logger.info(
                                    f"Loaded {len(self.class_names)} classes from YAML file {yaml_path}"
                                )
                                break
                    except Exception as e:
                        logger.error(f"Error parsing YAML file {yaml_path}: {e}")
                        continue

            # FINAL FALLBACK: Use COCO classes if nothing else worked
            if self.class_names is None or len(self.class_names) == 0:
                logger.warning(
                    "Could not load custom class names from any source, using COCO classes"
                )
                self.class_names = COCO_CLASSES

            logger.debug(f"Final class list contains {len(self.class_names)} classes")

        except Exception as e:
            logger.error(f"Error in custom model loading: {e}")
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
