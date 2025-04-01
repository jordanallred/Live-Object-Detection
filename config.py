"""
Configuration settings for the home surveillance system.
"""

import os


# Create required directories
def create_directories():
    for dir_path in [
        CONFIG["save_dir"],
        os.path.join(CONFIG["save_dir"], "images"),
    ]:
        os.makedirs(dir_path, exist_ok=True)


# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Configuration dictionary
CONFIG = {
    "camera_source": 0,  # Use 0 for webcam, or RTSP URL for IP camera
    "use_test_video": True,  # Set to True to use a pre-recorded video instead of camera
    "test_video_path": "test_video.mp4",  # Path to pre-recorded video for testing
    "test_video_loop": True,  # Whether to loop the test video
    "detection_interval": 1,  # Seconds between detection runs
    "confidence_threshold": 0.5,  # Minimum confidence score for detections
    "save_dir": "detections",  # Directory to save detection images
    "history_limit": 100,  # Maximum number of historical detections to keep
    # Model settings
    "model_name": "fasterrcnn_resnet50_fpn",  # Object detection model to use
    "use_custom_model": True,  # Whether to use a custom model file
    "custom_model_path": "models/yolo11n.pt",  # Path to custom model file
    "custom_model_labels_path": "",  # Path to labels file for custom model
    # Object detection settings
    "enabled_objects": [  # List of object classes to detect, empty list means detect all
        "car",
        "truck",
    ],
    "detection_colors": {  # Colors for different object types (BGR format)
        "default": (0, 255, 0),  # Green for most objects
        "highlight": (0, 0, 255),  # Red for all detected objects
    },
    "display_detection_boxes": False,  # Whether to show detection boxes on live feed
    "detection_backoff": 10,
    # Seconds to wait before triggering detection for the same object class
}

# COCO dataset class names
COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Create directories when module is imported
create_directories()
