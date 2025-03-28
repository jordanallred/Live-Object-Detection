"""
Configuration settings for the home surveillance system.
"""

import os

# Create required directories
def create_directories():
    for dir_path in [
        CONFIG['save_dir'],
        os.path.join(CONFIG['save_dir'], 'images'),
        os.path.join(CONFIG['save_dir'], 'clips'),
        'static',
        'static/css',
        'static/js',
        'templates',
        'templates/components'
    ]:
        os.makedirs(dir_path, exist_ok=True)

# Configuration dictionary
CONFIG = {
    'camera_source': 0,  # Use 0 for webcam, or RTSP URL for IP camera
    'use_test_video': True,  # Set to True to use a pre-recorded video instead of camera
    'test_video_path': 'test_video.mp4',  # Path to pre-recorded video for testing
    'test_video_loop': True,  # Whether to loop the test video
    'detection_interval': 1,  # Seconds between detection runs
    'confidence_threshold': 0.5,  # Minimum confidence score for detections
    'save_dir': 'detections',  # Directory to save detection clips and images
    'history_limit': 100,  # Maximum number of historical detections to keep
    'clip_duration': 5,  # Seconds of video to save when object detected
    'model_name': 'fasterrcnn_resnet50_fpn',  # Object detection model to use
}

# COCO dataset class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Create directories when module is imported
create_directories()