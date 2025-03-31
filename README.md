# Live Object Detection System

A modern, responsive home surveillance system with object detection capabilities and a user-friendly web interface.

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Latest-lightgrey.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-ee4c2c.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green.svg)

## 🚀 Features

- **Real-time Object Detection**: Leverages pre-trained models from torchvision or custom models
- **Multiple Video Sources**: Supports webcams, IP cameras, and pre-recorded test videos
- **Modern Web Interface**: Responsive design with real-time video feed, detection history, and configuration
- **Configurable Detection Settings**: Customize confidence thresholds, detection intervals, and object filters
- **Object Filtering**: Select specific objects to detect (e.g., only cars and people)
- **Detection History**: Browse through past detections with timestamps and object information
- **Custom Model Support**: Use your own [YOLO](https://docs.ultralytics.com/models/) models with custom label sets
- **Performance Optimization**: Configurable settings to balance detection quality and system resource usage

## 📋 Requirements

- Python 3.12 or later
- PyTorch and torchvision
- OpenCV
- Flask
- NumPy

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jordanallred/Live-Object-Detection.git
   cd Live-Object-Detection
   ```

2. Create a virtual environment and activate it:
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   uv pip install -r requirements.txt
   ```
   Note: On Windows, `requirements.txt` will install `triton-windows` intead of `triton` for PyTorch compatibility.



## 🚀 Quick Start

1. Run the application:
   ```bash
   python3 app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. The system will start with default settings using your webcam. You can configure video sources and detection settings through the web interface.

## 📖 Usage Guide

### Video Sources

- **Webcam**: Use your computer's built-in camera or an external USB webcam
- **Test Video**: Upload and use a pre-recorded video file for testing

### Object Detection Models

- **Built-in Models**: Choose from several pre-trained models with COCO dataset classes:
  - Faster R-CNN (ResNet-50 FPN)
  - Faster R-CNN (MobileNet V3)
  - RetinaNet
  - SSD300
  - SSDLite320

- **Custom Models**: Upload and use your own models:
  - Supported formats: Any model supported by YOLO models
  - Optional custom label mapping file (JSON or TXT)

### Detection Settings

- **Confidence Threshold**: Adjust minimum confidence score for detections (0.1-0.9)
- **Detection Interval**: Set time between detection runs (0.5-10 seconds)
- **Detection Backoff**: Prevent redundant detections of the same object class
- **Object Filters**: Select specific object types to detect
- **Display Detection Boxes**: Toggle visibility of bounding boxes on live feed

## 🔧 Configuration

The system uses a configuration dictionary in `config.py`. Most settings can be adjusted through the web interface, but you can also modify default values directly in the code:

```python
CONFIG = {
    'camera_source': 0,  # 0 for webcam, 1 for external camera, or RTSP URL
    'use_test_video': True,  # Set to True to use a pre-recorded video
    'test_video_path': 'test_video.mp4',  # Path to test video file
    'test_video_loop': True,  # Loop test video when reaching the end
    'detection_interval': 1,  # Seconds between detection runs
    'confidence_threshold': 0.5,  # Minimum confidence for detections (0.1-0.9)
    'save_dir': 'detections',  # Directory to save detection images
    'history_limit': 100,  # Maximum number of historical detections to keep
    'model_name': 'fasterrcnn_resnet50_fpn',  # Default torchvision model
    'use_custom_model': False,  # Set to True to use a custom model
    'custom_model_path': '',  # Path to custom model file
    'custom_model_labels_path': '',  # Path to custom model labels
    'enabled_objects': ['car', 'truck'],  # Objects to detect (empty for all)
    'display_detection_boxes': False,  # Show detection boxes on live feed
    'detection_backoff': 10,  # Seconds to wait before detecting same object class
}
```

## 🧩 Project Structure

```
Live-Object-Detection/
├── app.py                  # Main Flask application
├── config.py               # Configuration settings
├── pyproject.toml          # Project metadata
├── utils/
│   ├── __init__.py
│   ├── detector.py         # Object detection module
│   └── video.py            # Video processing module
├── static/
│   ├── css/
│   │   └── style.css       # Application styling
│   └── js/
│       └── main.js         # Frontend JavaScript
├── templates/
│   ├── index.html          # Main application page
│   └── layout.html         # Base HTML template
├── models/                 # Directory for custom models
└── detections/             # Directory for saved detections
    └── images/             # Saved detection images
```

## 🔍 API Endpoints

The system provides several API endpoints:

- `GET /` - Main web interface
- `GET /video_feed` - Live video stream
- `GET /history` - JSON array of detection history
- `GET /available_objects` - List of available object classes for detection
- `GET /detections/<path:filename>` - Access saved detection images
- `POST /upload_test_video` - Upload a test video file
- `POST /update_config` - Update system configuration
- `POST /upload_model` - Upload a custom model file
- `POST /upload_labels` - Upload a custom labels file

## 🧠 Object Detection

The system uses PyTorch's pre-trained detection models by default but can be configured to use custom models. All built-in models are trained on the COCO dataset, which can detect 80 different object classes including people, vehicles, animals, and common household items.

### Performance Considerations

- Detection is performed in a separate thread to avoid blocking the video stream
- Adjusting the detection interval affects CPU/GPU usage and responsiveness
- Lower confidence thresholds increase detection sensitivity but may introduce false positives
- Disabling bounding box display improves performance on low-end systems

## 📊 Future Enhancements

- **Motion Detection**: Pre-filter frames with motion before running object detection
- **Alert System**: Email or push notifications for specific detected objects
- **Multi-camera Support**: Manage and view multiple camera streams
- **Time-based Rules**: Configure different detection settings based on time of day
- **Zone-based Detection**: Define specific areas of interest for detection

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html) for detection models
- [OpenCV](https://opencv.org/) for video processing
- [Flask](https://flask.palletsprojects.com/) for the web server
- [Material Icons](https://fonts.google.com/icons) for UI elements

## ℹ️ Author

Jordan Allred - [GitHub Profile](https://github.com/jordanallred)

---

Feel free to contribute to this project by submitting pull requests or creating issues for bugs and feature requests.
