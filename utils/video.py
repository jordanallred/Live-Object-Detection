"""
Video processing module for the home surveillance system.
"""

import logging

import cv2
import os
import time
import datetime
import json
import threading
import queue
import numpy as np
from config import CONFIG
from utils.detector import ObjectDetector

# Global variables
frame_queue = queue.Queue(maxsize=60)  # Increased buffer for streaming frames
detection_history = []
current_frame = None
latest_detections = None  # Store the most recent detection results
processing_lock = threading.Lock()

logger = logging.getLogger("video")


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class VideoProcessor:
    """Handles video capture and processing."""

    def __init__(self):
        """Initialize the video processor."""
        self.camera = None
        self.last_detection_time = 0
        self.detector = ObjectDetector()
        self.detection_thread = None
        self.display_thread = None
        self.stop_detection = False
        self.stop_display = False

        # Performance monitoring
        self.fps_capture = 0
        self.fps_detection = 0
        self.fps_display = 0

        # Detection backoff tracking
        self.last_detection_by_class = {}  # Dictionary to track last detection time by class

        # Enhanced frame buffers with separate capture and display streams
        self.raw_frame_buffer = queue.Queue(maxsize=5)  # Buffer for frames from camera
        self.display_frame_buffer = queue.Queue(
            maxsize=5
        )  # Buffer for frames ready to display

    def detection_worker(self):
        """Worker thread that processes frames for object detection separately."""
        global latest_detections

        # Initialize performance monitoring
        detection_count = 0
        start_time = time.time()

        while not self.stop_detection:
            try:
                # Get a frame from the raw frame buffer with timeout
                try:
                    frame = self.raw_frame_buffer.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Detect objects
                all_detections = self.detector.detect_objects(frame)

                # Apply detection backoff for each class
                current_time = time.time()
                filtered_detections = []

                for detection in all_detections:
                    object_class = detection["class_name"]
                    # Check if this class is within backoff period
                    if object_class in self.last_detection_by_class:
                        last_time = self.last_detection_by_class[object_class]
                        backoff_seconds = CONFIG.get("detection_backoff", 0)

                        # Skip this detection if it's too soon after last detection of same class
                        if current_time - last_time < backoff_seconds:
                            continue

                    # This detection passed the backoff check
                    filtered_detections.append(detection)
                    # Update the last detection time for this class
                    self.last_detection_by_class[object_class] = current_time

                # Store the latest detections with a timestamp (all detections, not just filtered)
                with processing_lock:
                    latest_detections = {
                        "detections": all_detections,  # Store all for display purposes
                        "timestamp": current_time,
                        "frame": frame,
                    }

                # If objects detected (after filtering), save image only
                if filtered_detections:
                    # Save detection image
                    image_with_boxes = self.detector.draw_detections(
                        frame.copy(), filtered_detections
                    )
                    self.save_detection_image(
                        filtered_detections, image_with_boxes, current_time
                    )

                # Update performance metrics
                detection_count += 1
                if detection_count % 10 == 0:  # Update FPS every 10 detections
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        self.fps_detection = detection_count / elapsed
                        if detection_count >= 100:  # Reset counter after 100 detections
                            detection_count = 0
                            start_time = time.time()

                # Mark task as done
                self.raw_frame_buffer.task_done()

            except Exception as e:
                logger.error(f"Error in detection thread: {e}")
                import traceback

                traceback.print_exc()

    def open_camera(self):
        """Open the camera source or test video with comprehensive codec support."""
        if self.camera is not None:
            self.camera.release()

        # Determine if using test video or live camera
        if CONFIG["use_test_video"]:
            # Try to open the test video file
            video_path = CONFIG["test_video_path"]
            if not os.path.exists(video_path):
                raise ValueError(f"Test video file not found: {video_path}")

            # First attempt - try standard opening
            logger.info(f"Opening test video: {video_path}")
            self.camera = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

            # Check if camera opened successfully and can read a frame
            success = self.camera.isOpened()
            if success:
                # Try to read a frame to verify codec works
                ret, test_frame = self.camera.read()
                if not ret:
                    logger.warning(
                        "Camera opened but couldn't read frame - potential codec issue"
                    )
                    success = False
                    # Reset position to beginning
                    self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # If standard opening failed, try codec-specific solutions
            if not success:
                logger.warning(
                    "Failed to open video normally, attempting with alternative methods"
                )
                self.camera.release()

                # Try to transcode the video to a more compatible format
                try:
                    import subprocess
                    from shutil import which

                    # Check if ffmpeg is available
                    if which("ffmpeg") is None:
                        logger.error("ffmpeg not found in PATH, cannot transcode video")
                        raise FileNotFoundError("ffmpeg command not found")

                    # Create a temporary file for the transcoded video
                    import tempfile

                    temp_dir = tempfile.gettempdir()
                    temp_output = os.path.join(temp_dir, "transcoded_video.mp4")

                    logger.info(f"Transcoding video to {temp_output}")

                    # Transcode to H.264 which is widely supported
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_path,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-pix_fmt",
                        "yuv420p",  # Ensure pixel format compatibility
                        "-g",
                        "30",  # Keyframe every 30 frames for better seeking
                        "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure dimensions are even
                        "-an",  # No audio
                        temp_output,
                    ]

                    logger.info(f"Running: {' '.join(cmd)}")
                    process = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )

                    # Log output for debugging
                    if process.returncode != 0:
                        logger.error(f"Transcoding failed: {process.stderr}")
                    else:
                        logger.info("Transcoding completed successfully")

                        # Try opening the transcoded file
                        self.camera = cv2.VideoCapture(temp_output, cv2.CAP_FFMPEG)
                        if not self.camera.isOpened():
                            logger.error("Failed to open transcoded video")
                        else:
                            logger.info("Successfully opened transcoded video")

                            # Try reading a frame to confirm it works
                            ret, frame = self.camera.read()
                            if not ret:
                                logger.error("Cannot read from transcoded video")
                                self.camera.release()
                            else:
                                # Success! Use the transcoded video
                                logger.info("Transcoded video working correctly")
                                success = True

                except Exception as e:
                    logger.error(f"Error during video transcoding: {e}")
                    import traceback

                    logger.error(traceback.format_exc())

                # If transcoding failed, try one more fallback method
                if not success or not self.camera.isOpened():
                    logger.warning("Trying one more method with explicit MJPG decoder")
                    self.camera = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                    if self.camera.isOpened():
                        # Try forcing a different decoder
                        self.camera.set(
                            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
                        )
                        self.camera.set(
                            cv2.CAP_PROP_BUFFERSIZE, 10
                        )  # Increase buffer size
        else:
            # Handle IP camera or webcam connections
            if isinstance(CONFIG["camera_source"], str) and (
                CONFIG["camera_source"].startswith("rtsp")
                or CONFIG["camera_source"].startswith("http")
            ):
                logger.info(f"Connecting to IP camera: {CONFIG['camera_source']}")
                # For IP cameras, use FFMPEG backend with tuned parameters
                self.camera = cv2.VideoCapture(CONFIG["camera_source"], cv2.CAP_FFMPEG)
                self.camera.set(
                    cv2.CAP_PROP_BUFFERSIZE, 5
                )  # Larger buffer for network streams

                # Add a longer timeout for network cameras
                connect_timeout = time.time() + 15  # 15-second timeout
                while not self.camera.isOpened() and time.time() < connect_timeout:
                    logger.info("Waiting for IP camera connection...")
                    time.sleep(1)
                    self.camera = cv2.VideoCapture(
                        CONFIG["camera_source"], cv2.CAP_FFMPEG
                    )
            else:
                # Regular webcam
                logger.info(f"Using camera source: {CONFIG['camera_source']}")
                self.camera = cv2.VideoCapture(CONFIG["camera_source"])

        # Create fallback frame generator if we can't open the video source
        if not self.camera.isOpened():
            logger.error("Could not open any video source, using fallback mode")

            # Create a black frame generator for placeholder
            self._use_placeholder = True
            self._placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                self._placeholder_frame,
                "No video available",
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Set default properties
            self.frame_width, self.frame_height = 640, 480
            self.fps = 10
            self.total_frames = 0
        else:
            self._use_placeholder = False

            # Get video properties
            self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.camera.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0 or self.fps > 60:  # Cap unreasonable FPS values
                self.fps = 30

            # Store total frames for test videos
            self.total_frames = (
                int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
                if CONFIG["use_test_video"]
                else 0
            )

            source_type = "Test video" if CONFIG["use_test_video"] else "Camera"
            logger.info(
                f"{source_type} opened: {self.frame_width} x {self.frame_height} @ {self.fps} FPS"
            )

    def save_detection_image(self, detections, frame, detection_time=None):
        """
        Save a detection image with bounding boxes.

        Args:
            detections: List of detection dictionaries
            frame: Image with detection boxes drawn
            detection_time: Optional timestamp of detection (defaults to now)

        Returns:
            Dictionary with metadata about saved image
        """
        # Generate timestamp
        if detection_time is None:
            detection_time = time.time()

        timestamp = datetime.datetime.fromtimestamp(detection_time).strftime(
            "%Y%m%d_%H%M%S"
        )
        filename = f"{timestamp}"

        # Create paths
        image_path = os.path.join(CONFIG["save_dir"], "images", f"{filename}.jpg")
        meta_path = os.path.join(CONFIG["save_dir"], "images", f"{filename}.json")

        # Ensure directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Save the image
        cv2.imwrite(image_path, frame)

        # Create metadata
        metadata = {
            "timestamp": timestamp,
            "detections": detections,
            "image_path": f"images/{filename}.jpg",
            "id": timestamp,
        }

        # No video path since we're only saving images
        metadata["video_path"] = None

        # Write metadata using the custom JSON encoder
        with open(meta_path, "w") as f:
            json.dump(metadata, f, cls=NumpyJSONEncoder)

        # Add to detection history
        with processing_lock:
            detection_history.insert(0, metadata)
            # Limit history size
            while len(detection_history) > CONFIG["history_limit"]:
                detection_history.pop()

        return metadata

    def display_worker(self):
        """Worker thread that processes frames for display with detection overlays."""
        global current_frame, latest_detections

        # Initialize performance monitoring
        display_count = 0
        start_time = time.time()

        while not self.stop_display:
            try:
                # Get a frame from the raw frame buffer with short timeout
                try:
                    frame = self.raw_frame_buffer.get(timeout=0.01)
                except queue.Empty:
                    # If no new frames, sleep briefly to avoid CPU spinning
                    time.sleep(0.001)
                    continue

                # Store the original frame for global access
                current_frame = frame.copy()

                # Create a working copy for display with detections
                display_frame = frame.copy()

                # Draw latest detections on the frame if enabled and available
                if (
                    CONFIG.get("display_detection_boxes", True)
                    and latest_detections is not None
                ):
                    # Only use detections that aren't too old (within 2 detection intervals)
                    current_time = time.time()
                    max_age = CONFIG["detection_interval"] * 2
                    if current_time - latest_detections["timestamp"] <= max_age:
                        # Draw bounding boxes on the display frame
                        if latest_detections["detections"]:
                            display_frame = self.detector.draw_detections(
                                display_frame, latest_detections["detections"]
                            )

                # Put processed frame in display buffer for web streaming
                if not self.display_frame_buffer.full():
                    self.display_frame_buffer.put(display_frame.copy())

                # Always update the global frame queue for web streaming
                while frame_queue.full():
                    # If queue is full, remove oldest frame
                    try:
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                    except queue.Empty:
                        break

                # Add the current frame to the queue
                frame_queue.put(display_frame.copy())

                # Update performance metrics
                display_count += 1
                if display_count % 30 == 0:  # Update FPS every 30 frames
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        self.fps_display = display_count / elapsed
                        if display_count >= 300:  # Reset counter after 300 frames
                            display_count = 0
                            start_time = time.time()

                # Mark task as done
                self.raw_frame_buffer.task_done()

            except Exception as e:
                logger.error(f"Error in display thread: {e}")
                import traceback

                traceback.print_exc()

    def process_frame(self, frame):
        """
        Process a single frame with object detection.
        This method is now primarily used for scheduling detection
        at specified intervals.

        Args:
            frame: BGR image frame from the camera

        Returns:
            Original frame (no longer needed but kept for compatibility)
        """
        # Sample frames for detection at specified intervals
        current_time = time.time()
        if current_time - self.last_detection_time >= CONFIG["detection_interval"]:
            self.last_detection_time = current_time

            # Queue frame for asynchronous detection (non-blocking)
            if not self.raw_frame_buffer.full():
                self.raw_frame_buffer.put(frame.copy())

        return frame  # Return original frame (mostly for compatibility)

    def capture_and_process(self):
        """Continuously capture and process frames from the camera or test video."""
        try:
            self.open_camera()

            # Clear any old data in queues
            while not self.raw_frame_buffer.empty():
                try:
                    self.raw_frame_buffer.get_nowait()
                except queue.Empty:
                    break

            while not self.display_frame_buffer.empty():
                try:
                    self.display_frame_buffer.get_nowait()
                except queue.Empty:
                    break

            # Start the worker threads
            self.stop_detection = False
            self.stop_display = False

            # Start detection thread
            self.detection_thread = threading.Thread(
                target=self.detection_worker, name="DetectionThread", daemon=True
            )
            self.detection_thread.start()
            logger.info("Started object detection thread...")

            # Start display processing thread
            self.display_thread = threading.Thread(
                target=self.display_worker, name="DisplayThread", daemon=True
            )
            self.display_thread.start()
            logger.info("Started display processing thread...")

            # Initialize capture performance monitoring
            frame_count = 0
            start_time = time.time()
            fps_display_interval = 5  # seconds between FPS prints

            logger.debug(f"Starting capture loop with camera FPS: {self.fps}")
            while True:
                # Read frame from camera (this is now the only operation in the main thread loop)
                ret, frame = self.camera.read()

                # Update capture FPS counter
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= fps_display_interval:
                    self.fps_capture = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()

                # Handle end of video for test videos
                if not ret:
                    if CONFIG["use_test_video"]:
                        # Restart camera if loop is enabled
                        if CONFIG["test_video_loop"]:
                            self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            # Re-open camera (this will restart the video)
                            self.open_camera()
                        continue
                    else:
                        logger.error("Error reading frame. Reconnecting...")
                        time.sleep(1)
                        self.open_camera()
                        continue

                # Always feed the raw frame directly to the raw frame buffer
                # to ensure worker threads have access to latest frames
                if not self.raw_frame_buffer.full():
                    self.raw_frame_buffer.put(frame.copy())
                else:
                    # If buffer is full, clear oldest frame
                    try:
                        self.raw_frame_buffer.get_nowait()
                        self.raw_frame_buffer.task_done()
                        # Now add the new frame
                        self.raw_frame_buffer.put(frame.copy())
                    except queue.Empty:
                        pass

                # Also process frame for recordings
                self.process_frame(frame)

                # For compatibility and immediate display, also feed directly to global frame queue
                if not frame_queue.full():
                    frame_queue.put(frame.copy())

                # For test videos, respect the original fps to maintain proper playback speed
                if CONFIG["use_test_video"]:
                    target_delay = 1 / max(self.fps, 1)  # Time per frame in seconds
                    time.sleep(target_delay)

        except Exception as e:
            logger.error(f"Error in capture thread: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Stop all worker threads
            logger.info("Stopping worker threads...")

            # Stop the detection thread if it's running
            if self.detection_thread and self.detection_thread.is_alive():
                logger.info("Stopping detection thread...")
                self.stop_detection = True
                # Wait for the thread to finish (with timeout)
                self.detection_thread.join(timeout=2.0)
                logger.debug(
                    "Detection thread stopped"
                    if not self.detection_thread.is_alive()
                    else "Detection thread timeout - continuing shutdown"
                )

            # Stop the display thread if it's running
            if self.display_thread and self.display_thread.is_alive():
                logger.info("Stopping display thread...")
                self.stop_display = True
                # Wait for the thread to finish (with timeout)
                self.display_thread.join(timeout=2.0)
                logger.error(
                    "Display thread stopped"
                    if not self.display_thread.is_alive()
                    else "Display thread timeout - continuing shutdown"
                )

            # Video recording cleanup has been removed

            # Release camera
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")


def generate_frames():
    """
    Generator function for streaming video frames.

    Yields:
        JPEG-encoded frame bytes for Flask's Response
    """
    logger.info("Starting video stream generator")

    while True:
        # Get the frame from the global frame queue
        if not frame_queue.empty():
            try:
                frame = frame_queue.get()

                # Encode the frame as JPEG
                ret, buffer = cv2.imencode(".jpg", frame)
                if ret:
                    # Yield the frame in the format expected by Flask's Response
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )

                # Mark as done
                frame_queue.task_done()
            except Exception as e:
                logger.error(f"Error in frame streaming: {e}")
        else:
            # If queue is empty, provide a short delay to prevent CPU spinning
            time.sleep(0.01)


# Initialize and start the video processor thread
def start_video_processor():
    """Initialize and start the video processor in a background thread."""
    logger.info("Starting video processor...")
    video_processor = VideoProcessor()

    # Start the capture thread with a name for easier identification
    capture_thread = threading.Thread(
        target=video_processor.capture_and_process, name="CaptureThread", daemon=True
    )
    capture_thread.start()
    logger.debug(f"Started capture thread: {capture_thread.name}")

    # Return the processor instance for reference
    return video_processor
