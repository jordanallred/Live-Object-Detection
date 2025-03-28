"""
Video processing module for the home surveillance system.
"""

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
        self.display_frame_buffer = queue.Queue(maxsize=5)  # Buffer for frames ready to display
        
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
                
                detection_start = time.time()
                
                # Detect objects
                all_detections = self.detector.detect_objects(frame)
                
                # Apply detection backoff for each class
                current_time = time.time()
                filtered_detections = []
                
                for detection in all_detections:
                    object_class = detection['class_name']
                    # Check if this class is within backoff period
                    if object_class in self.last_detection_by_class:
                        last_time = self.last_detection_by_class[object_class]
                        backoff_seconds = CONFIG.get('detection_backoff', 0)
                        
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
                        'detections': all_detections,  # Store all for display purposes
                        'timestamp': current_time,
                        'frame': frame
                    }
                
                # If objects detected (after filtering), save image only
                if filtered_detections:
                    # Save detection image
                    image_with_boxes = self.detector.draw_detections(frame.copy(), filtered_detections)
                    self.save_detection_image(filtered_detections, image_with_boxes, current_time)
                
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
                print(f"Error in detection thread: {e}")
                import traceback
                traceback.print_exc()

    def open_camera(self):
        """Open the camera source or test video."""
        if self.camera is not None:
            self.camera.release()

        # Determine if using test video or live camera
        if CONFIG['use_test_video']:
            # Try to open the test video file
            video_path = CONFIG['test_video_path']
            if not os.path.exists(video_path):
                raise ValueError(f"Test video file not found: {video_path}")

            self.camera = cv2.VideoCapture(video_path)
            print(f"Using test video: {video_path}")
        else:
            # Try to open the camera
            self.camera = cv2.VideoCapture(CONFIG['camera_source'])
            print(f"Using camera source: {CONFIG['camera_source']}")

        if not self.camera.isOpened():
            raise ValueError(f"Could not open video source")

        # Get video properties
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default to 30 FPS if unable to determine

        # Store the total frame count for test videos
        self.total_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT)) if CONFIG['use_test_video'] else 0

        source_type = "Test video" if CONFIG['use_test_video'] else "Camera"
        print(f"{source_type} opened: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")

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
            
        timestamp = datetime.datetime.fromtimestamp(detection_time).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}"
        
        # Create paths
        image_path = os.path.join(CONFIG['save_dir'], 'images', f"{filename}.jpg")
        meta_path = os.path.join(CONFIG['save_dir'], 'images', f"{filename}.json")
        
        # Save the image
        cv2.imwrite(image_path, frame)
        
        # Create metadata
        metadata = {
            'timestamp': timestamp,
            'detections': detections,
            'image_path': f"images/{filename}.jpg",
            'id': timestamp
        }
        
        # No video path since we're only saving images
        metadata['video_path'] = None
        
        # Write metadata
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
            
        # Add to detection history
        with processing_lock:
            detection_history.insert(0, metadata)
            # Limit history size
            while len(detection_history) > CONFIG['history_limit']:
                detection_history.pop()
                
        return metadata
        
    # Video recording functionality has been removed

    # Video recording functionality has been removed

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
                if CONFIG.get('display_detection_boxes', True) and latest_detections is not None:
                    # Only use detections that aren't too old (within 2 detection intervals)
                    current_time = time.time()
                    max_age = CONFIG['detection_interval'] * 2
                    if current_time - latest_detections['timestamp'] <= max_age:
                        # Draw bounding boxes on the display frame
                        if latest_detections['detections']:
                            display_frame = self.detector.draw_detections(
                                display_frame, latest_detections['detections']
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
                print(f"Error in display thread: {e}")
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
        if current_time - self.last_detection_time >= CONFIG['detection_interval']:
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
                target=self.detection_worker, 
                name="DetectionThread",
                daemon=True
            )
            self.detection_thread.start()
            print("Started object detection thread")
            
            # Start display processing thread
            self.display_thread = threading.Thread(
                target=self.display_worker,
                name="DisplayThread",
                daemon=True
            )
            self.display_thread.start()
            print("Started display processing thread")

            # Initialize capture performance monitoring
            frame_count = 0
            start_time = time.time()
            fps_display_interval = 5  # seconds between FPS prints

            print(f"Starting capture loop with camera FPS: {self.fps}")
            while True:
                # Read frame from camera (this is now the only operation in the main thread loop)
                ret, frame = self.camera.read()
                
                # Update capture FPS counter
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= fps_display_interval:
                    self.fps_capture = frame_count / elapsed
                    print(f"Performance Metrics:")
                    print(f"  Capture FPS: {self.fps_capture:.2f}")
                    print(f"  Detection FPS: {self.fps_detection:.2f}")
                    print(f"  Display FPS: {self.fps_display:.2f}")
                    print(f"  Raw frame queue size: {self.raw_frame_buffer.qsize()}")
                    print(f"  Display frame queue size: {self.display_frame_buffer.qsize()}")
                    frame_count = 0
                    start_time = time.time()

                # Handle end of video for test videos
                if not ret:
                    if CONFIG['use_test_video']:
                        print("End of test video reached. Restarting...")
                        
                        # Restart camera if loop is enabled
                        if CONFIG['test_video_loop']:
                            self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            # Re-open camera (this will restart the video) 
                            self.open_camera()
                        continue
                    else:
                        print("Error reading frame. Reconnecting...")
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
                if CONFIG['use_test_video']:
                    target_delay = 1/max(self.fps, 1)  # Time per frame in seconds
                    time.sleep(target_delay)

        except Exception as e:
            print(f"Error in capture thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Stop all worker threads
            print("Stopping worker threads...")
            
            # Stop the detection thread if it's running
            if self.detection_thread and self.detection_thread.is_alive():
                print("Stopping detection thread...")
                self.stop_detection = True
                # Wait for the thread to finish (with timeout)
                self.detection_thread.join(timeout=2.0)
                print("Detection thread stopped" if not self.detection_thread.is_alive() 
                     else "Detection thread timeout - continuing shutdown")
                     
            # Stop the display thread if it's running
            if self.display_thread and self.display_thread.is_alive():
                print("Stopping display thread...")
                self.stop_display = True
                # Wait for the thread to finish (with timeout)
                self.display_thread.join(timeout=2.0)
                print("Display thread stopped" if not self.display_thread.is_alive() 
                     else "Display thread timeout - continuing shutdown")
                
            # Video recording cleanup has been removed

            # Release camera
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception as e:
                    print(f"Error releasing camera: {e}")

def generate_frames():
    """
    Generator function for streaming video frames.

    Yields:
        JPEG-encoded frame bytes for Flask's Response
    """
    print("Starting video stream generator")
    
    while True:
        # Get the frame from the global frame queue
        if not frame_queue.empty():
            try:
                frame = frame_queue.get()
                
                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    # Yield the frame in the format expected by Flask's Response
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                # Mark as done
                frame_queue.task_done()
            except Exception as e:
                print(f"Error in frame streaming: {e}")
        else:
            # If queue is empty, provide a short delay to prevent CPU spinning
            time.sleep(0.01)

# Initialize and start the video processor thread
def start_video_processor():
    """Initialize and start the video processor in a background thread."""
    print("Starting video processor...")
    video_processor = VideoProcessor()
    
    # Start the capture thread with a name for easier identification
    capture_thread = threading.Thread(
        target=video_processor.capture_and_process, 
        name="CaptureThread",
        daemon=True
    )
    capture_thread.start()
    print(f"Started capture thread: {capture_thread.name}")
    
    # Return the processor instance for reference
    return video_processor