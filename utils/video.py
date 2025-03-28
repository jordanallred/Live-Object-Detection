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
frame_queue = queue.Queue(maxsize=30)
detection_history = []
current_frame = None
processing_lock = threading.Lock()

class VideoProcessor:
    """Handles video capture and processing."""

    def __init__(self):
        """Initialize the video processor."""
        self.camera = None
        self.last_detection_time = 0
        self.detector = ObjectDetector()
        self.recording_clips = []
        self.frame_buffer = {}  # Store frames for each recording

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

    def start_clip_recording(self, detections, frame):
        """
        Start recording a video clip when objects are detected.

        Args:
            detections: List of detection dictionaries
            frame: Current frame with detections
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}"

        # Create unique names for the detection files
        video_path = os.path.join(CONFIG['save_dir'], 'clips', f"{filename}.mp4")
        image_path = os.path.join(CONFIG['save_dir'], 'images', f"{filename}.jpg")
        meta_path = os.path.join(CONFIG['save_dir'], 'images', f"{filename}.json")

        # Save the detection frame as an image
        cv2.imwrite(image_path, frame)

        # Initialize frame buffer for this recording
        recording_id = timestamp
        self.frame_buffer[recording_id] = []
        
        # Calculate how many frames we need to record the full clip duration
        frames_needed = int(CONFIG['clip_duration'] * self.fps)
        
        # Store the first frame
        self.frame_buffer[recording_id].append(frame.copy())
        
        # Make sure we have enough pre-allocated space for all frames to maintain duration
        self.frame_buffer[recording_id] = [None] * frames_needed
        self.frame_buffer[recording_id][0] = frame.copy()  # Store the first frame

        print(f"Started recording: {video_path} - allocated buffer for {frames_needed} frames at {self.fps} FPS")

        # Save initial metadata
        metadata = {
            'timestamp': timestamp,
            'detections': detections,
            'video_path': f"clips/{filename}.mp4",  # Use MP4 format
            'image_path': f"images/{filename}.jpg",
            'id': recording_id
        }

        # Write initial metadata
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)

        # Add to recording clips list with end time
        end_time = time.time() + CONFIG['clip_duration']
        self.recording_clips.append({
            'end_time': end_time,
            'metadata': metadata,
            'id': recording_id,
            'video_path': video_path
        })

        # Add to detection history
        with processing_lock:
            detection_history.insert(0, metadata)
            # Limit history size
            while len(detection_history) > CONFIG['history_limit']:
                detection_history.pop()

    def update_recordings(self, frame):
        """
        Update active recordings and remove completed ones.

        Args:
            frame: Current frame to add to active recordings
        """
        current_time = time.time()
        completed = []

        # Add frame to active recordings
        for recording in self.recording_clips:
            recording_id = recording['id']
            if current_time <= recording['end_time']:
                # Store frame in buffer
                if recording_id in self.frame_buffer:
                    # Calculate how far we are into the recording
                    elapsed_time = current_time - (recording['end_time'] - CONFIG['clip_duration'])
                    # Calculate which frame index this should be based on elapsed time
                    frame_index = min(int(elapsed_time * self.fps), len(self.frame_buffer[recording_id]) - 1)
                    # Store frame at the appropriate position to maintain correct duration
                    if 0 <= frame_index < len(self.frame_buffer[recording_id]):
                        self.frame_buffer[recording_id][frame_index] = frame.copy()
            else:
                completed.append(recording)

        # Process and remove completed recordings
        for recording in completed:
            recording_id = recording['id']
            video_path = recording['video_path']

            # Check if we have frames to save
            if recording_id in self.frame_buffer and len(self.frame_buffer[recording_id]) > 0:
                frames = self.frame_buffer[recording_id]
                print(f"Saving {len(frames)} frames to {video_path}")

                # Create video writer using the MP4V codec as suggested
                try:
                    # Use MP4V codec which works on Stack Overflow examples
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(
                        video_path,
                        fourcc,
                        self.fps,  # Use native FPS instead of fixed value
                        (self.frame_width, self.frame_height)
                    )
                    
                    # Check if writer is initialized properly
                    if not writer.isOpened():
                        print(f"MP4V codec failed, trying alternative codec")
                        writer.release()
                        
                        # Try with X264 codec
                        fourcc = cv2.VideoWriter_fourcc(*'X264')
                        writer = cv2.VideoWriter(
                            video_path,
                            fourcc,
                            self.fps,  # Use native FPS
                            (self.frame_width, self.frame_height)
                        )
                        
                        # If that still doesn't work, try fallback to AVI
                        if not writer.isOpened():
                            writer.release()
                            print(f"Falling back to AVI format: {video_path}")
                            
                            # Change to AVI format for compatibility
                            video_path = video_path.replace('.mp4', '.avi')
                            
                            # Update metadata if needed
                            for rec in self.recording_clips:
                                if rec.get('video_path').endswith('.mp4'):
                                    rec['video_path'] = rec['video_path'].replace('.mp4', '.avi')
                                    if 'metadata' in rec:
                                        rec['metadata']['video_path'] = rec['metadata']['video_path'].replace('.mp4', '.avi')
                                        meta_file = os.path.join(CONFIG['save_dir'], 'images', f"{rec['id']}.json")
                                        if os.path.exists(meta_file):
                                            with open(meta_file, 'w') as f:
                                                json.dump(rec['metadata'], f)
                            
                            # Use XVID with AVI (widely compatible)
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            writer = cv2.VideoWriter(
                                video_path,
                                fourcc,
                                self.fps,  # Use native FPS
                                (self.frame_width, self.frame_height)
                            )
                    
                    # Write all frames at once
                    if writer.isOpened():
                        frame_count = 0
                        first_frame = None
                        # Find the first valid frame to use as a placeholder for any None frames
                        for f in frames:
                            if f is not None:
                                first_frame = f
                                break
                                
                        # If we have at least one valid frame, proceed with writing
                        if first_frame is not None:
                            for f in frames:
                                # If a frame is None (not captured), use the previous valid frame
                                if f is None:
                                    writer.write(first_frame)
                                else:
                                    writer.write(f)
                                    first_frame = f  # Update our placeholder frame
                                frame_count += 1
                                
                        print(f"Completed recording: {video_path} - Saved {frame_count} frames")
                    else:
                        print(f"Failed to create video writer for {video_path}")
                        
                    # Release writer
                    writer.release()
                except Exception as e:
                    print(f"Error writing video: {e}")

                # Clean up buffer
                del self.frame_buffer[recording_id]

            # Remove from active recordings
            self.recording_clips.remove(recording)

    def process_frame(self, frame):
        """
        Process a single frame with object detection.

        Args:
            frame: BGR image frame from the camera

        Returns:
            Processed frame with detections drawn
        """
        global current_frame

        # Store the original frame
        current_frame = frame.copy()

        # Run object detection at specified intervals
        current_time = time.time()
        if current_time - self.last_detection_time >= CONFIG['detection_interval']:
            self.last_detection_time = current_time

            # Detect objects
            detections = self.detector.detect_objects(frame)

            # Draw detections on frame
            if detections:
                frame = self.detector.draw_detections(frame, detections)

                # Start recording a clip if objects detected
                self.start_clip_recording(detections, frame)

        # Update any active recordings
        self.update_recordings(frame)

        return frame

    def capture_and_process(self):
        """Continuously capture and process frames from the camera or test video."""
        try:
            self.open_camera()

            while True:
                ret, frame = self.camera.read()

                # Handle end of video for test videos
                if not ret:
                    if CONFIG['use_test_video'] and CONFIG['test_video_loop']:
                        print("End of test video reached. Restarting...")
                        # Finish any active recordings
                        current_time = time.time()
                        for recording in list(self.recording_clips):  # Use a copy of the list
                            recording['end_time'] = current_time - 1  # Make it end immediately

                        # Process any completed recordings
                        self.update_recordings(None)

                        # Clear frame buffers
                        self.frame_buffer = {}
                        self.recording_clips = []

                        # Restart camera
                        self.open_camera()
                        continue
                    else:
                        print("Error reading frame. Reconnecting...")
                        time.sleep(1)
                        self.open_camera()
                        continue

                # Process the frame
                processed_frame = self.process_frame(frame)

                # Put the processed frame in the queue for the web server
                if not frame_queue.full():
                    frame_queue.put(processed_frame)

                # Add a slight delay when using test video to simulate real-time
                if CONFIG['use_test_video']:
                    time.sleep(1/max(self.fps, 1))  # Ensure we don't divide by zero

        except Exception as e:
            print(f"Error in capture thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure we properly clean up resources
            # Save any pending recordings
            for recording in list(self.recording_clips):  # Use a copy of the list
                recording_id = recording['id']
                video_path = recording['video_path']

                # Check if we have frames to save
                if recording_id in self.frame_buffer and len(self.frame_buffer[recording_id]) > 0:
                    frames = self.frame_buffer[recording_id]
                    print(f"Saving {len(frames)} frames to {video_path} during shutdown")

                    # Create video writer using the MP4V codec as suggested
                    try:
                        # Use MP4V codec which works on Stack Overflow examples
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(
                            video_path,
                            fourcc,
                            self.fps,  # Use native FPS
                            (self.frame_width, self.frame_height)
                        )
                        
                        # Check if writer is initialized properly
                        if not writer.isOpened():
                            print(f"MP4V codec failed during shutdown, trying alternative codec")
                            writer.release()
                            
                            # Try with X264 codec
                            fourcc = cv2.VideoWriter_fourcc(*'X264')
                            writer = cv2.VideoWriter(
                                video_path,
                                fourcc,
                                self.fps,  # Use native FPS
                                (self.frame_width, self.frame_height)
                            )
                            
                            # If that still doesn't work, try fallback to AVI
                            if not writer.isOpened():
                                writer.release()
                                print(f"Falling back to AVI format during shutdown: {video_path}")
                                
                                # Change to AVI format for compatibility
                                video_path = video_path.replace('.mp4', '.avi')
                                
                                # Update metadata if needed
                                for rec in self.recording_clips:
                                    if rec.get('video_path').endswith('.mp4'):
                                        rec['video_path'] = rec['video_path'].replace('.mp4', '.avi')
                                        if 'metadata' in rec:
                                            rec['metadata']['video_path'] = rec['metadata']['video_path'].replace('.mp4', '.avi')
                                            meta_file = os.path.join(CONFIG['save_dir'], 'images', f"{rec['id']}.json")
                                            if os.path.exists(meta_file):
                                                with open(meta_file, 'w') as f:
                                                    json.dump(rec['metadata'], f)
                                
                                # Use XVID with AVI (widely compatible)
                                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                writer = cv2.VideoWriter(
                                    video_path,
                                    fourcc,
                                    self.fps,  # Use native FPS
                                    (self.frame_width, self.frame_height)
                                )
                        
                        # Write all frames at once
                        if writer.isOpened():
                            frame_count = 0
                            first_frame = None
                            # Find the first valid frame to use as a placeholder for any None frames
                            for f in frames:
                                if f is not None:
                                    first_frame = f
                                    break
                                    
                            # If we have at least one valid frame, proceed with writing
                            if first_frame is not None:
                                for f in frames:
                                    # If a frame is None (not captured), use the previous valid frame
                                    if f is None:
                                        writer.write(first_frame)
                                    else:
                                        writer.write(f)
                                        first_frame = f  # Update our placeholder frame
                                    frame_count += 1
                                    
                            print(f"Completed shutdown recording: {video_path} - Saved {frame_count} frames")
                        else:
                            print(f"Failed to create video writer for {video_path}")
                            
                        # Release writer
                        writer.release()
                    except Exception as e:
                        print(f"Error writing video during shutdown: {e}")

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
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            # Yield the frame in the format expected by Flask's Response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # If queue is empty, short delay before checking again
            time.sleep(0.01)

# Initialize and start the video processor thread
def start_video_processor():
    """Initialize and start the video processor in a background thread."""
    video_processor = VideoProcessor()
    threading.Thread(target=video_processor.capture_and_process, daemon=True).start()
    return video_processor