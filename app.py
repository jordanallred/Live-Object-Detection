"""
Main application file for the home surveillance system.
"""
import json
import logging
import os
import traceback

from flask import Flask, Response, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from utils.video import generate_frames, start_video_processor
from config import CONFIG
from utils.video import NumpyJSONEncoder

# After initializing the Flask app, configure it to use your custom encoder for all JSON responses

# Initialize Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
app.config['UPLOAD_FOLDER'] = '.'  # Current directory for uploads
app.config['DETECTIONS_DIR'] = os.path.join(os.getcwd(),
                                            CONFIG['save_dir'])  # Absolute path to detections
app.json_encoder = NumpyJSONEncoder

# Global reference to the video processor for restart handling
video_processor = None


@app.route('/')
def index():
    """Route for the main page."""
    return render_template('index.html',
                           use_test_video=CONFIG['use_test_video'],
                           test_video_path=CONFIG['test_video_path'] if CONFIG[
                               'use_test_video'] else "",
                           config=CONFIG)


@app.route('/video_feed')
def video_feed():
    """Route for streaming the processed video feed."""
    print("Video feed route accessed")
    # Use a response object for streaming with proper encoding
    response = Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

    # Disable caching for this stream
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response


@app.route('/history')
def get_history():
    """Get detection history."""
    try:
        # Import logger
        logger = logging.getLogger("detector")
        logger.info("Loading detection history")

        # Get history from the global detection_history list
        from utils.video import detection_history, NumpyJSONEncoder

        # Log the first entry to help with debugging
        if detection_history and len(detection_history) > 0:
            logger.debug(f"First history entry: {detection_history[0]}")

        # Use the custom encoder to convert NumPy types to Python native types
        return Response(
            json.dumps(detection_history, cls=NumpyJSONEncoder),
            mimetype='application/json'
        )
    except Exception as e:
        # Log the error
        logger.error(f"Error loading detection history: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify([])  # Return empty list on error


@app.route('/available_objects')
def available_objects():
    """Route to get available objects for detection based on the current model."""
    from utils.detector import ObjectDetector

    try:
        # Create a temporary detector to get class names
        # Pass initialize_full=False to avoid loading the full model
        # This way we just get the class names without the computational overhead
        temp_detector = ObjectDetector(initialize_full=False)

        # Get the class names (removing background if present)
        class_names = temp_detector.class_names
        available_objects = [name for name in class_names if name != '__background__']

        # Get currently enabled objects from config
        enabled_objects = CONFIG['enabled_objects']

        return jsonify({
            'success': True,
            'objects': available_objects,
            'enabled_objects': enabled_objects
        })
    except Exception as e:
        print(f"Error getting available objects: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/detections/<path:filename>')
def serve_detection_file(filename):
    """Route to serve saved detection images and videos."""
    # Get the absolute path to the detections directory
    base_dir = app.config['DETECTIONS_DIR']

    # Handle subdirectories like images/ and clips/
    if '/' in filename:
        # Split the path into directory and actual filename
        sub_dir, file = filename.split('/', 1)
        file_path = os.path.join(base_dir, sub_dir, file)
    else:
        file_path = os.path.join(base_dir, filename)

    # Check if file exists, log if not
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print(f"Base directory: {base_dir}")
        if '/' in filename:
            sub_dir = filename.split('/', 1)[0]
            sub_dir_path = os.path.join(base_dir, sub_dir)
            if os.path.exists(sub_dir_path):
                print(f"Contents of {sub_dir_path}: {os.listdir(sub_dir_path)}")
        return f"File {filename} not found", 404

    # Determine mime type - only jpg images now
    mimetype = 'image/jpeg' if filename.endswith('.jpg') else None

    # Use send_file instead which works better with dynamically generated content
    response = send_file(file_path, mimetype=mimetype)

    return response


@app.route('/upload_test_video', methods=['POST'])
def upload_test_video():
    """Route to handle test video uploads."""
    if 'video_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['video_file']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file:
        # Make sure the filename is secure
        filename = secure_filename(file.filename)

        # Save the file
        file.save(filename)

        return jsonify({
            'success': True,
            'filename': filename
        })

    return jsonify({'success': False, 'error': 'Unknown error'})


@app.route('/update_config', methods=['POST'])
def update_config():
    """Route to update system configuration."""
    global video_processor

    if request.method == 'POST':
        # Config changed flag to track if we need to restart the video processor
        config_changed = False
        restart_required = False
        form_id = request.form.get('form_id', '')

        # Check which form was submitted by looking at form_id
        if form_id == 'video-source-form':
            # Video source form submitted
            video_source = request.form.get('video_source')

            if video_source == 'camera':
                if CONFIG['use_test_video'] or CONFIG['camera_source'] != int(
                        request.form.get('camera_id', 0)):
                    CONFIG['use_test_video'] = False
                    CONFIG['camera_source'] = int(request.form.get('camera_id', 0))
                    restart_required = True
            elif video_source == 'test_video':
                if not CONFIG['use_test_video'] or CONFIG['test_video_path'] != request.form.get(
                        'test_video'):
                    CONFIG['use_test_video'] = True
                    CONFIG['test_video_path'] = request.form.get('test_video')
                    restart_required = True

        # Handle model settings form
        elif form_id == 'model-settings-form':
            # Get use_custom_model value (string 'true' or 'false')
            use_custom_model_str = request.form.get('use_custom_model', 'false')
            use_custom_model = use_custom_model_str.lower() == 'true'

            # Check if model setting has changed
            if CONFIG['use_custom_model'] != use_custom_model:
                CONFIG['use_custom_model'] = use_custom_model
                restart_required = True

            # If using built-in model
            if not use_custom_model:
                new_model_name = request.form.get('model_name', 'fasterrcnn_resnet50_fpn')
                if CONFIG['model_name'] != new_model_name:
                    CONFIG['model_name'] = new_model_name
                    restart_required = True

            # If using custom model
            else:
                # Update custom model path if provided
                custom_model_path = request.form.get('custom_model_path', '')
                if custom_model_path and CONFIG['custom_model_path'] != custom_model_path:
                    CONFIG['custom_model_path'] = custom_model_path
                    restart_required = True

                # Update custom model labels path if provided
                custom_model_labels_path = request.form.get('custom_model_labels_path', '')
                if CONFIG['custom_model_labels_path'] != custom_model_labels_path:
                    CONFIG['custom_model_labels_path'] = custom_model_labels_path
                    restart_required = True

        # Handle detection settings form
        elif form_id == 'detection-settings-form':
            # Update detection settings if present
            if 'confidence_threshold' in request.form:
                new_value = float(request.form.get('confidence_threshold'))
                if CONFIG['confidence_threshold'] != new_value:
                    CONFIG['confidence_threshold'] = new_value
                    config_changed = True

            if 'detection_interval' in request.form:
                new_value = float(request.form.get('detection_interval'))
                if CONFIG['detection_interval'] != new_value:
                    CONFIG['detection_interval'] = new_value
                    config_changed = True

            if 'detection_backoff' in request.form:
                new_value = int(request.form.get('detection_backoff'))
                if CONFIG['detection_backoff'] != new_value:
                    CONFIG['detection_backoff'] = new_value
                    config_changed = True

            # Update object detection filters
            # For checkboxes, getlist() returns all selected values
            new_enabled_objects = request.form.getlist(
                'enabled_objects') if 'enabled_objects' in request.form else []
            if sorted(CONFIG['enabled_objects']) != sorted(new_enabled_objects):
                CONFIG['enabled_objects'] = new_enabled_objects
                config_changed = True

            # Handle the display detection boxes checkbox
            new_display_detection_boxes = 'display_detection_boxes' in request.form
            if CONFIG['display_detection_boxes'] != new_display_detection_boxes:
                CONFIG['display_detection_boxes'] = new_display_detection_boxes
                config_changed = True

        # If config needs restart, handle it
        if restart_required:
            try:
                # Stop the current video processor
                if video_processor:
                    # Request clean shutdown of the processor
                    video_processor.camera.release()

                # Start a new video processor
                video_processor = start_video_processor()
                print("Video processor restarted with new configuration")
            except Exception as e:
                print(f"Error restarting video processor: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': f'Error restarting video processor: {str(e)}'
                })

        # Return success message as JSON
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully' + (
                ' (system restarted)' if restart_required else ''),
            'config': CONFIG
        })

    return jsonify({'success': False, 'error': 'Invalid request method'})


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Handle model file uploads."""
    if request.method == 'POST':
        if 'model_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'})

        file = request.files['model_file']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})

        if file:
            # Ensure models directory exists
            models_dir = os.path.join(os.getcwd(), 'models')
            os.makedirs(models_dir, exist_ok=True)

            # Determine file extension and set appropriate mime types
            filename = secure_filename(file.filename)
            file_path = os.path.join(models_dir, filename)

            try:
                # Save the file
                file.save(file_path)

                # Return relative path
                rel_path = os.path.join('models', filename)

                return jsonify({
                    'success': True,
                    'file_path': rel_path,
                    'message': 'Model file uploaded successfully'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': f'Error saving file: {str(e)}'})

    return jsonify({'success': False, 'error': 'Invalid request'})


@app.route('/upload_labels', methods=['POST'])
def upload_labels():
    """Handle labels file uploads."""
    if request.method == 'POST':
        if 'labels_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'})

        file = request.files['labels_file']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})

        if file:
            # Ensure models directory exists
            models_dir = os.path.join(os.getcwd(), 'models')
            os.makedirs(models_dir, exist_ok=True)

            # Determine file extension and set appropriate mime types
            filename = secure_filename(file.filename)
            file_path = os.path.join(models_dir, filename)

            try:
                # Save the file
                file.save(file_path)

                # Return relative path
                rel_path = os.path.join('models', filename)

                return jsonify({
                    'success': True,
                    'file_path': rel_path,
                    'message': 'Labels file uploaded successfully'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': f'Error saving file: {str(e)}'})

    return jsonify({'success': False, 'error': 'Invalid request'})


if __name__ == '__main__':
    # Create package files to make imports work
    os.makedirs('utils', exist_ok=True)
    with open('utils/__init__.py', 'w') as f:
        f.write('# Package initialization\n')

    # Start the video processor in a separate thread
    video_processor = start_video_processor()
    print(f"Main video processor initialized: {video_processor}")

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
