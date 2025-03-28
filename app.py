"""
Main application file for the home surveillance system.
"""

import os
from flask import Flask, Response, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from utils.video import generate_frames, detection_history, processing_lock, start_video_processor
from config import CONFIG

# Initialize Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
app.config['UPLOAD_FOLDER'] = '.'  # Current directory for uploads


@app.route('/')
def index():
    """Route for the main page."""
    return render_template('index.html',
                           use_test_video=CONFIG['use_test_video'],
                           test_video_path=CONFIG['test_video_path'] if CONFIG[
                               'use_test_video'] else "")


@app.route('/video_feed')
def video_feed():
    """Route for streaming the processed video feed."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/history')
def get_history():
    """Route to get detection history as JSON."""
    with processing_lock:
        return jsonify(detection_history)


@app.route('/detections/<path:filename>')
def serve_detection_file(filename):
    """Route to serve saved detection images and videos."""
    return send_from_directory(CONFIG['save_dir'], filename)


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
    if request.method == 'POST':
        # Get form data
        video_source = request.form.get('video_source')

        if video_source == 'camera':
            CONFIG['use_test_video'] = False
            CONFIG['camera_source'] = int(request.form.get('camera_id', 0))
        elif video_source == 'test_video':
            CONFIG['use_test_video'] = True
            CONFIG['test_video_path'] = request.form.get('test_video')

        # Update detection settings if present
        if 'confidence_threshold' in request.form:
            CONFIG['confidence_threshold'] = float(request.form.get('confidence_threshold'))

        if 'detection_interval' in request.form:
            CONFIG['detection_interval'] = float(request.form.get('detection_interval'))

        if 'clip_duration' in request.form:
            CONFIG['clip_duration'] = int(request.form.get('clip_duration'))

        # Restart video processor with new settings
        # In a real implementation, you would need to handle this more gracefully
        # For now, we'll just redirect to the home page
        return render_template('index.html',
                               use_test_video=CONFIG['use_test_video'],
                               test_video_path=CONFIG['test_video_path'] if CONFIG[
                                   'use_test_video'] else "")

    return jsonify({'success': False, 'error': 'Invalid request method'})


if __name__ == '__main__':
    # Create package files to make imports work
    os.makedirs('utils', exist_ok=True)
    with open('utils/__init__.py', 'w') as f:
        f.write('# Package initialization\n')

    # Start the video processor in a separate thread
    video_processor = start_video_processor()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)