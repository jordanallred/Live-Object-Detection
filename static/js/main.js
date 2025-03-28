/**
 * Main JavaScript for the Home Surveillance System
 */

// Open tab function
function openTab(tabName) {
    const tabs = document.getElementsByClassName('tab');
    const contents = document.getElementsByClassName('content');

    // Deactivate all tabs and hide content
    for (let i = 0; i < tabs.length; i++) {
        tabs[i].classList.remove('active');
    }
    for (let i = 0; i < contents.length; i++) {
        contents[i].style.display = 'none';
    }

    // Activate selected tab and show content
    document.getElementById(tabName).style.display = 'block';
    document.querySelector(`.tab[onclick="openTab('${tabName}')"]`).classList.add('active');

    // Load history if history tab selected
    if (tabName === 'history') {
        loadDetectionHistory();
    }
}

// Load detection history
function loadDetectionHistory() {
    fetch('/history')
        .then(response => response.json())
        .then(data => {
            const historyContainer = document.getElementById('detection-history');

            if (data.length === 0) {
                historyContainer.innerHTML = '<p>No detections found.</p>';
                return;
            }

            historyContainer.innerHTML = '';

            data.forEach(detection => {
                // Format timestamp
                const timestamp = detection.timestamp;
                const formattedTime = `${timestamp.substring(8, 10)}/${timestamp.substring(6, 8)}/${timestamp.substring(0, 4)} ${timestamp.substring(11, 13)}:${timestamp.substring(13, 15)}:${timestamp.substring(15, 17)}`;

                // Get detected objects
                const objectCounts = {};
                detection.detections.forEach(d => {
                    objectCounts[d.class_name] = (objectCounts[d.class_name] || 0) + 1;
                });

                const objectList = Object.entries(objectCounts)
                    .map(([obj, count]) => `${obj} (${count})`)
                    .join(', ');

                // Create detection item
                const detectionItem = document.createElement('div');
                detectionItem.className = 'detection-item';
                detectionItem.innerHTML = `
                    <img src="/detections/${detection.image_path}" class="detection-img" alt="Detection">
                    <div class="detection-info">
                        <div class="detection-timestamp">${formattedTime}</div>
                        <div class="detection-objects">${objectList}</div>
                    </div>
                    <div class="buttons">
                        <button class="btn btn-video" onclick="playVideo('${detection.video_path}', '${formattedTime}')">Play Video</button>
                        <button class="btn btn-details" onclick="showDetails('${formattedTime}', ${JSON.stringify(detection).replace(/"/g, '&quot;')})">Details</button>
                    </div>
                `;

                historyContainer.appendChild(detectionItem);
            });
        })
        .catch(error => {
            console.error('Error loading detection history:', error);
            document.getElementById('detection-history').innerHTML = '<p>Error loading detection history.</p>';
        });
}

// Play video function
function playVideo(videoPath, timestamp) {
    const videoPlayer = document.getElementById('video-player');
    videoPlayer.src = `/detections/${videoPath}`;
    document.getElementById('video-title').textContent = `Detection Video - ${timestamp}`;
    document.getElementById('videoModal').style.display = 'block';
    videoPlayer.play();
}

// Show details function
function showDetails(timestamp, detection) {
    document.getElementById('details-title').textContent = `Detection Details - ${timestamp}`;
    document.getElementById('details-content').textContent = JSON.stringify(detection, null, 2);
    document.getElementById('detailsModal').style.display = 'block';
}

// Close modal function
function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
    if (modalId === 'videoModal') {
        document.getElementById('video-player').pause();
    }
}

// Upload test video function
function uploadTestVideo() {
    const fileInput = document.getElementById('video-upload');
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select a video file first.');
        return;
    }

    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append('video_file', file);

    fetch('/upload_test_video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Video uploaded successfully!');
            // Add the new video to the dropdown
            const option = document.createElement('option');
            option.value = data.filename;
            option.text = data.filename;
            option.selected = true;
            document.getElementById('test-video-select').appendChild(option);
        } else {
            alert('Error uploading video: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error uploading video:', error);
        alert('Error uploading video. Check console for details.');
    });
}

// Event listeners for UI controls
document.addEventListener('DOMContentLoaded', function() {
    // Update confidence threshold display
    const confidenceSlider = document.getElementById('confidence-threshold');
    if (confidenceSlider) {
        confidenceSlider.addEventListener('input', function() {
            document.getElementById('confidence-value').textContent = this.value;
        });
    }

    // Toggle camera/test video selection
    const useCameraRadio = document.getElementById('use-camera');
    if (useCameraRadio) {
        useCameraRadio.addEventListener('change', function() {
            document.getElementById('camera-select').disabled = !this.checked;
            document.getElementById('test-video-select').disabled = this.checked;
        });
    }

    const useTestVideoRadio = document.getElementById('use-test-video');
    if (useTestVideoRadio) {
        useTestVideoRadio.addEventListener('change', function() {
            document.getElementById('test-video-select').disabled = !this.checked;
            document.getElementById('camera-select').disabled = this.checked;
        });
    }

    // Close modal when clicking outside
    window.onclick = function(event) {
        if (event.target.className === 'modal') {
            event.target.style.display = 'none';
            if (event.target.id === 'videoModal') {
                document.getElementById('video-player').pause();
            }
        }
    };

    // Initial tab
    openTab('live');
});