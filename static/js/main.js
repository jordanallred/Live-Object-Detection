/**
 * Main JavaScript for the Live Object Detection System
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

            if (!data || data.length === 0) {
                historyContainer.innerHTML = `
                    <div style="text-align: center; padding: 3rem; background: var(--gray-100); border-radius: var(--border-radius); grid-column: 1 / -1;">
                        <span class="material-icons" style="font-size: 48px; color: var(--gray-400); display: block; margin-bottom: 1rem;">search_off</span>
                        <p>No detections found yet. Objects will appear here when detected.</p>
                    </div>`;
                return;
            }

            historyContainer.innerHTML = '';

            data.forEach(detection => {
                try {
                    // Format timestamp - handle both string format and numeric format
                    let formattedTime;
                    const timestamp = detection.timestamp;

                    if (typeof timestamp === 'string') {
                        // Try to parse the timestamp based on its format
                        if (timestamp.match(/^\d{8}_\d{6}$/)) { // Format: YYYYMMDD_HHMMSS
                            formattedTime = `${timestamp.substring(6, 8)}/${timestamp.substring(4, 6)}/${timestamp.substring(0, 4)} ${timestamp.substring(9, 11)}:${timestamp.substring(11, 13)}:${timestamp.substring(13, 15)}`;
                        } else {
                            // Default formatting for other string formats
                            const date = new Date(timestamp);
                            if (!isNaN(date.getTime())) {
                                formattedTime = date.toLocaleString();
                            } else {
                                // If all else fails, just use the raw timestamp
                                formattedTime = timestamp;
                            }
                        }
                    } else if (typeof timestamp === 'number') {
                        // Handle numeric timestamp (unix timestamp)
                        const date = new Date(timestamp * 1000); // Convert seconds to milliseconds
                        formattedTime = date.toLocaleString();
                    } else {
                        console.error('Invalid timestamp format:', timestamp);
                        formattedTime = 'Unknown time';
                    }

                    // Check if detections array exists
                    if (!detection.detections || !Array.isArray(detection.detections)) {
                        console.error('Missing or invalid detections array:', detection);
                        return; // Skip this detection
                    }

                    // Get detected objects
                    const objectCounts = {};
                    detection.detections.forEach(d => {
                        // Check if class_name exists
                        if (!d.class_name) {
                            console.error('Missing class_name in detection:', d);
                            return; // Skip this detection item
                        }
                        objectCounts[d.class_name] = (objectCounts[d.class_name] || 0) + 1;
                    });

                    const objectList = Object.entries(objectCounts)
                        .map(([obj, count]) => `${obj} (${count})`)
                        .join(', ');

                    // Create detection item
                    const detectionItem = document.createElement('div');
                    detectionItem.className = 'detection-item';

                    // Check if image path exists
                    if (!detection.image_path) {
                        console.error('Missing image_path in detection:', detection);
                        return; // Skip this detection
                    }

                    // Create the HTML for the detection item
                    detectionItem.innerHTML = `
                        <img src="/detections/${detection.image_path}" class="detection-img" alt="Detection">
                        <div class="detection-info">
                            <div class="detection-timestamp">
                                <span class="material-icons" style="font-size: 16px; vertical-align: middle; margin-right: 5px; color: var(--primary);">schedule</span>
                                ${formattedTime}
                            </div>
                            <div class="detection-objects">${objectList}</div>
                        </div>
                        <div class="buttons">
                            <button class="btn btn-details" onclick="showDetails('${formattedTime}', ${JSON.stringify(detection).replace(/"/g, '&quot;')})">
                                <span class="material-icons" style="vertical-align: middle; font-size: 16px; margin-right: 4px;">visibility</span>
                                Details
                            </button>
                        </div>
                    `;

                    historyContainer.appendChild(detectionItem);
                } catch (err) {
                    console.error('Error processing detection:', err, detection);
                }
            });

            // Make the detection images clickable to open in modal
            makeImagesClickable();
        })
        .catch(error => {
            console.error('Error loading detection history:', error);
            document.getElementById('detection-history').innerHTML = `
                <div style="text-align: center; padding: 3rem; background: var(--gray-100); border-radius: var(--border-radius); grid-column: 1 / -1;">
                    <span class="material-icons" style="font-size: 48px; color: var(--danger); display: block; margin-bottom: 1rem;">error</span>
                    <p>Error loading detection history. Please try again later.</p>
                    <p style="font-family: monospace; text-align: left; margin-top: 1rem; padding: 1rem; background: rgba(0,0,0,0.05); border-radius: 4px; overflow: auto; font-size: 12px;">${error.toString()}</p>
                    <a href="/debug_history" target="_blank" class="btn btn-details" style="margin-top: 1rem; display: inline-block;">
                        <span class="material-icons" style="vertical-align: middle; font-size: 16px; margin-right: 4px;">bug_report</span>
                        Debug History
                    </a>
                </div>`;
        });
}
// Function to make detection images clickable
function makeImagesClickable() {
    const historyImages = document.querySelectorAll('#detection-history .detection-img');
    historyImages.forEach(img => {
        img.style.cursor = 'pointer';
        img.addEventListener('click', function () {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = this.src;
        });
    });
}

// Show details function
function showDetails(timestamp, detection) {
    document.getElementById('details-title').innerHTML = `
        <span class="material-icons" style="vertical-align: middle; margin-right: 10px; color: var(--primary);">info</span>
        Detection Details - ${timestamp}
    `;

    // Format the JSON for better readability
    const formattedJson = JSON.stringify(detection, null, 2);

    // Create a more friendly display of the detection data
    let objectList = '';
    if (detection.detections && detection.detections.length > 0) {
        objectList = '<div style="margin-bottom: 15px; background-color: var(--gray-100); padding: 15px; border-radius: var(--border-radius);">';
        objectList += '<h3 style="margin-bottom: 10px; color: var(--primary-dark);">Detected Objects</h3>';

        // Count objects by class
        const objectCounts = {};
        detection.detections.forEach(d => {
            objectCounts[d.class_name] = (objectCounts[d.class_name] || 0) + 1;
        });

        // Display as badges
        Object.entries(objectCounts).forEach(([className, count]) => {
            objectList += `<span class="badge badge-primary" style="margin: 3px;">${className}: ${count}</span>`;
        });

        objectList += '</div>';
    }

    // Set the content with formatted JSON
    document.getElementById('details-content').innerHTML = objectList + '<pre style="margin-top: 10px;">' + formattedJson + '</pre>';

    // Display the modal
    document.getElementById('detailsModal').style.display = 'block';
}

// Close modal function
function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';

    // If this is the image modal, reset the image src to avoid caching issues on next open
    if (modalId === 'imageModal') {
        document.getElementById('modalImage').src = '';
    }
}

// Upload test video function
function uploadTestVideo() {
    const fileInput = document.getElementById('video-upload');
    if (!fileInput.files || fileInput.files.length === 0) {
        showStatus('video-source-form', 'Please select a video file first.', false);
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('video_file', file);

    // Show loading indicator
    const uploadBtn = document.querySelector('button[onclick="uploadTestVideo()"]');
    const originalBtnText = uploadBtn.innerHTML;
    uploadBtn.innerHTML = '<div class="loader" style="width: 16px; height: 16px; margin-right: 8px; display: inline-block;"></div> Uploading...';
    uploadBtn.disabled = true;

    fetch('/upload_test_video', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            // Reset button
            uploadBtn.innerHTML = originalBtnText;
            uploadBtn.disabled = false;

            if (data.success) {
                showStatus('video-source-form', `Video '${data.filename}' uploaded successfully!`, true);

                // Add the new video to the dropdown
                const option = document.createElement('option');
                option.value = data.filename;
                option.text = data.filename;
                option.selected = true;

                const selectElement = document.getElementById('test-video-select');
                // Remove existing options
                while (selectElement.options.length > 0) {
                    selectElement.remove(0);
                }

                // Add the new option
                selectElement.appendChild(option);

                // Ensure the radio button is selected
                document.getElementById('use-test-video').checked = true;
                document.getElementById('test-video-select').disabled = false;
                document.getElementById('camera-select').disabled = true;
            } else {
                showStatus('video-source-form', 'Error uploading video: ' + data.error, false);
            }
        })
        .catch(error => {
            // Reset button
            uploadBtn.innerHTML = originalBtnText;
            uploadBtn.disabled = false;

            console.error('Error uploading video:', error);
            showStatus('video-source-form', 'Error uploading video. Check console for details.', false);
        });
}

// Helper function to display form submission status
function showStatus(formId, message, isSuccess = true) {
    const statusElement = document.getElementById(formId + '-status');
    if (statusElement) {
        // Clear any existing timeout
        if (statusElement.fadeTimeout) {
            clearTimeout(statusElement.fadeTimeout);
        }

        // Remove any previous fade-out animation
        statusElement.classList.remove('fade-out');

        // Add icon to message
        const icon = isSuccess
            ? '<span class="material-icons" style="vertical-align: middle; margin-right: 8px; font-size: 20px;">check_circle</span>'
            : '<span class="material-icons" style="vertical-align: middle; margin-right: 8px; font-size: 20px;">error</span>';

        statusElement.innerHTML = icon + message;
        statusElement.className = 'status-message ' + (isSuccess ? 'status-success' : 'status-error');

        // Add pulse animation to the save button if success
        if (isSuccess) {
            const formElement = document.getElementById(formId);
            if (formElement) {
                const saveButton = formElement.querySelector('button[type="submit"]');
                if (saveButton) {
                    saveButton.classList.add('pulse');
                    setTimeout(() => {
                        saveButton.classList.remove('pulse');
                    }, 2000);
                }
            }
        }

        // Hide the status message after 4 seconds with fade-out animation
        statusElement.fadeTimeout = setTimeout(() => {
            statusElement.classList.add('fade-out');

            // Remove the element from the DOM after animation completes
            setTimeout(() => {
                statusElement.className = 'status-message';
            }, 500); // Match this with animation duration
        }, 4000);
    }
}

// Handle form submission via AJAX
function submitFormAjax(formId) {
    const form = document.getElementById(formId);

    form.addEventListener('submit', function (event) {
        event.preventDefault();

        const formData = new FormData(form);
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalBtnText = submitBtn.innerHTML;

        // Show loading indicator
        submitBtn.innerHTML = '<div class="loader" style="width: 16px; height: 16px; margin-right: 8px; display: inline-block;"></div> Saving...';
        submitBtn.disabled = true;

        // Submit the form via AJAX
        fetch(form.action, {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                // Reset button
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;

                if (data.success) {
                    showStatus(formId, data.message || 'Settings updated successfully', true);

                    // Update the confidence threshold display if it was updated
                    if (formId === 'detection-settings-form' && data.config && data.config.confidence_threshold) {
                        const confidenceValue = document.getElementById('confidence-value');
                        if (confidenceValue) {
                            confidenceValue.textContent = data.config.confidence_threshold;
                        }
                    }
                } else {
                    showStatus(formId, data.error || 'Error updating settings', false);
                }
            })
            .catch(error => {
                // Reset button
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;

                console.error('Error submitting form:', error);
                showStatus(formId, 'Error communicating with server', false);
            });
    });
}

// Function to toggle model options based on radio selection
function toggleModelOptions() {
    const useCustomModel = document.getElementById('use-custom-model').checked;
    document.getElementById('builtin-model-options').style.display = useCustomModel ? 'none' : 'block';
    document.getElementById('custom-model-options').style.display = useCustomModel ? 'block' : 'none';
}

// Function to handle model file selection
function handleModelFileSelect(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];

        // Create a FormData object to upload the file
        const formData = new FormData();
        formData.append('model_file', file);

        // Show uploading status
        const modelPathInput = document.getElementById('custom-model-path');
        const originalValue = modelPathInput.value;
        modelPathInput.value = 'Uploading...';
        modelPathInput.disabled = true;

        // Upload the file to the server
        fetch('/upload_model', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update the path input with the saved file path
                    modelPathInput.value = data.file_path;
                    showStatus('model-settings-form', `Model file '${file.name}' uploaded successfully!`, true);
                } else {
                    modelPathInput.value = originalValue;
                    showStatus('model-settings-form', `Error: ${data.error}`, false);
                }
            })
            .catch(error => {
                console.error('Error uploading model file:', error);
                modelPathInput.value = originalValue;
                showStatus('model-settings-form', 'Error uploading file. Check console for details.', false);
            })
            .finally(() => {
                modelPathInput.disabled = false;
            });
    }
}

// Function to handle labels file selection
function handleLabelsFileSelect(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];

        // Create a FormData object to upload the file
        const formData = new FormData();
        formData.append('labels_file', file);

        // Show uploading status
        const labelsPathInput = document.getElementById('custom-model-labels-path');
        const originalValue = labelsPathInput.value;
        labelsPathInput.value = 'Uploading...';
        labelsPathInput.disabled = true;

        // Upload the file to the server
        fetch('/upload_labels', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update the path input with the saved file path
                    labelsPathInput.value = data.file_path;
                    showStatus('model-settings-form', `Labels file '${file.name}' uploaded successfully!`, true);
                } else {
                    labelsPathInput.value = originalValue;
                    showStatus('model-settings-form', `Error: ${data.error}`, false);
                }
            })
            .catch(error => {
                console.error('Error uploading labels file:', error);
                labelsPathInput.value = originalValue;
                showStatus('model-settings-form', 'Error uploading file. Check console for details.', false);
            })
            .finally(() => {
                labelsPathInput.disabled = false;
            });
    }
}

// Function to load available detection objects based on current model
function loadAvailableDetectionObjects() {
    const checkboxGroup = document.getElementById('objects-checkbox-group');
    if (!checkboxGroup) return;

    // Show loading indicator
    checkboxGroup.innerHTML = `
        <div class="loader-container">
            <div class="loader"></div>
            <span>Loading available objects...</span>
        </div>
    `;

    // Fetch the available objects from the server
    fetch('/available_objects')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.objects) {
                // Get the enabled objects from the config
                const enabledObjects = data.enabled_objects || [];
                checkboxGroup.innerHTML = '';

                // Update object count badge
                updateObjectCountBadge(enabledObjects.length, data.objects.length);

                // Sort objects alphabetically for better UX
                const sortedObjects = [...data.objects].sort();

                // Create a checkbox for each available object
                sortedObjects.forEach(className => {
                    // Skip '__background__' class which is not a real detectable object
                    if (className === '__background__') return;

                    const checkboxId = `detect-${className}`;
                    const isChecked = enabledObjects.includes(className);

                    // Determine icon based on object class
                    let icon = 'lens';
                    if (className === 'person') icon = 'person';
                    else if (['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'automobile'].includes(className)) icon = 'directions_car';
                    else if (['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'].includes(className)) icon = 'pets';
                    else if (['airplane', 'train', 'boat'].includes(className)) icon = 'commute';
                    else if (['laptop', 'tv', 'cell phone', 'keyboard', 'mouse'].includes(className)) icon = 'devices';
                    else if (['chair', 'couch', 'bed', 'dining table', 'toilet'].includes(className)) icon = 'chair';
                    else if (['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'].includes(className)) icon = 'restaurant';
                    else if (['banana', 'apple', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'].includes(className)) icon = 'restaurant_menu';

                    // Create checkbox item
                    const checkboxItem = document.createElement('div');
                    checkboxItem.className = 'checkbox-item';
                    checkboxItem.innerHTML = `
                        <input type="checkbox" id="${checkboxId}" name="enabled_objects" value="${className}" 
                               ${isChecked ? 'checked' : ''} onchange="updateObjectCountBadge()">
                        <label for="${checkboxId}">
                            <span class="material-icons" style="vertical-align: middle; margin-right: 4px; font-size: 16px;">
                                ${icon}
                            </span>
                            ${className.charAt(0).toUpperCase() + className.slice(1)}
                        </label>
                    `;

                    checkboxGroup.appendChild(checkboxItem);
                });

                // Set up Select All and Clear All buttons
                setupObjectFilterButtons(data.objects);
            } else {
                // If there was an error or no objects returned
                checkboxGroup.innerHTML = `
                    <div style="text-align: center; padding: 1rem;">
                        <span class="material-icons" style="color: var(--warning); font-size: 24px; margin-bottom: 0.5rem;">warning</span>
                        <p>Could not load available objects. Using default COCO classes.</p>
                    </div>
                `;

                // Load default COCO classes as fallback
                loadDefaultObjectClasses();
            }
        })
        .catch(error => {
            console.error('Error loading available objects:', error);
            checkboxGroup.innerHTML = `
                <div style="text-align: center; padding: 1rem;">
                    <span class="material-icons" style="color: var(--danger); font-size: 24px; margin-bottom: 0.5rem;">error</span>
                    <p>Error loading available objects. Please try refreshing the page.</p>
                </div>
            `;

            // Load default COCO classes as fallback
            loadDefaultObjectClasses();
        });
}

// Function to update the object count badge
function updateObjectCountBadge(selectedCount = null, totalCount = null) {
    const badge = document.getElementById('object-count-badge');
    if (!badge) return;

    if (selectedCount === null) {
        // Count the selected checkboxes
        const checkedBoxes = document.querySelectorAll('input[name="enabled_objects"]:checked');
        selectedCount = checkedBoxes.length;
    }

    badge.textContent = selectedCount + (totalCount ? ` / ${totalCount}` : '') + ' selected';

    // Update badge color based on selection count
    if (selectedCount === 0) {
        badge.className = 'badge badge-warning';
    } else if (selectedCount < 5) {
        badge.className = 'badge badge-primary';
    } else {
        badge.className = 'badge badge-success';
    }
}

// Function to set up Select All and Clear All buttons
function setupObjectFilterButtons(objectsList) {
    const selectAllBtn = document.getElementById('select-all-objects');
    const clearAllBtn = document.getElementById('clear-all-objects');

    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', function () {
            const checkboxes = document.querySelectorAll('input[name="enabled_objects"]');
            checkboxes.forEach(checkbox => {
                checkbox.checked = true;
            });
            updateObjectCountBadge(checkboxes.length, objectsList.length);
        });
    }

    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', function () {
            const checkboxes = document.querySelectorAll('input[name="enabled_objects"]');
            checkboxes.forEach(checkbox => {
                checkbox.checked = false;
            });
            updateObjectCountBadge(0, objectsList.length);
        });
    }
}

// Fallback function to load default COCO classes
function loadDefaultObjectClasses() {
    // Common COCO classes (without background and N/A entries)
    const cocoClasses = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ];

    const checkboxGroup = document.getElementById('objects-checkbox-group');
    const enabledObjects = [];  // We don't know which objects are enabled by default

    checkboxGroup.innerHTML = '';

    // Create a checkbox for each object class
    cocoClasses.forEach(className => {
        const checkboxId = `detect-${className.replace(' ', '-')}`;

        // Determine icon based on object class
        let icon = 'lens';
        if (className === 'person') icon = 'person';
        else if (['car', 'truck', 'bus', 'motorcycle', 'bicycle'].includes(className)) icon = 'directions_car';
        else if (['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'].includes(className)) icon = 'pets';

        // Create checkbox item
        const checkboxItem = document.createElement('div');
        checkboxItem.className = 'checkbox-item';
        checkboxItem.innerHTML = `
            <input type="checkbox" id="${checkboxId}" name="enabled_objects" value="${className}" 
                   onchange="updateObjectCountBadge()">
            <label for="${checkboxId}">
                <span class="material-icons" style="vertical-align: middle; margin-right: 4px; font-size: 16px;">
                    ${icon}
                </span>
                ${className.charAt(0).toUpperCase() + className.slice(1)}
            </label>
        `;

        checkboxGroup.appendChild(checkboxItem);
    });

    // Set up filter buttons
    setupObjectFilterButtons(cocoClasses);
    updateObjectCountBadge(0, cocoClasses.length);
}

// Event listeners for UI controls
document.addEventListener('DOMContentLoaded', function () {
    // Update confidence threshold display
    const confidenceSlider = document.getElementById('confidence-threshold');
    if (confidenceSlider) {
        confidenceSlider.addEventListener('input', function () {
            const confidenceValue = document.getElementById('confidence-value');
            confidenceValue.textContent = this.value;

            // Update badge color based on confidence level
            const value = parseFloat(this.value);
            confidenceValue.className = 'badge ' +
                (value >= 0.7 ? 'badge-success' :
                    value >= 0.4 ? 'badge-primary' :
                        'badge-warning');
        });

        // Initialize badge color
        const value = parseFloat(confidenceSlider.value);
        const confidenceValue = document.getElementById('confidence-value');
        confidenceValue.className = 'badge ' +
            (value >= 0.7 ? 'badge-success' :
                value >= 0.4 ? 'badge-primary' :
                    'badge-warning');
    }

    // Toggle camera/test video selection
    const useCameraRadio = document.getElementById('use-camera');
    if (useCameraRadio) {
        useCameraRadio.addEventListener('change', function () {
            document.getElementById('camera-select').disabled = !this.checked;
            document.getElementById('test-video-select').disabled = this.checked;
        });
    }

    const useTestVideoRadio = document.getElementById('use-test-video');
    if (useTestVideoRadio) {
        useTestVideoRadio.addEventListener('change', function () {
            document.getElementById('test-video-select').disabled = !this.checked;
            document.getElementById('camera-select').disabled = this.checked;
        });
    }

    // Close modal when clicking outside
    window.onclick = function (event) {
        if (event.target.classList.contains('modal')) {
            event.target.style.display = 'none';

            // If this is the image modal, reset the image src
            if (event.target.id === 'imageModal') {
                document.getElementById('modalImage').src = '';
            }
        }
    };

    // Set up AJAX form submission for settings forms
    submitFormAjax('video-source-form');
    submitFormAjax('detection-settings-form');
    submitFormAjax('model-settings-form');

    // Load available detection objects when settings tab is opened
    const settingsTab = document.querySelector('.tab[onclick="openTab(\'settings\')"]');
    if (settingsTab) {
        settingsTab.addEventListener('click', function () {
            loadAvailableDetectionObjects();
        });
    }

    // If we're already on the settings tab, load objects now
    if (window.location.hash === '#settings') {
        loadAvailableDetectionObjects();
    }

    // Initial tab
    openTab('live');
});
