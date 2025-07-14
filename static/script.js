const mainVideo = document.getElementById('live-video-stream');
        const startMainCameraBtn = document.getElementById('start-main-camera-btn');
        const stopMainCameraBtn = document.getElementById('stop-main-camera-btn');
        const mainCameraStatus = document.getElementById('main-camera-status');
        const recognitionStatus = document.getElementById('recognition-status');
        const liveFeedCanvas = document.getElementById('live-feed-canvas');
        const liveFeedCtx = liveFeedCanvas.getContext('2d');

        // --- Enrollment Camera Elements ---
        const enrollmentVideoPreview = document.getElementById('enrollment-video-preview');
        const enrollmentCanvasPreview = document.getElementById('enrollment-canvas-preview');
        const enrollmentCtxPreview = enrollmentCanvasPreview.getContext('2d');
        const enrollNameInput = document.getElementById('enroll-name');
        const captureBtn = document.getElementById('capture-btn'); // Now acts as "Start Capture" or "Retake"
        const confirmCaptureBtn = document.getElementById('confirm-capture-btn'); // New button for confirmation
        const enrollBtn = document.getElementById('enroll-btn');
        const enrollmentMessage = document.getElementById('enrollment-message');

        // --- Shared Camera Select ---
        const cameraSelect = document.getElementById('camera-select');

        // --- Dashboard Elements ---
        const presentCount = document.getElementById('present-count');
        const totalEntriesToday = document.getElementById('total-entries-today');
        const dashboardStatus = document.getElementById('dashboard-status');
        const attendanceLogsBody = document.getElementById('attendance-logs-body');
        const logsStatus = document.getElementById('logs-status');

        // --- Global State Variables ---
        let currentMainStream = null;
        let currentEnrollmentStream = null;
        let mainCaptureInterval = null;
        let capturedImageData = null; // To store the image data for enrollment
        let recentLogsData = []; // Array to hold recent logs for display

        const FRAME_RATE = 5; // Frames per second to send to backend
        const RECENT_LOGS_DISPLAY_LIMIT = 5; // Limit for recent attendance logs to display

        // --- Utility Functions ---
        function updateDashboard(summary) {
            presentCount.textContent = summary.present_count;
            totalEntriesToday.textContent = summary.total_entries_today;
            dashboardStatus.textContent = ''; // Clear loading status
        }

        async function fetchInitialAttendanceLogs() {
            logsStatus.textContent = 'Loading attendance logs...';
            try {
                // Fetch only the limited number of recent logs on initial load
                const response = await fetch(`/api/attendance_logs?limit=${RECENT_LOGS_DISPLAY_LIMIT}`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const logs = await response.json();
                recentLogsData = logs; // Store fetched logs
                renderRecentLogs(); // Render them to the table
                logsStatus.textContent = ''; // Clear loading status
            } catch (error) {
                console.error('Error fetching initial attendance logs:', error);
                logsStatus.textContent = 'Failed to load initial attendance logs.';
            }
        }

        function renderRecentLogs() {
            attendanceLogsBody.innerHTML = ''; // Clear previous logs
            if (recentLogsData.length === 0) {
                attendanceLogsBody.innerHTML = '<tr><td colspan="3" class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 text-center">No recent activity.</td></tr>';
            } else {
                recentLogsData.forEach(log => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${log.user_id}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${new Date(log.timestamp).toLocaleString()}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm font-bold ${log.status === 'IN' ? 'text-green-600' : 'text-red-600'}">${log.status}</td>
                    `;
                    attendanceLogsBody.appendChild(row);
                });
            }
        }

        // --- Camera Management (Shared) ---
        async function getConnectedCameras() {
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                cameraSelect.innerHTML = ''; // Clear existing options
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                if (videoDevices.length === 0) {
                    cameraSelect.innerHTML = '<option value="">No cameras found</option>';
                    startMainCameraBtn.disabled = true;
                    captureBtn.disabled = true;
                } else {
                    videoDevices.forEach(device => {
                        const option = document.createElement('option');
                        option.value = device.deviceId;
                        option.textContent = device.label || `Camera ${cameraSelect.options.length + 1}`;
                        cameraSelect.appendChild(option);
                    });
                    startMainCameraBtn.disabled = false;
                    captureBtn.disabled = false; // Enable capture button for enrollment
                }
            } catch (error) {
                console.error('Error enumerating devices:', error);
                cameraSelect.innerHTML = '<option value="">Error loading cameras</option>';
                startMainCameraBtn.disabled = true;
                captureBtn.disabled = true;
            }
        }

        // --- Main Camera Control (Dashboard) ---
        async function startMainCamera() {
            // Stop any active streams first
            stopMainCamera();
            stopEnrollmentCamera(); 

            const selectedCameraId = cameraSelect.value;
            if (!selectedCameraId) {
                alert('Please select a camera.');
                return;
            }

            try {
                mainCameraStatus.textContent = 'Main Camera Status: Starting...';
                recognitionStatus.textContent = 'Status: Starting main camera...';
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: selectedCameraId ? { exact: selectedCameraId } : undefined }
                });
                mainVideo.srcObject = stream;
                currentMainStream = stream;

                mainVideo.onloadedmetadata = () => {
                    mainVideo.play();
                    liveFeedCanvas.width = mainVideo.videoWidth;
                    liveFeedCanvas.height = mainVideo.videoHeight;
                    mainCameraStatus.textContent = 'Main Camera Status: Running';
                    recognitionStatus.textContent = 'Status: Live';
                    recognitionStatus.style.color = 'blue';
                    startMainCameraBtn.disabled = true;
                    stopMainCameraBtn.disabled = false;

                    if (mainCaptureInterval) clearInterval(mainCaptureInterval);
                    mainCaptureInterval = setInterval(sendFrameToBackend, 1000 / FRAME_RATE);
                };

            } catch (error) {
                console.error('Error accessing main camera:', error);
                mainCameraStatus.textContent = 'Main Camera Status: Failed to start';
                recognitionStatus.textContent = 'Status: Main Camera Error';
                recognitionStatus.style.color = 'red';
                alert(`Failed to start main camera: ${error.message}. Please ensure camera is connected and not in use.`);
                startMainCameraBtn.disabled = false;
                stopMainCameraBtn.disabled = true;
            }
        }

        function stopMainCamera() {
            if (currentMainStream) {
                currentMainStream.getTracks().forEach(track => track.stop());
                currentMainStream = null;
                mainVideo.srcObject = null;
                mainVideo.load(); // Reset video element
                clearInterval(mainCaptureInterval);
                mainCaptureInterval = null;
                liveFeedCtx.clearRect(0, 0, liveFeedCanvas.width, liveFeedCanvas.height); // Clear processed canvas
            }
            mainCameraStatus.textContent = 'Main Camera Status: Stopped';
            recognitionStatus.textContent = 'Status: Main Camera Stopped';
            recognitionStatus.style.color = 'gray';
            startMainCameraBtn.disabled = false;
            stopMainCameraBtn.disabled = true;
            // Do NOT clear recentLogsData or dashboard summary here. They should persist.
        }

        // --- Frame Processing (Frontend to Backend for Main Feed) ---
        async function sendFrameToBackend() {
            if (!currentMainStream || mainVideo.readyState !== mainVideo.HAVE_ENOUGH_DATA) {
                recognitionStatus.textContent = 'Status: Waiting for video data...';
                recognitionStatus.style.color = 'orange';
                return;
            }

            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = mainVideo.videoWidth;
            tempCanvas.height = mainVideo.videoHeight;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(mainVideo, 0, 0, tempCanvas.width, tempCanvas.height);
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.7);

            try {
                const response = await fetch('/api/process_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_data: imageData })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const result = await response.json();

                // Display processed image on live feed canvas
                const img = new Image();
                img.onload = () => {
                    liveFeedCtx.clearRect(0, 0, liveFeedCanvas.width, liveFeedCanvas.height);
                    liveFeedCtx.drawImage(img, 0, 0, liveFeedCanvas.width, liveFeedCanvas.height);
                };
                img.src = result.processed_image_data;

                // Update status text below canvas
                if (result.attendance_log) {
                    recognitionStatus.textContent = `${result.attendance_log.user_id} - ${result.attendance_log.status}!`;
                    recognitionStatus.style.color = result.attendance_log.status === 'IN' ? 'green' : 'red';
                    
                    // Prepend new log and enforce limit
                    recentLogsData.unshift(result.attendance_log);
                    if (recentLogsData.length > RECENT_LOGS_DISPLAY_LIMIT) {
                        recentLogsData = recentLogsData.slice(0, RECENT_LOGS_DISPLAY_LIMIT);
                    }
                    renderRecentLogs(); // Re-render the table with new data
                } else {
                    recognitionStatus.textContent = "Status: Live (Processing)";
                    recognitionStatus.style.color = 'blue';
                }
                
                // Update dashboard summary
                updateDashboard(result.dashboard_summary);

            } catch (error) {
                console.error('Error processing frame:', error);
                recognitionStatus.textContent = `Status: Error (${error.message})`;
                recognitionStatus.style.color = 'red';
            }
        }

        // --- Enrollment Camera Control ---
        async function startEnrollmentCameraPreview() {
            // Stop main camera if running to avoid conflicts
            stopMainCamera(); 
            stopEnrollmentCamera(); // Ensure any previous enrollment stream is stopped

            const selectedCameraId = cameraSelect.value;
            if (!selectedCameraId) {
                enrollmentMessage.textContent = 'Please select a camera in Camera Management.';
                enrollmentMessage.style.color = 'red';
                return null;
            }

            try {
                enrollmentMessage.textContent = 'Opening camera for capture...';
                enrollmentMessage.style.color = 'blue';
                
                // Show enrollment video preview, hide canvas with old image
                enrollmentVideoPreview.style.display = 'block';
                enrollmentCanvasPreview.style.display = 'none'; // Hide canvas when showing live feed

                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: selectedCameraId ? { exact: selectedCameraId } : undefined }
                });
                enrollmentVideoPreview.srcObject = stream;
                currentEnrollmentStream = stream;

                enrollmentVideoPreview.onloadedmetadata = () => {
                    enrollmentVideoPreview.play();
                    enrollmentCanvasPreview.width = enrollmentVideoPreview.videoWidth;
                    enrollmentCanvasPreview.height = enrollmentVideoPreview.videoHeight;
                    enrollmentMessage.textContent = 'Camera ready for capture. Adjust face and click \'Confirm Capture\'.';
                    enrollmentMessage.style.color = 'green';
                    
                    // Update button states
                    captureBtn.textContent = 'Retake Capture';
                    confirmCaptureBtn.style.display = 'block'; // Show confirm button
                    enrollBtn.disabled = true; // Disable enroll until confirmed capture
                    capturedImageData = null; // Clear any previously captured image data
                };
                return stream;

            } catch (error) {
                console.error('Error accessing enrollment camera:', error);
                enrollmentMessage.textContent = `Failed to open camera: ${error.message}`;
                enrollmentMessage.style.color = 'red';
                enrollmentVideoPreview.style.display = 'none';
                enrollmentCanvasPreview.style.display = 'none';
                // Reset button states on error
                captureBtn.textContent = 'Capture Face';
                confirmCaptureBtn.style.display = 'none';
                enrollBtn.disabled = true;
                return null;
            }
        }

        function stopEnrollmentCamera() {
            if (currentEnrollmentStream) {
                currentEnrollmentStream.getTracks().forEach(track => track.stop());
                currentEnrollmentStream = null;
                enrollmentVideoPreview.srcObject = null;
                enrollmentVideoPreview.load(); // Reset video element
            }
            // Do NOT clear canvas here, as it might contain the captured image
            enrollmentVideoPreview.style.display = 'none'; // Hide video preview
        }

        // --- Enrollment Logic ---
        captureBtn.addEventListener('click', async () => {
            // This button now initiates the preview or retakes
            startEnrollmentCameraPreview();
        });

        confirmCaptureBtn.addEventListener('click', () => {
            if (currentEnrollmentStream && enrollmentVideoPreview.readyState === enrollmentVideoPreview.HAVE_ENOUGH_DATA) {
                // Draw the current frame from the video to the canvas
                enrollmentCtxPreview.drawImage(enrollmentVideoPreview, 0, 0, enrollmentCanvasPreview.width, enrollmentCanvasPreview.height);
                capturedImageData = enrollmentCanvasPreview.toDataURL('image/jpeg', 0.9); // Higher quality for enrollment

                enrollmentMessage.textContent = 'Image captured. Click Enroll to save.';
                enrollmentMessage.style.color = 'blue';
                
                // Hide video preview and show canvas with captured image
                stopEnrollmentCamera(); // Stop the live stream
                enrollmentCanvasPreview.style.display = 'block'; // Show the canvas with the captured image

                // Update button states
                confirmCaptureBtn.style.display = 'none'; // Hide confirm button
                enrollBtn.disabled = false; // Enable enroll button
                captureBtn.textContent = 'Retake Capture'; // Change to Retake
            } else {
                enrollmentMessage.textContent = 'Video stream not ready for capture.';
                enrollmentMessage.style.color = 'red';
                stopEnrollmentCamera(); // Stop enrollment camera if not ready
            }
        });


        enrollBtn.addEventListener('click', async () => {
            const name = enrollNameInput.value.trim();
            if (!name) {
                enrollmentMessage.textContent = 'Please enter an employee name.';
                enrollmentMessage.style.color = 'red';
                return;
            }
            if (!capturedImageData) {
                enrollmentMessage.textContent = 'Please capture a face first.';
                enrollmentMessage.style.color = 'red';
                return;
            }

            enrollBtn.disabled = true;
            enrollmentMessage.textContent = 'Enrolling face...';
            enrollmentMessage.style.color = 'blue';

            try {
                const response = await fetch('/api/enroll_face', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: name, image_data: capturedImageData })
                });

                const result = await response.json();
                if (response.ok) {
                    enrollmentMessage.textContent = `Success: ${result.message}`;
                    enrollmentMessage.style.color = 'green';
                    enrollNameInput.value = ''; // Clear input
                    capturedImageData = null; // Clear captured image
                    enrollBtn.disabled = true; // Disable enroll until new capture
                    captureBtn.textContent = 'Capture Face'; // Reset capture button text
                    enrollmentCanvasPreview.style.display = 'none'; // Hide canvas after successful enrollment
                    fetchInitialAttendanceLogs(); // Refresh logs after enrollment
                } else {
                    enrollmentMessage.textContent = `Error: ${result.detail || 'Enrollment failed.'}`;
                    enrollmentMessage.style.color = 'red';
                }
            } catch (error) {
                console.error('Enrollment API error:', error);
                enrollmentMessage.textContent = `Network Error: ${error.message}`;
                enrollmentMessage.style.color = 'red';
            } finally {
                enrollBtn.disabled = false; // Re-enable enroll button
            }
        });


        // --- Event Listeners and Initial Load ---
        startMainCameraBtn.addEventListener('click', startMainCamera);
        stopMainCameraBtn.addEventListener('click', stopMainCamera);

        // Initial setup
        document.addEventListener('DOMContentLoaded', () => {
            getConnectedCameras();
            fetchInitialAttendanceLogs(); // Fetch initial logs on load
            updateDashboard({ present_count: 0, total_entries_today: 0 }); // Set initial dashboard values
            stopMainCamera(); // Ensure main camera is stopped and UI reflects it initially
            stopEnrollmentCamera(); // Ensure enrollment camera is hidden initially
        });

        // Listen for camera selection changes (stops current main camera if active)
        cameraSelect.addEventListener('change', () => {
            if (currentMainStream) { 
                stopMainCamera();
            }
            // Also ensure enrollment camera is stopped if selecting a new camera
            stopEnrollmentCamera();
            // Reset enrollment UI when camera selection changes
            enrollmentMessage.textContent = '';
            captureBtn.textContent = 'Capture Face'; // Reset to original text
            confirmCaptureBtn.style.display = 'none';
            enrollBtn.disabled = true;
            capturedImageData = null;
            enrollmentCanvasPreview.style.display = 'none';
        });
