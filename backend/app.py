from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, Response
from flask_cors import CORS
import cv2
import time
from datetime import datetime
import numpy as np
from collections import deque
from keras.models import load_model
import telepot
import os
import threading
import base64
import sqlite3
import subprocess
import signal
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize the Telegram bot with your bot's API token
bot = telepot.Bot('7992159975:AAE1j1SEyGVTqclby0cwLpqvnVwNVUi1GB4')  # Replace with your actual token

# Global variables for detection state
detection_active = False
detection_thread = None
opencv_process = None
webcam_detector = None
webcam_cap = None
stream_active = False

def init_database(): 
    conn = sqlite3.connect('crime_detections.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            confidence REAL NOT NULL,
            frame_path TEXT NOT NULL,
            source TEXT DEFAULT 'Upload',
            type TEXT DEFAULT 'Violence',
            severity TEXT DEFAULT 'MEDIUM',
            status TEXT DEFAULT 'Active',
            telegram_sent BOOLEAN DEFAULT FALSE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_database()

class ViolenceDetector:
    def __init__(self):
        self.model = None
        self.prediction_queue = deque(maxlen=45)
        self.confidence_queue = deque(maxlen=45)
        self.violence_threshold = 0.70  # 50% threshold for realtime detection
        self.high_confidence_threshold = 0.85
        self.total_incidents = 0
        self.max_frames_per_incident = 3
        self.current_incident_start = None
        self.current_incident_frames = []
        self.last_incident_time = None
        self.consecutive_violence_frames = 0
        self.min_consecutive_frames = 3  # Reduced for faster response
        self.bot = telepot.Bot('7992159975:AAE1j1SEyGVTqclby0cwLpqvnVwNVUi1GB4')
        
    def load_model(self):
        print("[INFO] Loading violence detection model...")
        try:
            self.model = load_model('modelnew.h5')
            print("[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise Exception("Model loading failed. Ensure 'modelnew.h5' exists.")
        
    def detect_violence_in_frame(self, frame):
        """Enhanced violence detection with improved preprocessing and temporal analysis"""
        if self.model is None:
            self.load_model()
            
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (128, 128)).astype("float32")
            frame_normalized = frame_resized / 255.0
            
            preds = self.model.predict(np.expand_dims(frame_normalized, axis=0), verbose=0)[0]
            
            if len(preds) == 1:
                confidence = float(preds[0])
            else:
                confidence = float(preds[1]) if len(preds) > 1 else float(preds[0])
            
            violence_prediction = 1 if confidence > self.violence_threshold else 0
            
            if violence_prediction == 1:
                self.consecutive_violence_frames += 1
            else:
                self.consecutive_violence_frames = 0
            
            self.prediction_queue.append(violence_prediction)
            self.confidence_queue.append(confidence)
            
            violence_detected = self._strict_temporal_analysis()
            
            return violence_detected, confidence
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return False, 0.0
    

    
    def _strict_temporal_analysis(self):
        """Simple temporal analysis for realtime detection"""
        if len(self.prediction_queue) < self.min_consecutive_frames:
            return False
        
        # Check recent frames for violence
        recent_predictions = list(self.prediction_queue)[-self.min_consecutive_frames:]
        recent_confidences = list(self.confidence_queue)[-self.min_consecutive_frames:]
        
        # Simple criteria for realtime detection
        violence_detected = (
            any(conf > 0.50 for conf in recent_confidences) and  # Any frame > 50%
            sum(recent_predictions) >= 2  # At least 2 of last 3 frames
        )
        
        return violence_detected
        
    def send_telegram_alert(self):
        """Send Telegram alert with violence frames"""
        if not self.current_incident_frames:
            return
        
        try:
            chat_id = '-5096667007'
            avg_confidence = np.mean([conf for _, conf in self.current_incident_frames])
            max_confidence = max([conf for _, conf in self.current_incident_frames])
            
            message = f"""🚨 VIOLENCE DETECTED 🚨

📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Frames: {len(self.current_incident_frames)}
🎯 Max Confidence: {max_confidence:.1%}
📈 Avg Confidence: {avg_confidence:.1%}
📍 Real-time Camera Feed

Violence detected by AI system."""
            
            self.bot.sendMessage(chat_id, message)
            
            for idx, (frame_path, confidence) in enumerate(self.current_incident_frames, 1):
                if os.path.exists(frame_path):
                    with open(frame_path, 'rb') as photo:
                        caption = f"Frame {idx}/{len(self.current_incident_frames)} - {confidence:.1%}"
                        self.bot.sendPhoto(chat_id, photo, caption=caption)
                    time.sleep(0.3)
            
            print(f"[ALERT] Telegram alert sent - {len(self.current_incident_frames)} frames")
        except Exception as e:
            print(f"[ERROR] Telegram alert failed: {e}")
        
    def annotate_frame(self, frame, violence_detected, confidence):
        """Add comprehensive violence indicators to frame"""
        try:
            annotated_frame = frame.copy()
            height, width = annotated_frame.shape[:2]
            
            # Status text and color
            status_text = "VIOLENCE DETECTED" if violence_detected else "NO VIOLENCE"
            status_color = (0, 0, 255) if violence_detected else (0, 255, 0)
            
            # Background rectangle
            cv2.rectangle(annotated_frame, (10, 10), (500, 120), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (10, 10), (500, 120), status_color, 3)
            
            # Status text
            cv2.putText(annotated_frame, status_text, (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
            
            # Confidence
            conf_text = f"Confidence: {confidence:.1%}"
            cv2.putText(annotated_frame, conf_text, (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Consecutive frames
            consec_text = f"Consecutive: {self.consecutive_violence_frames}"
            cv2.putText(annotated_frame, consec_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(annotated_frame, timestamp, (width-150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Violence border and flashing effect
            if violence_detected:
                cv2.rectangle(annotated_frame, (0, 0), (width-1, height-1), (0, 0, 255), 8)
                # Flashing effect
                if int(time.time() * 2) % 2:
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                    annotated_frame = cv2.addWeighted(annotated_frame, 0.9, overlay, 0.1, 0)
            
            return annotated_frame
        except Exception as e:
            print(f"[ERROR] Failed to annotate frame: {e}")
            return frame
    
    def log_violence_incident(self, frame, confidence, frame_number):
        """Log violence frames with visual indicators"""
        timestamp = datetime.now()
        
        if (self.current_incident_start is None or 
            (timestamp - self.current_incident_start).total_seconds() > 60):
            if self.current_incident_frames:
                print(f"[INFO] Incident ended. Sending {len(self.current_incident_frames)} frames")
                self.send_telegram_alert()
            
            self.current_incident_start = timestamp
            self.current_incident_frames = []
            self.total_incidents += 1
            print(f"[DEBUG] New incident #{self.total_incidents}")
        
        if len(self.current_incident_frames) >= self.max_frames_per_incident:
            return
        
        try:
            os.makedirs('evidence_frames', exist_ok=True)
            frame_filename = f'evidence_frames/violence_{timestamp.strftime("%Y%m%d_%H%M%S%f")}_{frame_number}.jpg'
            
            annotated_frame = self.annotate_frame(frame, True, confidence)
            success = cv2.imwrite(frame_filename, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success:
                self.current_incident_frames.append((frame_filename, confidence))
                print(f"[SAVED] Violence frame {len(self.current_incident_frames)}/3 - Confidence: {confidence:.1%}")
                
                # Log to database
                try:
                    conn = sqlite3.connect('crime_detections.db')
                    cursor = conn.cursor()
                    
                    severity = 'CRITICAL' if confidence >= 0.8 else 'MEDIUM' if confidence >= 0.6 else 'LOW'
                    db_frame_path = os.path.basename(frame_filename)
                    
                    cursor.execute('''
                        INSERT INTO incidents (timestamp, confidence, frame_path, source, type, severity, status, telegram_sent)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        float(confidence),
                        db_frame_path,
                        'Webcam',
                        'Violence',
                        severity,
                        'Active',
                        False
                    ))
                    conn.commit()
                    conn.close()
                    
                except Exception as e:
                    print(f"[ERROR] Database logging failed: {e}")
        except Exception as e:
            print(f"[ERROR] Frame saving failed: {e}")

def process_webcam_detection():
    """Enhanced real-time webcam detection with improved sensitivity"""
    global detection_active
    detector = ViolenceDetector()
    
    # Try multiple camera backends and indices
    cap = None
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        for camera_index in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"[INFO] Camera found at index {camera_index} with backend {backend}")
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    cap.release()
                    cap = None
            except Exception as e:
                print(f"[DEBUG] Failed camera {camera_index} with backend {backend}: {e}")
                if cap:
                    cap.release()
                cap = None
        
        if cap and cap.isOpened():
            break
    
    if cap is None or not cap.isOpened():
        print("[ERROR] Could not open any webcam with any backend")
        detection_active = False
        return 0
    
    # Set camera properties for better performance
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid lag
        
        # Verify settings
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Camera settings - Width: {actual_width}, Height: {actual_height}, FPS: {actual_fps}")
    except Exception as e:
        print(f"[WARN] Could not set camera properties: {e}")
    
    print("[INFO] Starting enhanced real-time webcam detection")
    print(f"[INFO] Violence threshold: {detector.violence_threshold:.1%}")
    print(f"[INFO] High confidence threshold: {detector.high_confidence_threshold:.1%}")
    print(f"[INFO] Minimum consecutive frames required: {detector.min_consecutive_frames}")
    
    frame_number = 0
    violence_detections = 0
    last_detection_time = time.time()
    
    consecutive_failures = 0
    max_failures = 10
    
    try:
        while detection_active:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"[ERROR] Too many consecutive failures ({consecutive_failures}), stopping detection")
                    break
                print(f"[ERROR] Failed to read from webcam (attempt {consecutive_failures}/{max_failures})")
                time.sleep(0.2)
                continue
            
            consecutive_failures = 0  # Reset on successful read
                
            violence_detected, confidence = detector.detect_violence_in_frame(frame)
            
            # Annotate frame for display
            annotated_frame = detector.annotate_frame(frame, violence_detected, confidence)
            
            if violence_detected:
                violence_detections += 1
                print(f"[DETECTION] Frame {frame_number} - Confidence: {confidence:.1%} - Consecutive: {detector.consecutive_violence_frames} - Total: {violence_detections}")
                detector.log_violence_incident(frame, confidence, frame_number)
                last_detection_time = time.time()
            
            frame_number += 1
            
            # Enhanced debug output
            if frame_number % 150 == 0:
                recent_confidences = list(detector.confidence_queue)[-10:] if detector.confidence_queue else [0]
                avg_confidence = np.mean(recent_confidences)
                print(f"[DEBUG] Frame {frame_number} - Consecutive: {detector.consecutive_violence_frames} - Avg confidence: {avg_confidence:.1%} - Detections: {violence_detections}")
            
            time.sleep(0.05)  # ~20 FPS to reduce load
    
    except Exception as e:
        print(f"[ERROR] Detection loop error: {e}")
    finally:
        # Send final batch when stopping
        if detector.current_incident_frames:
            print(f"[INFO] Detection stopped. Sending final {len(detector.current_incident_frames)} frames")
            detector.send_telegram_alert()
        
        cap.release()
        print(f"[INFO] Webcam detection stopped:")
        print(f"[INFO] - Total frames processed: {frame_number}")
        print(f"[INFO] - Violence detections: {violence_detections}")
        print(f"[INFO] - Incidents logged: {detector.total_incidents}")
    
    return detector.total_incidents

def process_video_with_detection(input_video, output_video_path):
    """Enhanced video processing with better frame sampling and detection"""
    detector = ViolenceDetector()
    cap = cv2.VideoCapture(input_video)
        
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_video}")
        return 0
        
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs('processed_videos', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_number = 0
    processed_frames = 0
    violence_frames = 0
    
    # Adaptive frame skipping based on FPS
    skip_frames = max(1, fps // 15)  # Process ~15 frames per second
    
    print(f"[INFO] Processing video - FPS: {fps}, Size: {width}x{height}, Total frames: {total_frames}")
    print(f"[INFO] Processing every {skip_frames + 1} frames for efficiency")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every nth frame for detection
        if frame_number % (skip_frames + 1) == 0:
            violence_detected, confidence = detector.detect_violence_in_frame(frame)
            processed_frames += 1
            
            # Enhanced logging
            if processed_frames % 50 == 0:
                progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                print(f"[DEBUG] Progress: {progress:.1f}% - Frame {frame_number}/{total_frames} - Consecutive: {detector.consecutive_violence_frames} - Last confidence: {confidence:.1%}")
            
            if violence_detected:
                violence_frames += 1
                print(f"[DETECTION] Violence at frame {frame_number} - Confidence: {confidence:.1%} - Consecutive: {detector.consecutive_violence_frames} - Total: {violence_frames}")
                detector.log_violence_incident(frame, confidence, frame_number)
        
        out.write(frame)
        frame_number += 1
    
    # Send final batch
    if detector.current_incident_frames:
        print(f"[INFO] Video ended. Sending final {len(detector.current_incident_frames)} frames")
        detector.send_telegram_alert()
    
    cap.release()
    out.release()
    
    print(f"[INFO] Video processing complete:")
    print(f"[INFO] - Total frames: {frame_number}")
    print(f"[INFO] - Processed frames: {processed_frames}")
    print(f"[INFO] - Violence detections: {violence_frames}")
    print(f"[INFO] - Incidents logged: {detector.total_incidents}")
    
    return detector.total_incidents

@app.route('/api/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    temp_video_path = f'temp_videos/uploaded_{int(time.time())}_{video_file.filename}'
    os.makedirs('temp_videos', exist_ok=True)
    video_file.save(temp_video_path)
    
    output_video_path = f'processed_videos/processed_{int(time.time())}.mp4'
    
    print(f"[INFO] Processing uploaded video: {video_file.filename}")
    incidents_detected = process_video_with_detection(temp_video_path, output_video_path)
    
    try:
        os.remove(temp_video_path)
    except:
        pass
    
    return jsonify({
        'message': f'Processing complete! {incidents_detected} incidents detected.',
        'incidents_detected': incidents_detected,
        'success': True
    })

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get all violence incidents from database"""
    try:
        conn = sqlite3.connect('crime_detections.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM incidents WHERE frame_path IS NOT NULL AND frame_path != "" ORDER BY timestamp DESC LIMIT 50')
        rows = cursor.fetchall()
        conn.close()
        
        incident_groups = {}
        for row in rows:
            timestamp_obj = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
            seconds = timestamp_obj.second
            rounded_seconds = (seconds // 30) * 30
            timestamp_key = timestamp_obj.replace(second=rounded_seconds, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
            
            if timestamp_key not in incident_groups:
                incident_groups[timestamp_key] = {
                    'id': str(row[0]),
                    'timestamp': row[1],
                    'confidence': float(row[2]),
                    'frames': [],
                    'source': row[4],
                    'type': row[5],
                    'severity': row[6],
                    'status': row[7],
                }
            
            if row[3] and row[3].strip() and len(incident_groups[timestamp_key]['frames']) < 3:
                evidence_dir = os.path.abspath('evidence_frames')
                full_frame_path = os.path.join(evidence_dir, row[3])
                
                if os.path.exists(full_frame_path):
                    frame_filename = os.path.basename(row[3])
                    frame_url = f"http://localhost:5000/api/frame/{frame_filename}"
                    incident_groups[timestamp_key]['frames'].append(frame_url)
        
        detections = list(incident_groups.values())
        
        print(f"[DEBUG] Returning {len(detections)} incidents to frontend")
        
        return jsonify(detections)
        
    except Exception as e:
        print(f"[ERROR] Failed to get detections: {e}")
        return jsonify([]), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'SecureWatch AI Backend is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test_camera', methods=['GET'])
def test_camera():
    """Test camera availability and functionality"""
    results = []
    backends = [
        (cv2.CAP_DSHOW, 'DirectShow'),
        (cv2.CAP_MSMF, 'Media Foundation'),
        (cv2.CAP_ANY, 'Any Available')
    ]
    
    for backend_id, backend_name in backends:
        for camera_index in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(camera_index, backend_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        results.append({
                            'camera_index': camera_index,
                            'backend': backend_name,
                            'status': 'working',
                            'width': int(width),
                            'height': int(height),
                            'fps': int(fps)
                        })
                    else:
                        results.append({
                            'camera_index': camera_index,
                            'backend': backend_name,
                            'status': 'opened_but_no_frame'
                        })
                    cap.release()
                else:
                    results.append({
                        'camera_index': camera_index,
                        'backend': backend_name,
                        'status': 'failed_to_open'
                    })
            except Exception as e:
                results.append({
                    'camera_index': camera_index,
                    'backend': backend_name,
                    'status': 'error',
                    'error': str(e)
                })
    
    working_cameras = [r for r in results if r['status'] == 'working']
    
    return jsonify({
        'working_cameras': len(working_cameras),
        'total_tested': len(results),
        'cameras': results,
        'recommendation': 'Use DirectShow backend if available' if any(r['backend'] == 'DirectShow' and r['status'] == 'working' for r in results) else 'Try external USB camera if built-in camera fails'
    })

def generate_webcam_stream():
    """Generate annotated webcam stream for frontend"""
    global webcam_detector, webcam_cap, stream_active
    
    if not webcam_cap or not webcam_detector:
        return
    
    frame_number = 0
    
    while stream_active:
        try:
            ret, frame = webcam_cap.read()
            if not ret:
                break
            
            # Detect violence and annotate frame
            violence_detected, confidence = webcam_detector.detect_violence_in_frame(frame)
            annotated_frame = webcam_detector.annotate_frame(frame, violence_detected, confidence)
            
            # Log violence if detected
            if violence_detected:
                webcam_detector.log_violence_incident(frame, confidence, frame_number)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_number += 1
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"[ERROR] Stream generation error: {e}")
            break

@app.route('/api/video_stream')
def video_stream():
    """Video streaming route for frontend"""
    return Response(generate_webcam_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    global opencv_process
    
    if opencv_process and opencv_process.poll() is None:
        return jsonify({'error': 'Detection already active'}), 400
    
    # Quick camera test before starting
    try:
        test_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not test_cap.isOpened():
            test_cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            test_cap.release()
            if not ret or frame is None:
                return jsonify({'error': 'Camera found but cannot read frames. Check camera permissions.'}), 400
        else:
            return jsonify({'error': 'No camera detected. Please check camera connection and permissions.'}), 400
    except Exception as e:
        return jsonify({'error': f'Camera test failed: {str(e)}'}), 400
    
    try:
        # Start the OpenCV detection window using current Python interpreter
        opencv_process = subprocess.Popen(
            [sys.executable, 'realtime_detection.py'],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        print("[INFO] OpenCV real-time detection window started")
        return jsonify({
            'message': 'OpenCV detection window started - Check your screen for the detection window', 
            'status': 'active',
            'pid': opencv_process.pid
        })
    except Exception as e:
        return jsonify({'error': f'Failed to start OpenCV detection: {str(e)}'}), 500

@app.route('/api/start_opencv_detection', methods=['POST'])
def start_opencv_detection():
    global opencv_process
    
    if opencv_process and opencv_process.poll() is None:
        return jsonify({'error': 'OpenCV detection already running'}), 400
    
    try:
        # Start the standalone OpenCV detection script using current Python interpreter
        opencv_process = subprocess.Popen(
            [sys.executable, 'realtime_detection.py'],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        print("[INFO] OpenCV real-time detection window started")
        return jsonify({
            'message': 'OpenCV detection window started', 
            'status': 'active',
            'pid': opencv_process.pid
        })
    except Exception as e:
        return jsonify({'error': f'Failed to start OpenCV detection: {str(e)}'}), 500

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    global opencv_process
    
    if not opencv_process or opencv_process.poll() is not None:
        return jsonify({'error': 'No detection active'}), 400
    
    try:
        # Terminate the OpenCV process
        opencv_process.terminate()
        opencv_process.wait(timeout=5)
        
        print("[INFO] OpenCV detection window stopped")
        return jsonify({'message': 'OpenCV detection window closed', 'status': 'inactive'})
    except subprocess.TimeoutExpired:
        # Force kill if it doesn't terminate gracefully
        opencv_process.kill()
        opencv_process.wait()
        return jsonify({'message': 'OpenCV detection force stopped', 'status': 'inactive'})
    except Exception as e:
        return jsonify({'error': f'Failed to stop OpenCV detection: {str(e)}'}), 500

@app.route('/api/stop_opencv_detection', methods=['POST'])
def stop_opencv_detection():
    global opencv_process
    
    if not opencv_process or opencv_process.poll() is not None:
        return jsonify({'error': 'No OpenCV detection running'}), 400
    
    try:
        # Terminate the OpenCV process
        opencv_process.terminate()
        opencv_process.wait(timeout=5)
        
        print("[INFO] OpenCV detection window stopped")
        return jsonify({'message': 'OpenCV detection stopped', 'status': 'inactive'})
    except subprocess.TimeoutExpired:
        # Force kill if it doesn't terminate gracefully
        opencv_process.kill()
        opencv_process.wait()
        return jsonify({'message': 'OpenCV detection force stopped', 'status': 'inactive'})
    except Exception as e:
        return jsonify({'error': f'Failed to stop OpenCV detection: {str(e)}'}), 500

@app.route('/api/detection_status', methods=['GET'])
def detection_status():
    global opencv_process
    
    opencv_running = opencv_process and opencv_process.poll() is None
    
    return jsonify({
        'active': opencv_running,
        'opencv_active': opencv_running,
        'opencv_status': 'running' if opencv_running else 'stopped',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/frame/<path:filename>')
def serve_frame(filename):
    """Serve frame images for frontend display"""
    try:
        evidence_dir = os.path.abspath('evidence_frames')
        frame_path = os.path.join(evidence_dir, filename)
        
        if os.path.exists(frame_path):
            return send_file(frame_path)
        else:
            return jsonify({'error': 'Frame not found'}), 404
    except Exception as e:
        print(f"[ERROR] Frame serving failed: {e}")
        return jsonify({'error': 'Frame serving failed'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("[INFO] Starting SecureWatch AI Backend...")
    print("[INFO] Backend will be available at http://localhost:5000")
    print("[INFO] Test camera at: http://localhost:5000/api/test_camera")
    print("[INFO] OpenCV detection: POST /api/start_opencv_detection")
    app.run(debug=True, host='0.0.0.0', port=5000)
