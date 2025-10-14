from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
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

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize the Telegram bot with your bot's API token
bot = telepot.Bot('6679358098:AAFmpDc7o4MwqDywDahyAK0Qq89IVZqNr04')  # Replace with your actual token

# Global variables for detection state
detection_active = False
detection_thread = None
last_status = {"violence": False, "confidence": 0.0, "timestamp": None}
latest_frame_filename = None

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
        self.prediction_queue = deque(maxlen=128)
        self.violence_threshold = 0.6
        self.total_incidents = 0
        self.max_frames_per_incident = 4
        self.current_incident_start = None
        self.current_incident_frames = []
        
    def load_model(self):
        print("[INFO] Loading violence detection model...")
        try:
            self.model = load_model('modelnew.h5')
            print("[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            # Create a dummy model for testing
            self.model = "dummy"
        
    def detect_violence_in_frame(self, frame):
        """Detect violence in a single frame and return prediction"""
        if self.model is None:
            self.load_model()
            
        try:
            if self.model == "dummy":
                # For testing without actual model
                prediction = np.random.random()
                violence_detected = prediction > self.violence_threshold
                return violence_detected, prediction
                
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (128, 128)).astype("float32")
            frame_normalized = frame_resized.reshape(128, 128, 3) / 255
            
            # Get prediction
            prediction = self.model.predict(np.expand_dims(frame_normalized, axis=0))[0][0]
            self.prediction_queue.append(prediction)
            
            # Simple threshold check - no smoothing needed for immediate logging
            violence_detected = prediction > self.violence_threshold
            
            return violence_detected, prediction
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return False, 0.0
        
    def _send_telegram_alert(self, timestamp, confidence, frame_path):
        """Send Telegram alert with violence detection details and image"""
        try:
            chat_id = '-949413618'  # Your Telegram group chat ID
            
            # Create alert message
            severity = 'CRITICAL' if confidence >= 0.8 else 'MEDIUM' if confidence >= 0.6 else 'LOW'
            
            message = f"""🚨 VIOLENCE DETECTED 🚨
            
📅 Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
🎯 Confidence: {confidence:.1%}
⚠️ Severity: {severity}
📍 Location: Security Camera Feed
            
SecureWatch AI has detected potential violence. Please review immediately."""
            
            # Send text message first
            bot.sendMessage(chat_id, message)
            print(f"[INFO] Telegram alert sent successfully")
            
            # Send image if frame exists
            if frame_path and os.path.exists(frame_path):
                try:
                    with open(frame_path, 'rb') as photo:
                        bot.sendPhoto(chat_id, photo, caption=f"Violence detected at {confidence:.1%} confidence")
                    print(f"[INFO] Telegram photo sent successfully")
                except Exception as photo_error:
                    print(f"[ERROR] Failed to send photo: {photo_error}")
            
            # Update database to mark telegram as sent
            try:
                conn = sqlite3.connect('crime_detections.db')
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE incidents 
                    SET telegram_sent = TRUE 
                    WHERE frame_path = ? AND timestamp = ?
                ''', (os.path.basename(frame_path), timestamp.strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
                conn.close()
            except Exception as db_error:
                print(f"[ERROR] Failed to update telegram_sent status: {db_error}")
                
        except telepot.exception.TelegramError as e:
            print(f"[ERROR] Telegram API error: {e}")
            print("[TROUBLESHOOTING] Check:")
            print("1. Bot token is correct")
            print("2. Bot is added to the group")
            print("3. Chat ID is correct")
            print("4. Bot has permission to send messages")
        except Exception as e:
            print(f"[ERROR] Telegram alert failed: {e}")
        
    def log_violence_incident(self, frame, confidence, frame_number, video_path=None):
        """Log violence detection with proper frame management - ONLY for violence=True frames"""
        if confidence <= self.violence_threshold:
            print(f"[WARNING] log_violence_incident called with low confidence {confidence:.3f} - skipping")
            return
            
        timestamp = datetime.now()
        
        if (self.current_incident_start is None or 
            (timestamp - self.current_incident_start).total_seconds() > 30):
            self.current_incident_start = timestamp
            self.current_incident_frames = []
            print(f"[DEBUG] Starting new incident period at {timestamp}")
        
        frame_filename = ""
        
        if len(self.current_incident_frames) < self.max_frames_per_incident:
            try:
                # Ensure directory exists with absolute path
                evidence_dir = os.path.abspath('evidence_frames')
                os.makedirs(evidence_dir, exist_ok=True)
                print(f"[DEBUG] Evidence directory: {evidence_dir}")
                
                # Create filename with absolute path
                frame_filename = os.path.join(evidence_dir, f'violence_{timestamp.strftime("%Y%m%d_%H%M%S")}_{frame_number}.jpg')
                print(f"[DEBUG] Attempting to save VIOLENCE frame to: {frame_filename}")
                
                # Validate frame data
                if frame is None or frame.size == 0:
                    print(f"[ERROR] Invalid frame data - frame is None or empty")
                    frame_filename = ""
                else:
                    # Save the frame with high quality
                    success = cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    if success and os.path.exists(frame_filename):
                        # Verify file was actually created and has content
                        file_size = os.path.getsize(frame_filename)
                        if file_size > 0:
                            self.current_incident_frames.append(frame_filename)
                            print(f"[INFO] VIOLENCE FRAME SAVED - Confidence: {confidence:.3f}, File: {frame_filename} ({file_size} bytes)")
                            print(f"[DEBUG] Violence frames in current incident: {len(self.current_incident_frames)}")
                            global latest_frame_filename
                            latest_frame_filename = frame_filename
                        else:
                            print(f"[ERROR] Frame file created but empty: {frame_filename}")
                            os.remove(frame_filename)  # Remove empty file
                            frame_filename = ""
                    else:
                        print(f"[ERROR] cv2.imwrite failed or file not created: {frame_filename}")
                        frame_filename = ""
                        
            except Exception as e:
                print(f"[ERROR] Frame saving exception: {e}")
                frame_filename = ""
        else:
            print(f"[INFO] VIOLENCE DETECTED - Frame limit reached for current incident ({self.max_frames_per_incident})")
        
        if frame_filename:  # Only log incidents that have actual frames saved
            try:
                conn = sqlite3.connect('crime_detections.db')
                cursor = conn.cursor()
                
                # Determine severity based on confidence
                if confidence >= 0.8:
                    severity = 'CRITICAL'
                elif confidence >= 0.6:
                    severity = 'MEDIUM'
                else:
                    severity = 'LOW'
                
                # Store relative path for database
                db_frame_path = os.path.basename(frame_filename)
                
                cursor.execute('''
                    INSERT INTO incidents (timestamp, confidence, frame_path, source, type, severity, status, telegram_sent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    float(confidence),
                    db_frame_path,  # Store just filename, not full path
                    'Upload',
                    'Violence',
                    severity,
                    'Active',
                    False
                ))
                conn.commit()
                conn.close()
                
                self.total_incidents += 1
                print(f"[INFO] VIOLENCE INCIDENT LOGGED - Total incidents: {self.total_incidents}, Frame: {db_frame_path}")
                
                self._send_telegram_alert(timestamp, confidence, frame_filename)
                
            except Exception as e:
                print(f"[ERROR] Database logging failed: {e}")
        else:
            print(f"[INFO] VIOLENCE DETECTED but no frame saved - not logging to database")

def process_video_with_detection(input_video, output_video_path):
    """Process video with immediate violence detection and logging"""
    detector = ViolenceDetector()
    
    # Open video
    if isinstance(input_video, int):
        cap = cv2.VideoCapture(input_video)
        is_webcam = True
    else:
        cap = cv2.VideoCapture(input_video)
        is_webcam = False
        
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_video}")
        return 0
        
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    os.makedirs('processed_videos', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_number = 0
    
    print(f"[INFO] Starting video processing - FPS: {fps}, Size: {width}x{height}")
    print(f"[INFO] VIOLENCE-ONLY MODE: Only frames with violence=True will be saved and logged")
    
    while True:
        # Respect stop for webcam
        if isinstance(input_video, int) and not detection_active:
            print("[INFO] Stop requested - ending live detection loop")
            break

        ret, frame = cap.read()
        if not ret:
            print("[DEBUG] End of video reached")
            break

        violence_detected, confidence = detector.detect_violence_in_frame(frame)

        # Update live status globals (no locking needed for simple reads)
        try:
            last_status = {
                "violence": bool(violence_detected),
                "confidence": float(confidence),
                "timestamp": datetime.now().isoformat(),
                "active": True
            }
        except Exception as _:
            pass

        if violence_detected and confidence > detector.violence_threshold:
            print(f"[DEBUG] VIOLENCE CONFIRMED at frame {frame_number} with confidence {confidence:.3f}")
            detector.log_violence_incident(frame, confidence, frame_number, output_video_path)
        elif violence_detected:
            print(f"[DEBUG] Violence detected but confidence too low: {confidence:.3f} (threshold: {detector.violence_threshold})")
        
        # Add overlay text
        status_text = f"SecureWatch AI | Violence: {violence_detected} | Confidence: {confidence:.2f}"
        color = (0, 0, 255) if violence_detected else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Write frame to output video
        out.write(frame)
        
        frame_number += 1
        
        # Show progress every 100 frames
        if frame_number % 100 == 0:
            print(f"[DEBUG] Processed {frame_number} frames, Violence incidents logged: {detector.total_incidents}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Mark inactive after finishing
    try:
        last_status["active"] = False
    except Exception:
        pass
    
    print(f"[INFO] Video processing complete - Violence incidents detected and logged: {detector.total_incidents}")
    return detector.total_incidents

@app.route('/api/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    # Save uploaded video
    temp_video_path = f'temp_videos/uploaded_{int(time.time())}_{video_file.filename}'
    os.makedirs('temp_videos', exist_ok=True)
    video_file.save(temp_video_path)
    
    # Process video
    output_video_path = f'processed_videos/processed_{int(time.time())}.mp4'
    
    print(f"[INFO] Processing uploaded video: {video_file.filename}")
    incidents_detected = process_video_with_detection(temp_video_path, output_video_path)
    
    # Cleanup temp file
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
    """Get all incidents from database formatted for frontend - ONLY violence incidents with frames"""
    try:
        conn = sqlite3.connect('crime_detections.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM incidents WHERE frame_path IS NOT NULL AND frame_path != "" ORDER BY timestamp DESC LIMIT 50')
        rows = cursor.fetchall()
        conn.close()
        
        incident_groups = {}
        for row in rows:
            timestamp_obj = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
            # Round to nearest 30 seconds for grouping
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
                    'videoPath': None
                }
            
            if row[3] and row[3].strip() and len(incident_groups[timestamp_key]['frames']) < 4:
                evidence_dir = os.path.abspath('evidence_frames')
                full_frame_path = os.path.join(evidence_dir, row[3])
                
                if os.path.exists(full_frame_path):
                    frame_filename = os.path.basename(row[3])
                    frame_url = f"http://localhost:5000/api/frame/{frame_filename}"
                    incident_groups[timestamp_key]['frames'].append(frame_url)
                    print(f"[DEBUG] Added VIOLENCE frame to incident: {frame_filename}")
                else:
                    print(f"[WARNING] Violence frame file not found: {full_frame_path}")
        
        detections = list(incident_groups.values())
        
        print(f"[DEBUG] Returning {len(detections)} violence-only detections to frontend")
        for detection in detections:
            print(f"[DEBUG] Violence Detection {detection['id']}: {len(detection['frames'])} frames, Confidence: {detection['confidence']:.3f}")
        
        return jsonify(detections)
        
    except Exception as e:
        print(f"[ERROR] Failed to get detections: {e}")
        return jsonify([]), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for frontend"""
    return jsonify({
        'status': 'healthy',
        'message': 'SecureWatch AI Backend is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/live_status', methods=['GET'])
def live_status():
    """Endpoint to get the latest status of live detection"""
    global last_status
    try:
        frame_url = None
        if latest_frame_filename and os.path.exists(latest_frame_filename):
            frame_url = f"http://localhost:5000/api/frame/{os.path.basename(latest_frame_filename)}"
        return jsonify({
            "active": bool(last_status.get("active", False)),
            "violence": bool(last_status.get("violence", False)),
            "confidence": float(last_status.get("confidence", 0.0)),
            "timestamp": last_status.get("timestamp"),
            "frame": frame_url
        })
    except Exception as e:
        print(f"[ERROR] live_status failed: {e}")
        return jsonify({"active": False, "violence": False, "confidence": 0.0}), 200

@app.route('/api/latest_frame', methods=['GET'])
def latest_frame():
    """Endpoint to get the latest frame with detected violence"""
    global latest_frame_filename
    if latest_frame_filename and os.path.exists(latest_frame_filename):
        return send_file(latest_frame_filename)
    else:
        return jsonify({'error': 'No latest frame available'}), 404

# API Routes for React frontend
@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    global detection_active, detection_thread, last_status
    data = request.json or {}
    source = data.get('source', 'webcam')

    if detection_active:
        return jsonify({'error': 'Detection already active'}), 400

    detection_active = True
    last_status = {"violence": False, "confidence": 0.0, "timestamp": datetime.now().isoformat(), "active": True}
    input_video = 0 if source == 'webcam' else data.get('video_path', 'your_video.mp4')
    output_video_file = f'processed_videos/live_detection_{int(time.time())}.mp4'

    detection_thread = threading.Thread(target=process_video_with_detection, args=(input_video, output_video_file), daemon=True)
    detection_thread.start()

    return jsonify({'message': 'Detection started', 'status': 'active'})

@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({'message': 'Detection stopped', 'status': 'inactive'})

@app.route('/api/detection_status', methods=['GET'])
def detection_status():
    return jsonify({
        'active': detection_active,
        'detections': []
    })

@app.route('/api/download_video/<video_id>', methods=['GET'])
def download_video(video_id):
    try:
        conn = sqlite3.connect('crime_detections.db')
        cursor = conn.cursor()
        cursor.execute('SELECT frame_path FROM incidents WHERE id = ?', (video_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0] and os.path.exists(row[0]):
            return send_file(row[0], as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/api/frame/<path:filename>')
def serve_frame(filename):
    """Serve frame images for frontend display"""
    try:
        evidence_dir = os.path.abspath('evidence_frames')
        frame_path = os.path.join(evidence_dir, filename)
        print(f"[DEBUG] Serving frame request: {filename}")
        print(f"[DEBUG] Full frame path: {frame_path}")
        
        if os.path.exists(frame_path):
            file_size = os.path.getsize(frame_path)
            print(f"[DEBUG] Frame found and served: {filename} ({file_size} bytes)")
            return send_file(frame_path)
        else:
            print(f"[ERROR] Frame not found: {frame_path}")
            # List available files for debugging
            if os.path.exists(evidence_dir):
                available_files = os.listdir(evidence_dir)
                print(f"[DEBUG] Available files in evidence_frames: {available_files[:10]}")  # Show first 10
            return jsonify({'error': 'Frame not found'}), 404
    except Exception as e:
        print(f"[ERROR] Frame serving failed: {e}")
        return jsonify({'error': 'Frame serving failed'}), 500

# Original Flask routes for backward compatibility
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_crime', methods=['POST'])
def detect_crime():
    global detection_active, detection_thread
    
    if 'source' in request.form:
        source = request.form['source']
        input_video = 0 if source == 'webcam' else 'your_video.mp4'
        output_video_file = 'annotated_video.avi'
        telegram_group_id = '-949413618'
        
        if not detection_active:
            detection_active = True
            detection_thread = threading.Thread(
                target=process_video_with_detection,
                args=(input_video, output_video_file)
            )
            detection_thread.start()

    return redirect(url_for('index'))

if __name__ == '__main__':
    print("[INFO] Starting SecureWatch AI Backend...")
    print("[INFO] Backend will be available at http://localhost:5000")
    print("[INFO] Make sure to update Telegram bot token and group ID")
    app.run(debug=True, host='0.0.0.0', port=5000)
