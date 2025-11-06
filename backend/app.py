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
bot = telepot.Bot('7992159975:AAE1j1SEyGVTqclby0cwLpqvnVwNVUi1GB4')  # Replace with your actual token

# Global variables for detection state
detection_active = False
detection_thread = None

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
        self.max_frames_per_incident = 3  # Limit to 3 frames per incident
        self.current_incident_start = None
        self.current_incident_frames = []
        self.last_incident_time = None
        
    def load_model(self):
        print("[INFO] Loading violence detection model...")
        try:
            self.model = load_model('modelnew.h5')
            print("[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = "dummy"
        
    def detect_violence_in_frame(self, frame):
        """Detect violence in a single frame with fast processing"""
        if self.model is None:
            self.load_model()
            
        try:
            if self.model == "dummy":
                prediction = np.random.random()
                violence_detected = prediction > self.violence_threshold
                return violence_detected, prediction
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (128, 128)).astype("float32")
            frame_normalized = frame_resized / 255.0
            
            # Get prediction
            prediction = self.model.predict(np.expand_dims(frame_normalized, axis=0), verbose=0)[0][0]
            self.prediction_queue.append(prediction)
            
            violence_detected = prediction > self.violence_threshold
            
            return violence_detected, prediction
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return False, 0.0
        
    def _send_batched_telegram_alert(self, incident_frames):
        """Send Telegram alert with ALL violence frames from incident"""
        if not incident_frames:
            return
            
        try:
            chat_id = '-5096667007'
            
            # Calculate stats from all frames
            avg_confidence = np.mean([conf for _, conf in incident_frames])
            max_confidence = max([conf for _, conf in incident_frames])
            
            severity = 'CRITICAL' if max_confidence >= 0.8 else 'MEDIUM' if max_confidence >= 0.6 else 'LOW'
            
            message = f"""🚨 VIOLENCE INCIDENT DETECTED 🚨

📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Frames Detected: {len(incident_frames)}
🎯 Max Confidence: {max_confidence:.1%}
📈 Avg Confidence: {avg_confidence:.1%}
⚠️ Severity: {severity}
📍 Location: Security Camera Feed

SecureWatch AI detected violence. Review {len(incident_frames)} critical frames."""
            
            bot.sendMessage(chat_id, message)
            print(f"[INFO] Telegram alert sent - {len(incident_frames)} frames")
            
            # Send all frames with individual scores
            for idx, (frame_path, confidence) in enumerate(incident_frames, 1):
                if frame_path and os.path.exists(frame_path):
                    try:
                        with open(frame_path, 'rb') as photo:
                            frame_severity = 'CRITICAL' if confidence >= 0.8 else 'MEDIUM' if confidence >= 0.6 else 'LOW'
                            caption = f"Frame {idx}/{len(incident_frames)} - Confidence: {confidence:.1%}"
                            bot.sendPhoto(chat_id, photo, caption=caption)
                        print(f"[INFO] Telegram photo {idx}/{len(incident_frames)} sent")
                        time.sleep(0.3)
                    except Exception as photo_error:
                        print(f"[ERROR] Failed to send photo {idx}: {photo_error}")
            
            try:
                conn = sqlite3.connect('crime_detections.db')
                cursor = conn.cursor()
                for frame_path, _ in incident_frames:
                    cursor.execute('UPDATE incidents SET telegram_sent = TRUE WHERE frame_path = ?', 
                                 (os.path.basename(frame_path),))
                conn.commit()
                conn.close()
            except Exception as db_error:
                print(f"[ERROR] Failed to update telegram_sent: {db_error}")
                
        except telepot.exception.TelegramError as e:
            print(f"[ERROR] Telegram API error: {e}")
        except Exception as e:
            print(f"[ERROR] Batched telegram alert failed: {e}")
        
    def log_violence_incident(self, frame, confidence, frame_number):
        """Log violence frames - collect max 3 frames per incident"""
        timestamp = datetime.now()
        
        if (self.current_incident_start is None or 
            (timestamp - self.current_incident_start).total_seconds() > 30):
            # Send previous incident batch if it has frames
            if self.current_incident_frames:
                print(f"[INFO] Incident ended. Sending {len(self.current_incident_frames)} frames")
                self._send_batched_telegram_alert(self.current_incident_frames)
            
            self.current_incident_start = timestamp
            self.current_incident_frames = []
            self.total_incidents += 1
            print(f"[DEBUG] New incident #{self.total_incidents}")
        
        if len(self.current_incident_frames) >= self.max_frames_per_incident:
            print(f"[INFO] Reached 3-frame limit for this incident")
            return
        
        try:
            evidence_dir = os.path.abspath('evidence_frames')
            os.makedirs(evidence_dir, exist_ok=True)
            
            frame_filename = os.path.join(evidence_dir, f'violence_{timestamp.strftime("%Y%m%d_%H%M%S%f")}_{frame_number}.jpg')
            
            if frame is None or frame.size == 0:
                print(f"[ERROR] Invalid frame data")
                return
            
            success = cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if success and os.path.exists(frame_filename):
                file_size = os.path.getsize(frame_filename)
                if file_size > 0:
                    self.current_incident_frames.append((frame_filename, confidence))
                    print(f"[INFO] VIOLENCE FRAME {len(self.current_incident_frames)}/3 - Confidence: {confidence:.1%}")
                    
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
                            'Upload',
                            'Violence',
                            severity,
                            'Active',
                            False
                        ))
                        conn.commit()
                        conn.close()
                        
                    except Exception as e:
                        print(f"[ERROR] Database logging failed: {e}")
                else:
                    os.remove(frame_filename)
        except Exception as e:
            print(f"[ERROR] Frame saving failed: {e}")

def process_video_with_detection(input_video, output_video_path):
    """Process video with frame-by-frame real-time violence detection"""
    detector = ViolenceDetector()
    
    if isinstance(input_video, int):
        cap = cv2.VideoCapture(input_video)
    else:
        cap = cv2.VideoCapture(input_video)
        
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_video}")
        return 0
        
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs('processed_videos', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_number = 0
    skip_frames = 2  # Process every 3rd frame for faster processing
    
    print(f"[INFO] Starting video processing - FPS: {fps}, Size: {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Send final batch
            if detector.current_incident_frames:
                print(f"[INFO] Video ended. Sending final {len(detector.current_incident_frames)} frames")
                detector._send_batched_telegram_alert(detector.current_incident_frames)
            break
            
        if frame_number % (skip_frames + 1) == 0:
            violence_detected, confidence = detector.detect_violence_in_frame(frame)
            
            if violence_detected and confidence > detector.violence_threshold:
                detector.log_violence_incident(frame, confidence, frame_number)
        
        status_text = f"SecureWatch AI | Processing..."
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out.write(frame)
        
        frame_number += 1
        
        if frame_number % 150 == 0:
            print(f"[DEBUG] Processed {frame_number} frames, Incidents: {detector.total_incidents}")
    
    cap.release()
    out.release()
    
    print(f"[INFO] Processing complete - Total incidents: {detector.total_incidents}")
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

@app.route('/api/start_detection', methods=['POST'])
def start_detection():
    global detection_active, detection_thread
    
    data = request.json or {}
    source = data.get('source', 'webcam')
    
    if detection_active:
        return jsonify({'error': 'Detection already active'}), 400
    
    detection_active = True
    input_video = 0 if source == 'webcam' else data.get('video_path', 'your_video.mp4')
    output_video_file = f'processed_videos/live_detection_{int(time.time())}.mp4'
    
    detection_thread = threading.Thread(
        target=process_video_with_detection,
        args=(input_video, output_video_file)
    )
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
    app.run(debug=True, host='0.0.0.0', port=5000)
