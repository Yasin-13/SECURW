import cv2
import numpy as np
from datetime import datetime
from collections import deque
from keras.models import load_model
import telepot
import os
import time
import sqlite3

class RealtimeViolenceDetector:
    def __init__(self):
        self.model = None
        self.prediction_queue = deque(maxlen=45)
        self.confidence_queue = deque(maxlen=45)
        self.violence_threshold = 0.70  # Same as video upload
        self.high_confidence_threshold = 0.75
        self.consecutive_violence_frames = 0
        self.min_consecutive_frames = 5  # Same as video upload
        self.current_incident_frames = []
        self.current_incident_start = None
        self.total_incidents = 0
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
        if len(self.prediction_queue) < self.min_consecutive_frames:
            return False
        
        recent_predictions = list(self.prediction_queue)[-self.min_consecutive_frames:]
        recent_confidences = list(self.confidence_queue)[-self.min_consecutive_frames:]
        
        # Same lenient criteria as video upload
        violence_ratio = sum(recent_predictions) / len(recent_predictions)
        avg_confidence = np.mean([conf for conf in recent_confidences if conf > self.violence_threshold]) if any(conf > self.violence_threshold for conf in recent_confidences) else 0
        
        violence_detected = (
            violence_ratio >= 0.6 or  # 60% of frames must be violence OR
            (avg_confidence >= 0.65 and self.consecutive_violence_frames >= 3) or  # Good confidence with 3 consecutive frames OR
            any(conf > 0.8 for conf in recent_confidences[-3:])  # Any high confidence in last 3 frames
        )
        
        return violence_detected
    
    def annotate_frame(self, frame, violence_detected, confidence):
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
        
        # Violence border
        if violence_detected:
            cv2.rectangle(annotated_frame, (0, 0), (width-1, height-1), (0, 0, 255), 8)
            # Flashing effect
            if int(time.time() * 2) % 2:
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.9, overlay, 0.1, 0)
        
        return annotated_frame
    
    def save_violence_frame(self, frame, confidence, frame_number):
        timestamp = datetime.now()
        
        if (self.current_incident_start is None or 
            (timestamp - self.current_incident_start).total_seconds() > 60):
            
            if self.current_incident_frames:
                self.send_telegram_alert()
            
            self.current_incident_start = timestamp
            self.current_incident_frames = []
            self.total_incidents += 1
        
        if len(self.current_incident_frames) >= 3:
            return
        
        try:
            # Validate frame is not empty or black
            if frame is None or frame.size == 0:
                print(f"[ERROR] Invalid frame - frame is None or empty")
                return
            
            # Check if frame is mostly black (corrupted)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_frame)
            if mean_brightness < 10:  # Very dark frame
                print(f"[WARNING] Frame appears to be black/corrupted (brightness: {mean_brightness:.1f})")
                return
            
            os.makedirs('evidence_frames', exist_ok=True)
            frame_filename = f'evidence_frames/violence_{timestamp.strftime("%Y%m%d_%H%M%S%f")}_{frame_number}.jpg'
            
            # Save original frame with annotations
            annotated_frame = self.annotate_frame(frame, True, confidence)
            
            # Validate annotated frame
            if annotated_frame is None or annotated_frame.size == 0:
                print(f"[ERROR] Annotated frame is invalid")
                return
            
            success = cv2.imwrite(frame_filename, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success and os.path.exists(frame_filename):
                # Verify file was written correctly
                file_size = os.path.getsize(frame_filename)
                if file_size > 1000:  # At least 1KB
                    self.current_incident_frames.append((frame_filename, confidence))
                    print(f"[SAVED] Violence frame {len(self.current_incident_frames)}/3 - Confidence: {confidence:.1%} - Size: {file_size} bytes")
                else:
                    print(f"[ERROR] Saved file too small ({file_size} bytes), removing")
                    os.remove(frame_filename)
            else:
                print(f"[ERROR] Failed to write frame to {frame_filename}")
                
        except Exception as e:
            print(f"[ERROR] Failed to save frame: {e}")
    
    def send_telegram_alert(self):
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
                    # Verify file before sending
                    file_size = os.path.getsize(frame_path)
                    if file_size > 1000:  # Valid file size
                        try:
                            with open(frame_path, 'rb') as photo:
                                caption = f"Frame {idx}/{len(self.current_incident_frames)} - {confidence:.1%} - {file_size} bytes"
                                self.bot.sendPhoto(chat_id, photo, caption=caption)
                            print(f"[SENT] Frame {idx} to Telegram - Size: {file_size} bytes")
                        except Exception as photo_error:
                            print(f"[ERROR] Failed to send photo {idx}: {photo_error}")
                    else:
                        print(f"[ERROR] Frame {idx} file too small ({file_size} bytes), skipping")
                    time.sleep(0.3)
                else:
                    print(f"[ERROR] Frame file not found: {frame_path}")
            
            print(f"[ALERT] Telegram alert sent - {len(self.current_incident_frames)} frames")
        except Exception as e:
            print(f"[ERROR] Telegram alert failed: {e}")

def run_realtime_detection():
    detector = RealtimeViolenceDetector()
    
    # Initialize camera
    cap = None
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        for camera_index in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"[INFO] Camera found at index {camera_index} with backend {backend}")
                        break
                    else:
                        cap.release()
                        cap = None
            except:
                if cap:
                    cap.release()
                cap = None
        if cap and cap.isOpened():
            break
    
    if not cap or not cap.isOpened():
        print("[ERROR] No camera available")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("[INFO] Starting real-time violence detection")
    print("[INFO] Press 'q' to quit, 's' to save current frame")
    
    frame_number = 0
    violence_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            # Detect violence
            violence_detected, confidence = detector.detect_violence_in_frame(frame)
            
            # Annotate frame
            display_frame = detector.annotate_frame(frame, violence_detected, confidence)
            
            # Handle violence detection
            if violence_detected:
                violence_count += 1
                # Validate frame before saving
                if frame is not None and frame.size > 0:
                    detector.save_violence_frame(frame, confidence, frame_number)
                    print(f"[VIOLENCE] Frame {frame_number} - Confidence: {confidence:.1%} - Count: {violence_count}")
                else:
                    print(f"[ERROR] Invalid frame detected at frame {frame_number}")
            
            # Display frame
            cv2.imshow('Real-time Violence Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if display_frame is not None and display_frame.size > 0:
                    save_path = f'manual_save_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                    success = cv2.imwrite(save_path, display_frame)
                    if success:
                        file_size = os.path.getsize(save_path)
                        print(f"[SAVED] Manual save: {save_path} - Size: {file_size} bytes")
                    else:
                        print(f"[ERROR] Failed to save manual frame")
                else:
                    print(f"[ERROR] Cannot save - invalid frame")
            
            frame_number += 1
            
            # Debug info every 300 frames
            if frame_number % 300 == 0:
                avg_conf = np.mean(list(detector.confidence_queue)[-10:]) if detector.confidence_queue else 0
                print(f"[DEBUG] Frame {frame_number} - Avg confidence: {avg_conf:.1%} - Violence count: {violence_count}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Detection error: {e}")
    finally:
        # Send final alert if needed
        if detector.current_incident_frames:
            detector.send_telegram_alert()
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Detection stopped - Total violence detections: {violence_count}")

if __name__ == "__main__":
    run_realtime_detection()