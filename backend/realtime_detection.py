import cv2
import numpy as np
from datetime import datetime
from collections import deque
from keras.models import load_model
import telepot
import os
import time
import sqlite3

class MultiCameraViolenceDetector:
    def __init__(self):
        self.model = None
        self.cameras = {}  # Dictionary to store camera objects
        self.camera_detectors = {}  # Individual detector state per camera
        self.violence_threshold = 0.70
        self.high_confidence_threshold = 0.75
        self.min_consecutive_frames = 5
        self.total_incidents = 0
        self.bot = telepot.Bot('7992159975:AAE1j1SEyGVTqclby0cwLpqvnVwNVUi1GB4')
        
    def init_camera_detector(self, camera_id):
        """Initialize detection state for a specific camera"""
        self.camera_detectors[camera_id] = {
            'prediction_queue': deque(maxlen=45),
            'confidence_queue': deque(maxlen=45),
            'consecutive_violence_frames': 0,
            'current_incident_frames': [],
            'current_incident_start': None
        }
        
    def load_model(self):
        print("[INFO] Loading violence detection model...")
        try:
            self.model = load_model('modelnew.h5')
            print("[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise Exception("Model loading failed. Ensure 'modelnew.h5' exists.")
    
    def detect_violence_in_frame(self, frame, camera_id):
        if self.model is None:
            self.load_model()
        
        if camera_id not in self.camera_detectors:
            self.init_camera_detector(camera_id)
        
        detector = self.camera_detectors[camera_id]
        
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
                detector['consecutive_violence_frames'] += 1
            else:
                detector['consecutive_violence_frames'] = 0
            
            detector['prediction_queue'].append(violence_prediction)
            detector['confidence_queue'].append(confidence)
            
            violence_detected = self._strict_temporal_analysis(camera_id)
            
            return violence_detected, confidence
        except Exception as e:
            print(f"[ERROR] Detection failed for camera {camera_id}: {e}")
            return False, 0.0
    
    def _strict_temporal_analysis(self, camera_id):
        detector = self.camera_detectors[camera_id]
        
        if len(detector['prediction_queue']) < self.min_consecutive_frames:
            return False
        
        recent_predictions = list(detector['prediction_queue'])[-self.min_consecutive_frames:]
        recent_confidences = list(detector['confidence_queue'])[-self.min_consecutive_frames:]
        
        violence_ratio = sum(recent_predictions) / len(recent_predictions)
        avg_confidence = np.mean([conf for conf in recent_confidences if conf > self.violence_threshold]) if any(conf > self.violence_threshold for conf in recent_confidences) else 0
        
        violence_detected = (
            violence_ratio >= 0.6 or
            (avg_confidence >= 0.65 and detector['consecutive_violence_frames'] >= 3) or
            any(conf > 0.8 for conf in recent_confidences[-3:])
        )
        
        return violence_detected
    
    def annotate_frame(self, frame, violence_detected, confidence, camera_id):
        annotated_frame = frame.copy()
        height, width = annotated_frame.shape[:2]
        
        detector = self.camera_detectors.get(camera_id, {})
        consecutive_frames = detector.get('consecutive_violence_frames', 0)
        
        # Status text and color
        status_text = "VIOLENCE DETECTED" if violence_detected else "NO VIOLENCE"
        status_color = (0, 0, 255) if violence_detected else (0, 255, 0)
        
        # Background rectangle
        cv2.rectangle(annotated_frame, (10, 10), (500, 140), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (500, 140), status_color, 3)
        
        # Camera ID
        cam_text = f"Camera: {camera_id}"
        cv2.putText(annotated_frame, cam_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status text
        cv2.putText(annotated_frame, status_text, (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        
        # Confidence
        conf_text = f"Confidence: {confidence:.1%}"
        cv2.putText(annotated_frame, conf_text, (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Consecutive frames
        consec_text = f"Consecutive: {consecutive_frames}"
        cv2.putText(annotated_frame, consec_text, (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated_frame, timestamp, (width-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Violence border
        if violence_detected:
            cv2.rectangle(annotated_frame, (0, 0), (width-1, height-1), (0, 0, 255), 8)
            if int(time.time() * 2) % 2:
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.9, overlay, 0.1, 0)
        
        return annotated_frame
    
    def save_violence_frame(self, frame, confidence, frame_number, camera_id):
        timestamp = datetime.now()
        detector = self.camera_detectors[camera_id]
        
        if (detector['current_incident_start'] is None or 
            (timestamp - detector['current_incident_start']).total_seconds() > 60):
            
            if detector['current_incident_frames']:
                self.send_telegram_alert(camera_id)
            
            detector['current_incident_start'] = timestamp
            detector['current_incident_frames'] = []
            self.total_incidents += 1
        
        if len(detector['current_incident_frames']) >= 3:
            return
        
        try:
            if frame is None or frame.size == 0:
                print(f"[ERROR] Invalid frame from camera {camera_id}")
                return
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_frame)
            if mean_brightness < 10:
                print(f"[WARNING] Dark frame from camera {camera_id} (brightness: {mean_brightness:.1f})")
                return
            
            os.makedirs('evidence_frames', exist_ok=True)
            frame_filename = f'evidence_frames/violence_cam{camera_id}_{timestamp.strftime("%Y%m%d_%H%M%S%f")}_{frame_number}.jpg'
            
            annotated_frame = self.annotate_frame(frame, True, confidence, camera_id)
            
            if annotated_frame is None or annotated_frame.size == 0:
                print(f"[ERROR] Annotated frame invalid for camera {camera_id}")
                return
            
            success = cv2.imwrite(frame_filename, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success and os.path.exists(frame_filename):
                file_size = os.path.getsize(frame_filename)
                if file_size > 1000:
                    detector['current_incident_frames'].append((frame_filename, confidence))
                    print(f"[SAVED] Camera {camera_id} frame {len(detector['current_incident_frames'])}/3 - Confidence: {confidence:.1%}")
                else:
                    print(f"[ERROR] File too small for camera {camera_id}, removing")
                    os.remove(frame_filename)
            else:
                print(f"[ERROR] Failed to write frame for camera {camera_id}")
                
        except Exception as e:
            print(f"[ERROR] Failed to save frame for camera {camera_id}: {e}")
    
    def send_telegram_alert(self, camera_id):
        detector = self.camera_detectors[camera_id]
        if not detector['current_incident_frames']:
            return
        
        try:
            chat_id = '-5096667007'
            frames = detector['current_incident_frames']
            avg_confidence = np.mean([conf for _, conf in frames])
            max_confidence = max([conf for _, conf in frames])
            
            message = f"""🚨 VIOLENCE DETECTED 🚨

📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📹 Camera: {camera_id}
📊 Frames: {len(frames)}
🎯 Max Confidence: {max_confidence:.1%}
📈 Avg Confidence: {avg_confidence:.1%}
📍 Multi-Camera Detection System

Violence detected by AI system."""
            
            self.bot.sendMessage(chat_id, message)
            
            for idx, (frame_path, confidence) in enumerate(frames, 1):
                if os.path.exists(frame_path):
                    file_size = os.path.getsize(frame_path)
                    if file_size > 1000:
                        with open(frame_path, 'rb') as photo:
                            caption = f"Camera {camera_id} - Frame {idx}/{len(frames)} - {confidence:.1%}"
                            self.bot.sendPhoto(chat_id, photo, caption=caption)
                        time.sleep(0.3)
            
            print(f"[ALERT] Telegram alert sent for camera {camera_id} - {len(frames)} frames")
        except Exception as e:
            print(f"[ERROR] Telegram alert failed for camera {camera_id}: {e}")
    
    def discover_cameras(self):
        """Discover available cameras including mobile devices"""
        available_cameras = []
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        print("[INFO] Scanning for cameras...")
        
        for camera_index in range(5):  # Check first 5 camera indices
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(camera_index, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = int(cap.get(cv2.CAP_PROP_FPS))
                            
                            camera_info = {
                                'id': camera_index,
                                'backend': backend,
                                'width': width,
                                'height': height,
                                'fps': fps,
                                'name': f"Camera {camera_index}"
                            }
                            
                            # Try to identify mobile cameras (usually higher resolution)
                            if width >= 1280 and height >= 720:
                                camera_info['name'] = f"Mobile Camera {camera_index}"
                            elif camera_index == 0:
                                camera_info['name'] = "Laptop Camera"
                            
                            available_cameras.append(camera_info)
                            cap.release()
                            break
                    cap.release()
                except Exception as e:
                    if cap:
                        cap.release()
                    continue
        
        return available_cameras
    
    def start_multi_camera_detection(self):
        """Start detection on all available cameras"""
        cameras = self.discover_cameras()
        
        if not cameras:
            print("[ERROR] No cameras found")
            return
        
        print(f"[INFO] Found {len(cameras)} cameras:")
        for cam in cameras:
            print(f"  - {cam['name']}: {cam['width']}x{cam['height']} @ {cam['fps']}fps")
        
        # Initialize cameras
        for cam_info in cameras:
            cam_id = cam_info['id']
            try:
                cap = cv2.VideoCapture(cam_id, cam_info['backend'])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cameras[cam_id] = {
                    'capture': cap,
                    'info': cam_info
                }
                self.init_camera_detector(cam_id)
                print(f"[INFO] Initialized {cam_info['name']}")
            except Exception as e:
                print(f"[ERROR] Failed to initialize camera {cam_id}: {e}")
        
        if not self.cameras:
            print("[ERROR] No cameras could be initialized")
            return
        
        print("[INFO] Starting multi-camera violence detection...")
        print("[INFO] Press 'q' to quit")
        frame_number = 0
        
        try:
            while True:
                all_frames = {}
                
                # Read from all cameras
                for cam_id, cam_data in self.cameras.items():
                    ret, frame = cam_data['capture'].read()
                    if ret and frame is not None:
                        violence_detected, confidence = self.detect_violence_in_frame(frame, cam_id)
                        annotated_frame = self.annotate_frame(frame, violence_detected, confidence, cam_id)
                        
                        if violence_detected:
                            print(f"[DETECTION] Camera {cam_id} - Frame {frame_number} - Confidence: {confidence:.1%}")
                            self.save_violence_frame(frame, confidence, frame_number, cam_id)
                        
                        all_frames[cam_id] = annotated_frame
                
                # Display frames in grid layout
                if all_frames:
                    self.display_camera_grid(all_frames)
                
                frame_number += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n[INFO] Detection stopped by user")
        except Exception as e:
            print(f"[ERROR] Detection error: {e}")
        finally:
            # Send final alerts and cleanup
            for cam_id in self.camera_detectors:
                if self.camera_detectors[cam_id]['current_incident_frames']:
                    self.send_telegram_alert(cam_id)
            
            for cam_data in self.cameras.values():
                cam_data['capture'].release()
            cv2.destroyAllWindows()
            
            print(f"[INFO] Multi-camera detection stopped - Total incidents: {self.total_incidents}")
    
    def display_camera_grid(self, frames):
        """Display multiple camera feeds in a grid layout"""
        if not frames:
            return
        
        num_cameras = len(frames)
        
        if num_cameras == 1:
            cam_id = list(frames.keys())[0]
            cv2.imshow(f"Camera {cam_id}", frames[cam_id])
        elif num_cameras == 2:
            # Side by side
            frame1 = list(frames.values())[0]
            frame2 = list(frames.values())[1]
            
            # Resize frames to same height
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            target_height = min(h1, h2, 400)
            
            frame1_resized = cv2.resize(frame1, (int(w1 * target_height / h1), target_height))
            frame2_resized = cv2.resize(frame2, (int(w2 * target_height / h2), target_height))
            
            combined = np.hstack([frame1_resized, frame2_resized])
            cv2.imshow("Multi-Camera Detection", combined)
        else:
            # Grid layout for more cameras
            grid_frames = []
            for i, (cam_id, frame) in enumerate(frames.items()):
                # Resize each frame
                resized = cv2.resize(frame, (320, 240))
                # Add camera label
                cv2.putText(resized, f"Cam {cam_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                grid_frames.append(resized)
            
            # Create grid
            if len(grid_frames) <= 4:
                if len(grid_frames) <= 2:
                    combined = np.hstack(grid_frames)
                else:
                    top_row = np.hstack(grid_frames[:2])
                    bottom_row = np.hstack(grid_frames[2:4] if len(grid_frames) > 3 else [grid_frames[2], np.zeros_like(grid_frames[2])])
                    combined = np.vstack([top_row, bottom_row])
            else:
                # More than 4 cameras - show first 4
                top_row = np.hstack(grid_frames[:2])
                bottom_row = np.hstack(grid_frames[2:4])
                combined = np.vstack([top_row, bottom_row])
            
            cv2.imshow("Multi-Camera Detection Grid", combined)

# Legacy class for backward compatibility
class RealtimeViolenceDetector(MultiCameraViolenceDetector):
    def __init__(self):
        super().__init__()
        self.prediction_queue = deque(maxlen=45)
        self.confidence_queue = deque(maxlen=45)
        self.consecutive_violence_frames = 0
        self.current_incident_frames = []
        self.current_incident_start = None
        
    def detect_violence_in_frame(self, frame):
        return super().detect_violence_in_frame(frame, 0)
        
    def annotate_frame(self, frame, violence_detected, confidence):
        return super().annotate_frame(frame, violence_detected, confidence, 0)
        
    def save_violence_frame(self, frame, confidence, frame_number):
        return super().save_violence_frame(frame, confidence, frame_number, 0)
    
    def send_telegram_alert(self):
        return super().send_telegram_alert(0)

def run_single_camera_detection():
    """Run detection on single camera (backward compatibility)"""
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
    
    print("[INFO] Starting single camera violence detection")
    print("[INFO] Press 'q' to quit")
    
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
                detector.save_violence_frame(frame, confidence, frame_number)
                print(f"[VIOLENCE] Frame {frame_number} - Confidence: {confidence:.1%} - Count: {violence_count}")
            
            # Display frame
            cv2.imshow('Real-time Violence Detection', display_frame)
            
            # Handle key presses
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_number += 1
    
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
    print("[INFO] Starting Multi-Camera Violence Detection System")
    print("[INFO] Choose detection mode:")
    print("1. Multi-camera detection (recommended)")
    print("2. Single camera detection (legacy)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            detector = MultiCameraViolenceDetector()
            detector.start_multi_camera_detection()
        elif choice == "2":
            run_single_camera_detection()
        else:
            print("[INFO] Invalid choice, starting multi-camera detection")
            detector = MultiCameraViolenceDetector()
            detector.start_multi_camera_detection()
            
    except KeyboardInterrupt:
        print("\n[INFO] Detection stopped by user")
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
    
    print("[INFO] Violence detection system stopped")