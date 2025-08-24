# AI Crime Detection System

A comprehensive real-time violence detection system using AI/ML with React frontend and Flask backend.

## Features

### 🎯 Core Functionality
- **Real-time webcam monitoring** - Live violence detection through webcam feed
- **Video file analysis** - Upload and analyze pre-recorded videos
- **Improved accuracy** - Violence only detected after 3-5 seconds of consistent activity
- **Smart frame logging** - Only 2-4 key frames saved per incident (not entire video)
- **Telegram alerts** - Automatic notifications to Telegram groups with incident details

### 🔧 Technical Improvements
- **Enhanced accuracy algorithm** - Requires sustained violence detection (3-5 seconds)
- **Optimized storage** - Selective frame capture instead of full video recording
- **Real-time processing** - Efficient frame-by-frame analysis
- **RESTful API** - Modern API endpoints for frontend integration
- **Database logging** - SQLite database for incident tracking

### 🎨 Modern Frontend
- **React-based UI** - Clean, responsive interface
- **Real-time status** - Live detection status and system health
- **Incident logs** - Historical view of detected incidents
- **Dual mode support** - Webcam and video upload tabs
- **Visual feedback** - Clear indicators for detection status

## Setup Instructions

### Backend Setup
1. Install Python dependencies:
   \`\`\`bash
   pip install -r backend/requirements.txt
   \`\`\`

2. Configure Telegram Bot:
   - Create a Telegram bot via @BotFather
   - Replace the bot token in `backend/app.py`
   - Get your Telegram group ID

3. Add your trained model:
   - Place your `modelnew.h5` model file in the backend directory

4. Run the Flask server:
   \`\`\`bash
   cd backend
   python app.py
   \`\`\`

### Frontend Setup
The React frontend runs automatically in the v0 environment.

### Database Setup
Run the database setup script:
\`\`\`bash
cd scripts
python setup_database.py
\`\`\`

## Configuration

### Telegram Integration
1. Replace `'6679358098:AAFmpDc7o4MwqDywDahyAK0Qq89IVZqNr04'` with your bot token
2. Replace `'your_telegram_group_id'` with your actual group ID

### Detection Parameters
- **Violence Duration Threshold**: 3-5 seconds (90-150 frames at 30 FPS)
- **Confidence Threshold**: 60% (adjustable in code)
- **Frame Capture**: Maximum 4 key frames per incident
- **Smoothing Window**: 15 frames for better accuracy

## API Endpoints

- `POST /api/start_detection` - Start detection process
- `POST /api/stop_detection` - Stop detection process
- `GET /api/detection_status` - Get current status
- `GET /api/detections` - Get detection history

## Usage

1. **Live Webcam**: Click "Start Detection" in the webcam tab
2. **Video Analysis**: Upload a video file and click "Analyze Video"
3. **Monitor Results**: View real-time alerts and detection logs
4. **Telegram Alerts**: Receive automatic notifications with incident details

## Security Notes

- Ensure proper camera permissions are granted
- Keep Telegram bot token secure
- Consider implementing user authentication for production use
- Monitor system resources during continuous detection
