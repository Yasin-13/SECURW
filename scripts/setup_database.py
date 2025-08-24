import sqlite3
from datetime import datetime

def create_detection_database():
    """Create database to store detection logs"""
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
    
    # Create system_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_level TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database created successfully with incidents table!")

if __name__ == "__main__":
    create_detection_database()
