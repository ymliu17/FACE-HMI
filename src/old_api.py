import requests
import time
import cv2
import sys
import os
import threading
from pipeline import get_block_fatigue, make_decision

# Configuration
REQUEST_INTERVAL = 0.5  # seconds between API checks
SESSION_EXTERNAL_ID = sys.argv[1]  # Get session ID from command line
WEBSERVER = "https://brainwellnessgamesmasterapi.azurewebsites.net"
#WEBSERVER = "https://brainwellnessgames-dev.azurewebsites.ne/"

# Game identifiers mapping
GAME_IDENTIFIERS = {
    1: "4c9421ae-e979-44f1-8c88-01bf5c195c92",  # Sound Sweeps
    2: "52524eab-69f6-43ad-b680-4ac1ef21b517",  # Target Tracker
    3: "f1365d89-cc5a-415e-a9f7-f5cfe697adff",  # Mixed Signals
    4: "05561BA9-5DB1-4F15-9C99-8090EACFCB42"   # Delayed Task Switching
}

class VideoRecorder(threading.Thread):
    """Thread for continuous video recording"""
    
    def __init__(self, session_id, block_idx, stop_event):
        threading.Thread.__init__(self)
        self.session_id = session_id
        self.block_idx = block_idx
        self.stop_event = stop_event
        self.video_path = os.path.join("Pilot", f"ses-{session_id}", f"block-{block_idx}.mp4")
        os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
        
    def run(self):
        """Main recording loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video device")
            return
            
        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0 if fps <= 0 else fps  # Default to 30 FPS if not detected
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, fps, (frame_width, frame_height))
        
        print(f"Recording started: {self.video_path} ({frame_width}x{frame_height} @ {fps}FPS)")
        
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Frame capture failed")
                    continue
                
                out.write(frame)
                
                # Maintain consistent frame rate
                time.sleep(1.0 / fps)
        finally:
            cap.release()
            out.release()
            print(f"Recording saved: {self.video_path}")

def check_game_status(api_url, interval=REQUEST_INTERVAL):
    """Check if game is ready to start"""
    while True:
        response = requests.get(api_url)
        if response.status_code == 200:
            status = response.json()["isGameReady"]
            if status:
                return True
        time.sleep(interval)

def main():
    print(f"Starting session: {SESSION_EXTERNAL_ID}")
    
    # Initialize session
    session_url = f"{WEBSERVER}/session/GetByExternalIdentifier?externalIdentifier={SESSION_EXTERNAL_ID}"
    session_info = requests.get(session_url).json()
    session_id = session_info["identifier"]
    print(f"Session ID: {session_id}")
    
    # API endpoints
    api_endpoints = {
        'device_status': f"{WEBSERVER}/device/UpdateStatus",
        'save_result': f"{WEBSERVER}/device/SaveExtendedResult",
        'get_block': f"{WEBSERVER}/block/GetLatestForSession?sessionIdentifier={session_id}",
        'game_status': None  # Will be set per-block
    }
    
    # State tracking
    consecutive_low_accuracy = False
    
    while True:
        print("\nStarting new block...")
        
        # Get current block information
        block_info = requests.get(api_endpoints['get_block']).json()
        block_id = block_info["identifier"]
        game_id = block_info["game_FK"]
        block_index = block_info["block_ID"]
        current_level = block_info["startLevel"]
        
        print(f"Block {block_index}: Game {game_id}, Level {current_level}")
        
        # Set up game status URL for this block
        api_endpoints['game_status'] = f"{WEBSERVER}/game/checkstatus?sessionId={block_id}"
        
        # Wait for game to be ready
        print("Waiting for game to be ready...")
        check_game_status(api_endpoints['game_status'])
        print("Game ready - starting recording")
        
        # Start video recording
        stop_recording = threading.Event()
        recorder = VideoRecorder(SESSION_EXTERNAL_ID, block_index, stop_recording)
        recorder.start()
        
        # Notify server device is ready
        requests.put(api_endpoints['device_status'], 
                   json={"isReady": True, "sessionIdentifier": block_id})
        
        # Monitor game status while recording
        try:
            while True:
                status = requests.get(api_endpoints['game_status']).json()
                if not status["isGameReady"]:
                    print("Game session ended")
                    break
                time.sleep(REQUEST_INTERVAL)
        finally:
            # Clean up recording
            stop_recording.set()
            recorder.join()
            
            # Notify server device is no longer ready
            requests.put(api_endpoints['device_status'],
                        json={"isReady": False, "sessionIdentifier": block_id})
        
        # Get game results
        results_url = f"{WEBSERVER}/game/GetResult?blockIdentifier={block_id}"
        game_results = requests.get(results_url).json()
        accuracy = game_results["accuracy"]
        print(f"Block accuracy: {accuracy:.1%}")
        
        # Analyze fatigue
        video_path = os.path.join("Pilot", f"ses-{SESSION_EXTERNAL_ID}", f"block-{block_index}.mp4")
        output_dir = os.path.join("Pilot", f"ses-{SESSION_EXTERNAL_ID}", f"block-{block_index}")
        os.makedirs(output_dir, exist_ok=True)
        
        fatigue_score = get_block_fatigue(video_path, output_dir, "pre_model/model.pth")
        
        if fatigue_score is None:
            print("Warning: Could not calculate fatigue score")
            continue
            
        print(f"Fatigue score: {fatigue_score:.2f}")
        
        # Determine next game/level
        next_game, next_level, consecutive_low_accuracy = make_decision(
            block_index, fatigue_score, accuracy, game_id, current_level, consecutive_low_accuracy
        )
        
        next_game_id = GAME_IDENTIFIERS[next_game]
        print(f"Next: Game {next_game_id}, Level {next_level}")
        
        # Save decision
        requests.put(api_endpoints['save_result'],
                   json={
                       "NextGameIdentifier": next_game_id,
                       "NextLevel": next_level,
                       "BlockIdentifier": block_id
                   })
        
        # Wait for next block to be ready
        print("Waiting for next block...")
        check_game_status(api_endpoints['game_status'])

if __name__ == "__main__":
    main()