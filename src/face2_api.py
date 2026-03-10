import requests
import time
import sys
import os
import random
from face2_script import get_block_fatigue, get_block_fatigue_4x, make_decision
from face2_script import VideoRecorder, ECGRecorder, make_sync_objects

# Configuration
REQUEST_INTERVAL = 0.5  # seconds between API checks
SESSION_EXTERNAL_ID = sys.argv[1]  # Get session ID from command line
GROUP_ARM_ID = sys.argv[2]  # 0 for control group, 1 for FACE group
USE_ROTATION = GROUP_ARM_ID in ("1","true","True","t","yes","y")
FATIGUE_N_SEGMENTS = 4  # number of segments to split video into for fatigue analysis
FATIGUE_CUTOFF = 1  # fatigue score threshold
DEFAULT_MODEL_PATH = "pre_model/model_9.pth" # path to default fatigue model
MODEL_DIR = "pre_model"
# WEBSERVER = "https://brainwellnessgamesmasterapi.azurewebsites.net"
WEBSERVER = "http://localhost:8080"
ECG_DEVICE_ADDRESS = "00:07:80:0F:31:34"  # replace with your PLUX address

# Game identifiers mapping
GAME_IDENTIFIERS = {
    1: "4c9421ae-e979-44f1-8c88-01bf5c195c92",  # Sound Sweeps
    2: "52524eab-69f6-43ad-b680-4ac1ef21b517",  # Target Tracker
    3: "f1365d89-cc5a-415e-a9f7-f5cfe697adff",  # Mixed Signals
    4: "05561BA9-5DB1-4F15-9C99-8090EACFCB42"   # Delayed Task Switching
}

def check_game_status(api_url, interval=REQUEST_INTERVAL):
    """Check if game is ready to start"""
    while True:
        response = requests.get(api_url)
        if response.status_code == 200:
            if response.json()["isGameReady"]:
                    return True
        time.sleep(interval)

def parse_subject_id(session_external_id: str) -> str:
    """
    SESSION_EXTERNAL_ID format: 'subidx_sesidx'
    """
    if not session_external_id or "_" not in session_external_id:
        return ""
    return session_external_id.split('_', 1)[0]

def select_model_for_subject(subject_id: str, model_dir: str = MODEL_DIR, default_model: str = DEFAULT_MODEL_PATH) -> str:
    """
    Naming rule: model_<subidx>.pth
    """
    subject_model = os.path.join(model_dir, f"model_{subject_id}.pth")

    if subject_id and os.path.isfile(subject_model):
        print(f"[Model] Using subject-specific model: {subject_model}")
        return subject_model

    print(f"[Model] Subject model not found, using default: {default_model}")
    return default_model

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
    total_blocks = 0 
    prev_game_id = None
    current_game_streak = 0
    played_games = set()
    
    while total_blocks < 20:
        print("\nStarting new block...")
        
        # Get current block information
        block_info = requests.get(api_endpoints['get_block']).json()
        block_id = block_info["identifier"]
        game_id = block_info["game_FK"]
        block_index = block_info["block_ID"]
        current_level = block_info["startLevel"]
        
        print(f"Block {block_index}: Game {game_id}, Level {current_level}")

        total_blocks += 1
        if prev_game_id is None or game_id != prev_game_id:
            current_game_streak = 1
            played_games.add(game_id)
            prev_game_id = game_id
        else:
            current_game_streak += 1

        print(f"[Session Stats] total_blocks={total_blocks}, streak={current_game_streak}, played={sorted(list(played_games))}")
        
        # Set up game status URL for this block
        api_endpoints['game_status'] = f"{WEBSERVER}/game/checkstatus?sessionId={block_id}"
        
        # Notify server device is ready
        requests.put(api_endpoints['device_status'], 
                   json={"isReady": True, "sessionIdentifier": block_id})

        # Wait for game to be ready
        print("Waiting for game to be ready...")
        check_game_status(api_endpoints['game_status'])
        print("Game ready - starting recording")

        # build shared sync objects
        stop_event, start_barrier = make_sync_objects()

        # start video
        video = VideoRecorder(
            session_id=SESSION_EXTERNAL_ID,
            block_idx=block_index,
            stop_event=stop_event,
            start_barrier=start_barrier
        )
        video.start()

        # start ECG
        try:
            ecg = ECGRecorder(
                session_id=SESSION_EXTERNAL_ID, 
                block_idx=block_index,
                stop_event=stop_event, 
                start_barrier=start_barrier, 
                address=ECG_DEVICE_ADDRESS,
                frequency=1000, 
                channels_code=0xFF)
            ecg.start()
        except Exception as e:
            ecg = None
            print(f"Warning: Could not start ECG recording: {e}")
        
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
            stop_event.set()
            video.join()
            ecg.join() if ecg else None
            
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
        
        subject_id = parse_subject_id(SESSION_EXTERNAL_ID)
        model_path = select_model_for_subject(subject_id)

        # fatigue_score = get_block_fatigue(video_path, output_dir, MODEL_PATH)
        fatigue_flag = get_block_fatigue_4x(video_path, 
                                            output_dir, 
                                            model_dir=model_path,
                                            n_segments=FATIGUE_N_SEGMENTS, cutoff=FATIGUE_CUTOFF)

        if fatigue_flag is None:
            print("Warning: Could not calculate fatigue flag, defaulting to not fatigued")
            fatigue_flag = False

        print(f"Fatigue status: {fatigue_flag}")

        # Determine next game/level
        next_game, next_level, consecutive_low_accuracy = make_decision(
            block_index, fatigue_flag, accuracy, game_id, current_level, consecutive_low_accuracy, use_rotation=USE_ROTATION
        )

        # game check after 10/15 blocks
        all_games = {1, 2, 3, 4}
        unplayed_games = list(all_games - played_games)

        if current_game_streak < 3:
            next_game = game_id  # Continue same game if streak < 3 
        elif 10 <= total_blocks < 15 and len(unplayed_games) >= 2:
            # after 10 blocks
            next_game = random.choice(unplayed_games)
        elif total_blocks >= 15 and len(unplayed_games) >= 1:
            # after 15 blocks
            next_game = random.choice(unplayed_games)

        # if current_game_streak >= 1:
        #     all_games = {1,2,3,4}
        #     unplayed_games = list(all_games - played_games)
        #     if unplayed_games:
        #         next_game = random.choice(unplayed_games)
        #     else:
        #         next_game = game_id
        
        next_game_id = GAME_IDENTIFIERS[next_game]
        print(f"Next: Game {next_game_id}, Level {next_level}")
        
        # Save decision
        requests.put(api_endpoints['save_result'],
                   json={
                       "NextGameIdentifier": next_game_id,
                       "NextLevel": next_level,
                       "BlockIdentifier": block_id
                   })
        
        time.sleep(1)  # brief pause before next block

        if total_blocks < 20:
            next_block_info = requests.get(api_endpoints['get_block']).json()
            api_endpoints['game_status'] = f"{WEBSERVER}/game/checkstatus?sessionId={next_block_info['identifier']}"

            # Wait for next block to be ready
            print("Waiting for next block...")
            check_game_status(api_endpoints['game_status'])


    print("Session complete.")

if __name__ == "__main__":
    main()