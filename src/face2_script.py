import torch
import os
import random
import cv2
from PIL import Image
import sys
import threading
import platform
import time
import csv
from threading import BrokenBarrierError
from uniface import RetinaFace
from torchvision import transforms

# Camera settings
CAM_INDEX = 0
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
CAM_FPS = 30
CAM_USE_MJPG = True
WRITER_FOURCC = 'mp4v' # macOS use 'avc1' for alternative

# PLUX import
pyver = ''.join(platform.python_version().split('.')[:2])  # e.g., '311', '312'
osDic = {
    "Darwin": f"M1_{pyver}",  # <- use M1_310 / M1_311 / M1_312
    "Linux": "Linux64",
    "Windows": f"Win{platform.architecture()[0][:2]}_{pyver}",
}
sys.path.append(f"./PLUX-API-Python3/{osDic[platform.system()]}")
try:
    import plux # type: ignore
except ImportError:
    print("Warning: PLUX library not found; ECG recording will be disabled.")

class ECGRecorder(threading.Thread):
    """
    Minimal ECG recorder (no meta, no fallbacks).
    Saves: Pilot/ses-<session_id>/block-<block_idx>/ecg_raw.csv
    """

    def __init__(
        self,
        session_id: str,
        block_idx: int,
        stop_event: threading.Event,
        start_barrier: threading.Barrier | None = None,
        address: str = "BTH00:07:80:0F:31:34",
        frequency: int = 1000,
        channels_code: int = 0x01,   # 0x01=1ch, 0x03=2ch, ...
    ):
        super().__init__()
        self.session_id = session_id
        self.block_idx = block_idx
        self.stop_event = stop_event
        self.start_barrier = start_barrier
        self.address = address
        self.frequency = int(frequency)
        self.channels_code = int(channels_code)
        self._rows = []
        self._t0 = None
        self._device = None

        base_dir = os.path.join("Pilot", f"ses-{session_id}", f"block-{block_idx}")
        os.makedirs(base_dir, exist_ok=True)
        self.csv_path = os.path.join(base_dir, "ecg_raw.csv")

    def run(self):
        import plux # type: ignore

        class _Device(plux.SignalsDev):      
            def onRawFrame(self, nSeq, data):
                outer = getattr(self, 'outer', None)
                if outer is None:
                    return True
                if outer._t0 is None:
                    outer._t0 = time.time()
                t_epoch = self.outer._t0 + (nSeq / self.outer.frequency)
                outer._rows.append([nSeq, t_epoch] + list(data))
                return outer.stop_event.is_set()
        
        addr_candidates = [self.address]
        if platform.system() == "Darwin" and self.address.startswith("BTH"):
            addr_candidates.append(self.address[3:])  # add alternative address

        # # device
        # self._device = _Device(self.address)
        # self._device.outer = self

        # optional sync with video
        if self.start_barrier is not None:
            try:
                self.start_barrier.wait(timeout=3)
            except BrokenBarrierError:
                print("ECGRecorder: start barrier broken/timed out; continuing.")

        connected = False
        for addr in addr_candidates:
            try:
                self._device = _Device(addr)
                self._device.outer = self
                attempts = 0
                while attempts < 3:
                    try:
                        # self._device.connect()  # no-arg connect
                        connected = True
                        break
                    except Exception as e:
                        attempts += 1
                        print(f"ECGRecorder: connect attempt {attempts} failed on {addr}: {e}")
                        time.sleep(1)
                if connected:
                    break
            except Exception as e:
                print(f"ECGRecorder: failed to connect to {addr}: {e}")

        if not connected:
            print("ECGRecorder: could not initialize communication port; ECG disabled for this block.")
            return

        try:
            self._t0 = time.time()
            self._device.start(self.frequency, self.channels_code, 16)
            self._device.loop()
        except Exception as e:
            print(f"ECGRecorder: error during recording: {e}")
        finally:
            for fn in ("stop", "disconnect", "close"):
                try:
                    getattr(self._device, fn)()
                except Exception:
                    pass
            # Save CSV only if we captured something
            try:
                if self._rows:
                    with open(self.csv_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["nSeq", "t_epoch"] + [f"ch{i+1}" for i in range(8)])
                        w.writerows(self._rows)
                    print(f"ECG saved: {self.csv_path}")
            except Exception as e:
                print(f"ECGRecorder: failed to write CSV (non-fatal): {e}")


class VideoRecorder(threading.Thread):
    """Thread for continuous video recording"""

    def __init__(self, session_id, block_idx, stop_event, start_barrier = None):
        threading.Thread.__init__(self)
        self.session_id = session_id
        self.block_idx = block_idx
        self.stop_event = stop_event
        self.start_barrier = start_barrier
        self.video_path = os.path.join("Pilot", f"ses-{session_id}", f"block-{block_idx}.mp4")
        os.makedirs(os.path.dirname(self.video_path), exist_ok=True)
        
    def run(self):
        """Main recording loop"""
        if sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION
        elif sys.platform == "win32":
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_V4L2

        # Sync with ECG start
        if self.start_barrier is not None:
            try:
                self.start_barrier.wait(timeout=5)
            except BrokenBarrierError:
                print("VideoRecorder: start barrier broken/timed out; continuing.")
        self.start_epoch = time.time()  # optional: store for downstream alignment

        cap = cv2.VideoCapture(CAM_INDEX, backend)
        if not cap.isOpened():
            print("Error: Could not open video device")
            return
            
        # Get camera properties
        if CAM_USE_MJPG:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(CAM_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(CAM_HEIGHT))
        cap.set(cv2.CAP_PROP_FPS, float(CAM_FPS))

        _ = cap.read()  # Warm up camera

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = cap.get(cv2.CAP_PROP_FPS)

        actual_fps = reported_fps if (reported_fps and reported_fps > 0) else float(CAM_FPS)
        if actual_fps <=0:
            actual_fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*WRITER_FOURCC)
        out = cv2.VideoWriter(self.video_path, fourcc, actual_fps, (actual_w, actual_h))

        print(f"Recording started: {self.video_path} "
            f"requested={CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS} "
            f"got={actual_w}x{actual_h}@{actual_fps:.2f} (reported={reported_fps:.2f})")
        
        try:
            period = 1.0 / actual_fps
            next_t = time.perf_counter()
            while not self.stop_event.is_set():
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(max(0.0, next_t - now))
                next_t += period

                ret, frame = cap.read()
                if not ret:
                    print("Warning: Frame capture failed")
                    continue
                out.write(frame)
                
        finally:
            cap.release()
            out.release()
            print(f"Recording saved: {self.video_path}")


def make_sync_objects():
    """Returns (stop_event, start_barrier)."""
    return threading.Event(), threading.Barrier(2)


def get_faces(block_path, seq_len=16):
    if os.path.exists(block_path):
        face_paths = [os.path.join(block_path, f) for f in sorted(os.listdir(block_path)) if f.lower().endswith('.jpg')]
            
        if not face_paths:
            print(f"No faces found in {block_path}")
            return None
            
        if len(face_paths) < seq_len:
            face_paths += [face_paths[-1]] * (seq_len - len(face_paths))

        seq_interval = (len(face_paths) - 1) // (seq_len - 1)
        faces = []
        for i in range(seq_len):
            if i == seq_len - 1:
                face = Image.open(face_paths[-1])
            else:
                face = Image.open(face_paths[i * seq_interval])
            face = transforms.ToTensor()(face)
            face = transforms.Resize((112, 112), antialias=True)(face)
            faces.append(face)
        
        faces = torch.stack(faces)
        return faces
    else:
        return None

def get_faces_segment(block_path, seq_len=16, start_sec=None, end_sec=None):
    if not os.path.exists(block_path):
        print(f"Block path does not exist: {block_path}")
        return None
    
    all_files = [f for f in sorted(os.listdir(block_path)) if f.lower().endswith('.jpg')]
    if not all_files:
        print(f"No faces found in {block_path}")
        return None

    face_paths = []
    for fname in all_files:
        stem = os.path.splitext(fname)[0]
        try:
            sec = int(stem)
        except ValueError:
            # Skip files that don't have a valid integer as filename
            continue

        if (start_sec is None or sec >= start_sec) and (end_sec is None or sec < end_sec):
            face_paths.append(os.path.join(block_path, fname))

    if not face_paths:
        print(f"No faces in requested segment [{start_sec}, {end_sec}) in {block_path}")
        return None
    
    if len(face_paths) < seq_len:
        face_paths += [face_paths[-1]] * (seq_len - len(face_paths))

    seq_interval = (len(face_paths) - 1) // (seq_len - 1)
    faces = []
    for i in range(seq_len):
        if i == seq_len - 1:
            face = Image.open(face_paths[-1])
        else:
            face = Image.open(face_paths[i * seq_interval])
        face = transforms.ToTensor()(face)
        face = transforms.Resize((112, 112), antialias=True)(face)
        faces.append(face)

    faces = torch.stack(faces)
    return faces    

def detect_face(frame, face_path, detector):
    faces = detector.detect(frame)

    if not faces:
        cv2.imwrite(face_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return
    
    face = faces[0]

    bbox = getattr(face, 'bbox', None)
    if bbox is None or len(bbox) < 4:
        cv2.imwrite(face_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    x1, y1, x2, y2 = bbox[:4]
    # clamp + crop
    h, w = frame.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        cv2.imwrite(face_path, frame)
        return

    face_img = frame[y1:y2, x1:x2]
    cv2.imwrite(face_path, face_img)

def extract_frames(video_path, face_outdir, frame_count, detector):
    if video_path.endswith('.mp4'):
        print(f'Processing video: {os.path.basename(video_path)}')

        video = cv2.VideoCapture(video_path)
        fps = round(video.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30  # Default to 30 if FPS is not available
        interval_secs = 5
        
        while True:
            success, frame = video.read()         
            if not success:
                break

            curr_time = frame_count / fps
            curr_sec = int(curr_time)
            if curr_time % interval_secs == 0:
                face_path = os.path.join(face_outdir, f"{curr_sec}.jpg")
                detect_face(frame, face_path, detector)

            frame_count += 1
        
        video.release()
        return frame_count

def infer_score(model_dir, face_outdir):
    model = torch.load(model_dir, map_location=torch.device('cpu'), weights_only=False)['model']
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device) 

    faces = get_faces(face_outdir)
    if faces is None:
        print("No faces found for inference.")
        return
    input_tensor = faces.view(1, -1, 3, 112, 112)
    print(" device = ", device)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        h_t = torch.zeros(2, 1, 1024).to(device)

        for i in range(input_tensor.size(1)):
                block_faces = input_tensor[:, i, :, :, :]
                output, h_t = model(block_faces, h_t)

    return torch.argmax(output, dim=1).item()

def infer_score_on_faces(model, device, faces):
    """
    Infer fatigue score on given faces tensor.
    """
    if faces is None:
        return None

    input_tensor = faces.view(1, -1, 3, 112, 112).to(device)

    with torch.no_grad():
        h_t = torch.zeros(2, 1, 1024).to(device)
        for i in range(input_tensor.size(1)):
            block_faces = input_tensor[:, i, :, :, :]
            output, h_t = model(block_faces, h_t)

    return torch.argmax(output, dim=1).item()

def get_block_fatigue(input_path, face_outdir, model_dir):

    if input_path.endswith('.mp4'):
        frame_count = 0

        # Initialize the face detector
        detector = RetinaFace(gpu_id=0 if torch.cuda.is_available() else -1)
        os.makedirs(face_outdir, exist_ok=True)
        extract_frames(input_path, face_outdir, frame_count, detector)

    pred = infer_score(model_dir, face_outdir)
    if pred is None:
        print("No faces found for this block.")
        return None
    
    fatigued = bool(pred)
    print(f"model_output={pred}, status={'fatigue' if fatigued else 'not fatigue'}")
    return fatigued

def get_block_fatigue_4x(input_path, face_outdir, model_dir, n_segments=4, cutoff=1):
    """
    Get block fatigue by segmenting the video into n_segments,
    inferring fatigue on each segment, and summing the scores.
    """
    # Extract faces from video
    if input_path.endswith('.mp4'):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        block_duration = int(total_frames / fps) if fps >0 and total_frames >0 else 120
        frame_count = 0
        # Initialize the face detector
        detector = RetinaFace(gpu_id=0 if torch.cuda.is_available() else -1)
        os.makedirs(face_outdir, exist_ok=True)
        extract_frames(input_path, face_outdir, frame_count, detector)
    else:
        block_duration = 120  # default duration if not video

    # Load model
    ckpt = torch.load(model_dir, map_location=torch.device('cpu'), weights_only=False)
    model = ckpt['model']
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Infer fatigue on each segment
    seg_len = max(1, block_duration // n_segments)
    preds = []

    for i in range(n_segments):
        start = i * seg_len
        end = block_duration if i == n_segments - 1 else (i + 1) * seg_len

        faces = get_faces_segment(face_outdir, seq_len=16, start_sec=start, end_sec=end)
        if faces is None:
            print(f"No faces for segment {i} [{start}, {end})")
            continue

        pred = infer_score_on_faces(model, device, faces)
        preds.append(pred)
        print(f"Segment {i} [{start}-{end}] -> {pred}")

    if not preds:
        print("No valid segments for this block.")
        return None

    sum_score = sum(preds)
    fatigued = sum_score >= cutoff
    print(f"Segment preds={preds}, sum={sum_score}, fatigued={fatigued} (cutoff={cutoff})")
    return fatigued


def single_block_change(fatigue, accuracy, level): 
    """
    x = novelty (hundreds digit)
    y = difficulty / staircase (tens digit)
    z = sublevel (ones digit, unchanged here)

    Screenshot policy:
      Acc=High,  Tired=True  -> change x and increase y
      Acc=High,  Tired=False -> increase y
      Acc=Low,   Tired=True  -> change x (random)
      Acc=Low,   Tired=False -> change neither
    """
    # ensure level is an integer  
    if not isinstance(level, int):
        try:
            level = int(level)  # Convert to integer if possible
        except (ValueError, TypeError):
            raise ValueError("level must be an integer (e.g., 540)")

    novelty, staircase, z = level // 100, (level // 10) % 10 , level % 10
    novelty = max(0, min(9, novelty))
    staircase = max(0, min(9, staircase))
    z = max(0, min(9, z))

    if accuracy and fatigue:
        # High acc + tired: change x AND increase y
        staircase = min(9, staircase + 1) 
        novelty = random.choice([x for x in range(10) if x != novelty])

    elif accuracy and not fatigue:
        # High acc + not tired: increase y only
        staircase = min(9, staircase + 1)

    elif (not accuracy) and fatigue:
        # Low acc + tired: change x only
        novelty = random.choice([x for x in range(10) if x != novelty])
    elif (not accuracy) and (not fatigue):
        # Low acc + not tired: change neither
        pass

    else:
        raise ValueError("Invalid input")
    
    new_level = novelty * 100 + staircase * 10 + z
    return new_level

def single_block_increase(accuracy, level):
    """
    x = novelty (hundreds digit)
    y = difficulty / staircase (tens digit)
    z = sublevel (ones digit, unchanged here)

    Screenshot policy:
      Acc=High -> increase y
      Acc=Low ->  change x only

    """
    # ensure level is an integer  
    if not isinstance(level, int):
        try:
            level = int(level)  # Convert to integer if possible
        except (ValueError, TypeError):
            raise ValueError("level must be an integer (e.g., 540)")

    novelty, staircase, z = level // 100, (level // 10) % 10 , level % 10
    novelty = max(0, min(9, novelty))
    staircase = max(0, min(9, staircase))
    z = max(0, min(9, z))

    if accuracy:
        # High acc: increase y only
        staircase = min(9, staircase + 1)
    else:
        # Low acc: change x only
        novelty = random.choice([x for x in range(10) if x != novelty])
    
    new_level = novelty * 100 + staircase * 10 + z
    return new_level


def two_blocks_change(level):
    # ensure level is an integer  
    if not isinstance(level, int):
        try:
            level = int(level)  # Convert to integer if possible
        except (ValueError, TypeError):
            raise ValueError("level must be an integer (e.g., 540)")

    novelty, staircase, z = level // 100, (level // 10) % 10 , level % 10
    # novelty = max(0, min(9, novelty))
    # staircase = max(0, min(9, staircase))
    # z = max(0, min(9, z))

    staircase = max(0, staircase - 1) # Ensure staircase doesn't go below 0
    new_level = novelty * 100 + staircase * 10 + z
    return new_level

def three_blocks_change(fatigue, accuracy): 
    '''
    game_FK are as follows:
    1 - Sound Sweeps
    2 - Target Tracker
    3 - Mixed Signals
    4 - Delayed Task Switching (or Task 4)
    '''
    if fatigue and accuracy:
        new_game = 4
    elif not fatigue and accuracy:
        new_game = 2
    elif fatigue and not accuracy:
        new_game = 3
    else:
        new_game = 1
    return new_game

def evaluate_accuracy(accuracy, game):
    # Sound Sweeps = .95, Target Tracker = .75, Mixed Signals = .8, Delayed Task Switching = .8
    if game == 1 and accuracy >= .9:
        return True
    elif game == 2 and accuracy >= .75:
        return True
    elif game == 3 and accuracy >= .8: # mixed signals to 0.8 temporarily
        return True
    elif game == 4 and accuracy >= .8: # task 4 to 0.8 temporarily
        return True
    else:
        return False

def make_decision(game_streak, fatigue, accuracy, game, level, conseccutive_low_accuracy, use_rotation=True):

    acc = evaluate_accuracy(accuracy, game)

    # automatic game rotation
    if use_rotation:
        new_level = single_block_change(fatigue, acc, level)
        if not acc:
            if conseccutive_low_accuracy:
                new_level = two_blocks_change(new_level)
            conseccutive_low_accuracy = True
        else:
            conseccutive_low_accuracy = False

        new_game = three_blocks_change(fatigue, acc) if game_streak >= 3 else game

    else:
        new_level = single_block_increase(acc, level)
        if not acc:
            if conseccutive_low_accuracy:
                new_level = two_blocks_change(new_level)
            conseccutive_low_accuracy = True
        else:
            conseccutive_low_accuracy = False

        new_game = game % 4 + 1 if game_streak >= 5 else game

    return new_game, new_level, conseccutive_low_accuracy