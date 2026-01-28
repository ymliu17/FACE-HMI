import torch
import os
import random
import cv2
from custom_retinaface import RetinaFace
from torchvision import transforms
from PIL import Image



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

def detect_face(frame, face_path):
    # Initialize RetinaFace detector with custom model path
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "pre_model", "RetinaFaceWeights.MNET_V2.onnx")
    detector = RetinaFace(model_path=model_path)
    
    # Detect faces in the frame
    faces, landmarks = detector.detect(frame)
    
    try:
        if len(faces) > 0:
            # Get the first face
            face = faces[0]
            # Extract the face region
            x1, y1, x2, y2 = face[:4]  # First 4 values are bounding box coordinates
            face_img = frame[int(y1):int(y2), int(x1):int(x2)]
            # Save the face image
            cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        else:
            print('No face detected')
            cv2.imwrite(face_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f'Error processing face: {e}')
        cv2.imwrite(face_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def extract_frames(video_path, face_outdir, frame_count):
    if video_path.endswith('.mp4'):
        print(f'Processing video: {os.path.basename(video_path)}')

        video = cv2.VideoCapture(video_path)
        fps = round(video.get(cv2.CAP_PROP_FPS))
        interval_secs = 5
        
        while True:
            success, frame = video.read()         
            if not success:
                break

            curr_time = frame_count / fps
            if curr_time % interval_secs == 0:
                face_path = os.path.join(face_outdir, f"{int(curr_time)}.jpg")
                detect_face(frame, face_path)

            frame_count += 1
        
        video.release()
        return frame_count

def infer_score(model_dir, face_outdir):
    model = torch.load(model_dir, map_location=torch.device('cpu'), weights_only=False)['model']
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device) 

    faces = get_faces(face_outdir)
    input_tensor = faces.view(1, -1, 3, 112, 112)
    print(" device = ", device)
    input_tensor = input_tensor.to(device)

    if faces is None:
        print("No faces found for inference.")
        return
    
    with torch.no_grad():
        h_t = torch.zeros(2, 1, 1024).to(device)

        for i in range(input_tensor.size(1)):
                block_faces = input_tensor[:, i, :, :, :]
                output, h_t = model(block_faces, h_t)

    return torch.argmax(output, dim=1).item()

def get_block_fatigue(input_path, face_outdir, model_dir):

    if input_path.endswith('.mp4'):
        frame_count = 0
        extract_frames(input_path, face_outdir, frame_count)
    
    fatigue = infer_score(model_dir, face_outdir)

    return fatigue

def single_block_change(fatigue, accuracy, level): 
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

    if fatigue and accuracy:
        staircase = min(9, staircase + 1) # Ensure staircase doesn't exceed 9
    elif not fatigue and accuracy:
        staircase = min(9, staircase + 1)
        novelty = random.choice([x for x in range(10) if x != novelty])
    elif not fatigue and not accuracy:
        novelty = random.choice([x for x in range(10) if x != novelty])
    elif fatigue and not accuracy:
        pass
    else:
        raise ValueError("Invalid input")
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
    novelty = max(0, min(9, novelty))
    staircase = max(0, min(9, staircase))
    z = max(0, min(9, z))

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
    # mixed signals = .95, MOT = .75, sound sweeper = .9, task 4 = .7
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

def make_decision(block, fatigue, accuracy, game, level, consec=False):
    if isinstance(block, str):
        block = int(block)
    acc = evaluate_accuracy(accuracy, game)
    if block % 3 == 0:
        new_game = three_blocks_change(fatigue, acc)
    else:
        new_game = game
    
    new_level = single_block_change(fatigue, acc, level)
    if not acc: 
        if consec:
            new_level = two_blocks_change(new_level)
        consec = True
    else:   
        consec = False

    return new_game, new_level, consec