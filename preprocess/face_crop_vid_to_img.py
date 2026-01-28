import cv2
import os
import time
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from custom_retinaface import RetinaFace


##orginal#####
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

        start_time = time.time()

        video = cv2.VideoCapture(video_path)
        fps = round(video.get(cv2.CAP_PROP_FPS))
        interval_secs = 5
        
        while True:
            # read in the frame
            success, frame = video.read()         
            if not success:
                break

            # detect face every interval_secs
            curr_time = frame_count / fps
            if curr_time % interval_secs == 0:
                face_path = os.path.join(face_outdir, f"{int(curr_time)}.jpg")
                detect_face(frame, face_path)

            frame_count += 1
        
        video.release()

        end_time = time.time()
        time_cost = end_time - start_time
        print(f'Time cost for processing {os.path.basename(video_path)}: {time_cost:.2f} seconds')

        return frame_count

def main():
    root_dir = '/Users/yangliu/Library/CloudStorage/Box-Box/CogT_Lab_Management/Data Management/Data Collection Data Storage/FACE Phase 1/Intervention Sessions/Video Data'

    out_dir = './prep_image'
    os.makedirs(out_dir, exist_ok=True)

    # Loop through each directory and file in the video data directory
    for subject in sorted(os.listdir(root_dir)):
        subject_path = os.path.join(root_dir, subject)

        if os.path.isdir(subject_path) and subject == 'F4008':
            for session in sorted(os.listdir(subject_path)):
                session_path = os.path.join(subject_path, session)
                if os.path.isdir(session_path):
                    # Initialize frame count
                    frame_count = 0
                    # Create a directory to store the faces
                    face_outdir = os.path.join(out_dir, subject, session)
                    os.makedirs(face_outdir, exist_ok=True)

                    for video in sorted(os.listdir(session_path)):
                        video_path = os.path.join(session_path, video)
                        extract_frames(video_path, face_outdir, frame_count)

if __name__ == '__main__':
    main()
