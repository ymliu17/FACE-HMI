import os
import sys
from uniface import RetinaFace as UniFaceRetinaFace

class RetinaFace:
    """
    Custom wrapper for UniFace RetinaFace that uses a custom model path.
    """
    def __init__(self, model_path=None):
        # Set the environment variable to use our custom model path
        if model_path:
            os.environ['UNIFACE_MODEL_PATH'] = model_path
        
        # Initialize the original RetinaFace
        self.detector = UniFaceRetinaFace()
    
    def detect(self, image):
        """
        Detect faces in the image.
        """
        return self.detector.detect(image) 