from deepface import DeepFace
import deepface
import numpy as np
import pandas as pd
from PIL import Image
from deepface.commons.functions import preprocess_face

BACKENDS = ["retinaface", "mtcnn", "ssd"]

# Action = 'race' or 'gender' or 'age' or 'emotion'
def face_information(image_path, action):
    input = np.array(image_path)

    try:
        info = DeepFace.analyze(input, actions=[action], enforce_detection=True)

    except ValueError:
        for b in BACKENDS:
            print(f"error occured, trying different backend: {b}")
            try:
                info = DeepFace.analyze(input, actions=[action], detector_backend=b)
            except ValueError:
                print("*changing backend")

        # If no face detected after exhausting all backends, disable enforce detection
        info = DeepFace.analyze(input, actions=[action], enforce_detection=False)

    if action not in ['gender', 'age']:
        return max(info[action], key=info[action].get)
    else:
        return info[action]

# Input: PIL image
# Output: Bounding box in Dict format
def detect_face(image, backend='opencv'):

    input = np.array(image)
    try:
        result = preprocess_face(input, detector_backend=backend, align=False, return_region=True)
        
    except ValueError:
        for b in BACKENDS:
            print(f"error occured, trying different backend: {b}")
            try:
                result = preprocess_face(input, detector_backend=backend, align=False, return_region=True)
                return result[1]
            except ValueError:
                print("*changing backend.")
        
        # If no face detected after exhausting all backends disable enforce detection
        result = preprocess_face(input, align=False, return_region=True, enforce_detection=False)

    return result[1]

# Input: PIL image
# Output cropped PIL image
def find_face(image):
    result = detect_face(image)

    FACTOR = 1.5
    x = result[0]/FACTOR
    y = result[1]/(FACTOR+1.5)
    w = result[2]*FACTOR
    h = result[3]*(FACTOR)

    result = image.crop((x, y, w+x, h+y))
    return result

def main():
    pass


if __name__ == "__main__":
    main()
  