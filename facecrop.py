from deepface import DeepFace
import deepface
import numpy as np
import pandas as pd
from deepface.commons.functions import preprocess_face
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import sys

from face_utils import *
from inference import *

def crop_face(img):
    face = find_face(img)
    final =  run(face)
    final = final.convert("RGB")
    final.save("out.jpeg")
    return final


def main():
    if len(sys.argv) == 2:
        img = Image.open(sys.argv[1])
        crop_face(img)


if __name__ == "__main__":
    main()