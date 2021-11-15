import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sys

preprocessing = transforms.ToTensor()

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
def preprocess_image(image):
    preproc_img = preprocessing(image)
    normalized_inp = preproc_img.unsqueeze(0)
    return(normalized_inp)
    
def find_instances(image_path, result_path= False):
    # TODO: Prompt user before segmenting further
    
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

    # If PIL Image type passed and not path
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    # If PNG, fix
    image = image.convert('RGB')
    input = preprocess_image(image)

    model.eval()
    output = model(input)[0]
    score_threshold = .95
    
    # Store resulting images here
    result_images = []
    
    # For box count
    index = range(len(output['boxes']))

    for idx, box, score, label in zip(index, output['boxes'], output['scores'], output['labels']):

        if score > score_threshold and label == 1:
            
            # print(f"This is a {COCO_NAMES[label]} (confidence:{score})")
            # print("with bounding boxes: ", box)

            top, right, bottom, left = box.cpu().detach().numpy().astype(int)
            
            # Problem in cropping dimensions caused by portrait versus non-portrait photos
            # TODO: if portrait: flip dimensions
            cropped = image.crop((top, right, bottom, left))
            result_images.append(cropped)

    return result_images


if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        find_instances(sys.argv[1])
        
    if len(sys.argv) == 3:
        find_instances(sys.argv[1], sys.argv[2])