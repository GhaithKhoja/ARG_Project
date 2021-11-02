import torchvision
import FaceRipper
import torch
import numpy as np
from torchvision import transforms
from pathlib import Path
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN

def ImportedMaskRCNN(num_classes):
    backbone = torchvision.models.mobilenet_v3_large(
        pretrained=True).features

    for param in backbone.parameters():
        param.requires_grad = False

    backbone.out_channels = 960 

    anchor_generator = AnchorGenerator(
        sizes=((64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
        )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
        )
    
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=14,
        sampling_ratio=2
        )
    
    model = MaskRCNN(
        backbone,num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler
        )
    return model

def get_pred(input):
    trans = transforms.Compose([transforms.ToTensor()])
    input = trans(input)
    model_path = Path(FaceRipper.app.config["MODEL_DIR"])
    filename = model_path/'model.pt'
    model = ImportedMaskRCNN(7)
    if torch.cuda.is_available():
        # if cuda gpu is available
        model.load_state_dict(torch.load(filename))
        device = torch.device('cuda:0')
        model.to(device)
        input = input.cuda()
    else:
        # settle for cpu
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    model.eval()
    return model([input])


def drawMask(input, pred):
    pred_score_threshold = 0.1
    mask_noise_threshold = 0.1
    
    dim = input.shape
    dim = (dim[0],dim[1])
    scores = pred['scores']
    masks = pred['masks']

    maskSum = np.zeros(dim)
    for i in range(len(scores)):
        if(scores[i] < pred_score_threshold):
            continue
        # mask
        mask = masks[i].detach().numpy()
        mask = mask.reshape(dim)
        mask = mask * (mask > mask_noise_threshold)
        maskSum += mask
    maskSum = maskSum != 0
    maskSum = maskSum.reshape((dim[0], dim[1], 1))
    mask_img = input * maskSum
    return mask_img
