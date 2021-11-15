import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys


# Important transformations
toPIL = transforms.ToPILImage()
resize = transforms.Resize(640)
preprocessing = transforms.Compose([transforms.Resize(640), 
                                    transforms.ToTensor()])
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

model_dict = {
    1: 'deeplabv3_resnet50',
    2: 'deeplabv3_resnet101',
    3: 'fcn_resnet50',
    4: 'fcn_resnet101',
    5: 'deeplabv3_mobilenet_v3_large',
    6: 'lraspp_mobilenet_v3_large'}

def initialize_model(name='deeplabv3'):
    if name == 'deeplabv3_resnet50':
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)

    if name == 'deeplabv3_resnet101':
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    if name == 'fcn_resnet50':
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)

    if name == 'fcn_resnet101':
        model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)

    if name == 'deeplabv3_mobilenet_v3_large':
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)

    if name == 'lraspp_mobilenet_v3_large':
        model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)

    return model

def preprocess_image(image):
    preproc_img = preprocessing(image)
    normalized_inp = normalize(preproc_img).unsqueeze(0)
    normalized_inp.requires_grad = True
    return(normalized_inp)

def make_prediction(model, image):
    input = preprocess_image(image)
    model = model.eval()
    output = model(input)['out']
    # print(output.shape, output.min().item(), output.max().item())
    return output

def visualize_prediction(output, nc=21):

    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(output).astype(np.uint8)
    g = np.zeros_like(output).astype(np.uint8)
    b = np.zeros_like(output).astype(np.uint8)

    idx = output == 0
    r[idx] = 255
    g[idx] = 255
    b[idx] = 255

    rgb = np.stack([r, g, b], axis=2)

    return rgb

def mask_image(image, mask, save):
    mask = toPIL(mask).convert("L")
    image = resize(image).convert("RGBA")
    background = Image.new("RGBA", image.size, (255,255,255,0))
    result = Image.composite(background, image, mask)
    
    if save:
        result.save(save)

    return result

def run(image_path, result_path=False, plot=False, model_no=6):
    model = initialize_model(model_dict[model_no])
    
    # In case model initialized outside
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    output = make_prediction(model, image)

    # Find most likely segmentation class for each pixel.
    out_max = torch.argmax(output, dim=1, keepdim=True)
    rgb = visualize_prediction(out_max.detach().cpu().squeeze().numpy())

    # Use mask "rgb" against original image
    result = mask_image(image, rgb, result_path)

    if plot:
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(image)
        axarr[1].imshow(result)
        plt.show()

    # Return final result
    return result

def main():
    # Arguments: image_path, result_path (optional)
    if len(sys.argv) == 3:
        run(sys.argv[1], sys.argv[2])
    else: 
        run(sys.argv[1])

if __name__ == "__main__":
    main()
