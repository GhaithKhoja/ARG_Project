from PIL import Image
import time
from inference import model_dict, initialize_model, make_prediction

def main():

    image = Image.open('./sample.jpg')

    for model_name in model_dict.values():
        model = initialize_model(model_name)

        start = time.time()
        output = make_prediction(model, image)
        end = time.time()
        print(f"Inference process took {end - start} seconds using model: {model_name}")

if __name__ == "__main__":
    main()