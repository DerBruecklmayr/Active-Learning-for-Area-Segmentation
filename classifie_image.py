import cv2
import numpy as np
from argparse import ArgumentParser
from typing import List, Tuple
from generate_dataset import get_all_data, get_conjunced_subset, TRANSFORMATION, CLASSES, CLASSES_COLORS

from model import CustomModel

from PIL import Image

from os.path import abspath, join

from Dataloader import DataSetHandler

import torch

from icecream import ic


def expand2square(pil_img) -> Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height))
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    

def display_image(img:np.ndarray, output_values:np.ndarray, output_indices:np.ndarray, colors:List[Tuple[int,int,int]], filename):
    ic(output_values, output_indices)

    mask = np.zeros((output_values.shape[0], output_values.shape[1], 3))

    ic(len(colors), np.min(output_indices), np.max(output_indices))

    for i in range(output_values.shape[0]):
        for j in range(output_values.shape[1]):
            mask[i,j,:] = colors[output_indices[i,j]]

    upscale_mask = cv2.resize(mask, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_NEAREST)
    
    img = cv2.addWeighted(img, 0.8, upscale_mask, 0.3, 0, dtype=cv2.CV_32F)

    path = abspath(join("results","Test", f"{filename}.jpg"))
    cv2.imwrite(path, img)
    print(f'saved image: {path}')

if __name__ == "__main__":
    parser = ArgumentParser(
                    prog='Classifie a Image')
    
    parser.add_argument('-b', type=bool, help="gen classification for test-batch")
    parser.add_argument('-i', '--image', type=str, help="image id")
    parser.add_argument('-s', '--model-size', type=str, default="34")
    parser.add_argument('-w', '--model-weights', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    
    args = parser.parse_args()

    model = CustomModel(args.model_size)
    model.load_state_dict(torch.load(abspath(args.model_weights)))
    model = model.to(args.device)

    data = get_all_data()
    validate = get_conjunced_subset(data['val'], 1, None)

    validate_dataset = np.unique(validate.to_numpy())

    if np.isin(args.image, validate_dataset):
        print(args.image)
        jpg_img = Image.open(abspath(join("VOCdevkit", "VOC2007","JPEGImages", f"{str(args.image).zfill(6)}.jpg"))).convert('RGB')
    
        square = np.array(expand2square(jpg_img))
        img = TRANSFORMATION(square.copy()).to(args.device)

        with torch.no_grad():
            output = torch.squeeze(torch.nn.Softmax2d()(model.forward(torch.unsqueeze(img, dim=0))), dim=0)

            max_output = torch.max(output, dim=0)
            
        display_image(square, max_output.values.cpu().numpy(), max_output.indices.cpu().numpy(), CLASSES_COLORS, str(args.image).zfill(6))
    else:
        print(f"valid images are: {validate_dataset}")
    exit()





