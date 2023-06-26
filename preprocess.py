from typing import Tuple, Dict, List
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from os.path import abspath, join
from icecream import ic
import numpy as np
from generate_dataset import get_all_data, get_conjunced_subset, CLASSES, CLASSES_COLORS


def get_image_square_with_bounding_boxes(image_number:str, classes:List[str]) -> Tuple[np.array, List[Tuple[str, Tuple[int,int,int,int]]]]:
    tree = ET.parse(abspath(join("VOCdevkit", "VOC2007","Annotations", f"{image_number}.xml")))
    root = tree.getroot()

    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    jpg_img = Image.open(abspath(join("VOCdevkit", "VOC2007","JPEGImages", f"{image_number}.jpg"))).convert('RGB')

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

    square = np.array(expand2square(jpg_img))

    output = (square, [])

    for child in root.findall('object'):
        object_class = child.find("name").text
        if object_class in classes:
            box = child.find("bndbox")
            box = child.find("bndbox")

            xmin = max(int(box.find("xmin").text), 0) + int((square.shape[0]-width)/2)
            ymin = max(int(box.find("ymin").text), 0) + int((square.shape[0]-height)/2)
            xmax = min(int(box.find("xmax").text), width)+ int((square.shape[0]-width)/2)
            ymax = min(int(box.find("ymax").text), height) + int((square.shape[0]-height)/2)

            output[1].append((object_class, (xmin, ymin, xmax, ymax)))

    return output


def convert_objects_to_np_array_with_background_full(image_resolution:Tuple[int,int], objects: Tuple[str, Tuple[int,int,int,int]], classes:List[str], output_resolution:Tuple[int,int]) -> np.array:
    output = np.zeros((len(classes) + 1, output_resolution[0], output_resolution[1]))
    
    output[-1, :,:] = np.invert(np.sum(output, 0).astype(np.bool_)).astype(int)
    
    for o in objects:
        # calc positions of objects
        channel = np.zeros((image_resolution[0], image_resolution[1]))
        channel = cv2.rectangle(channel, (o[1][0], o[1][1]), (o[1][2], o[1][3]), color=1, thickness=-1)
        output[classes.index(o[0]), :,:] += cv2.resize(channel, output_resolution) 
        
        # set background for center of object to 0
        x = int(((o[1][0] + (o[1][2] - o[1][0])/2)/image_resolution[0]) * output_resolution[0])
        y = int(((o[1][1] + (o[1][3] - o[1][1])/2)/image_resolution[1]) * output_resolution[1])
        
        output[-1, y, x] = 0

    return output.astype(np.bool_).astype(np.int_)


def convert_object_to_np_array_center_with_background_full(image_resolution:Tuple[int,int], objects: Tuple[str, Tuple[int,int,int,int]], classes:List[str], output_resolution:Tuple[int,int]) -> np.array:
    output = np.zeros((len(classes) +1 , output_resolution[0], output_resolution[1]))
    for o in objects:
        x = int(((o[1][0] + (o[1][2] - o[1][0])/2)/image_resolution[0]) * output_resolution[0])
        y = int(((o[1][1] + (o[1][3] - o[1][1])/2)/image_resolution[1]) * output_resolution[1])
        
        output[classes.index(o[0]), y, x] += 1

    output[-1, :,:] = np.invert(np.sum(output, 0).astype(np.bool_)).astype(int)
    return output.astype(np.bool_).astype(np.int_)


def display_image(img:np.array, objects:List[Tuple[str, Tuple[int,int,int,int]]], converted_objects:np.array, classes: List[str], colors:List[Tuple[int,int,int]]):
    mask = np.zeros((converted_objects.shape[1], converted_objects.shape[2], 3))

    # add bounding boxes
    for o in objects:
        img = cv2.rectangle(img, (o[1][0], o[1][3]), (o[1][2], o[1][1]), colors[classes.index(o[0])], 2)
    

    for d in range(converted_objects.shape[0] -1):
        o_mask = np.tile(np.expand_dims(converted_objects[d], 2), (1,1,3))
        o_mask[:,:,0] *= colors[d][0]
        o_mask[:,:,1] *= colors[d][1]
        o_mask[:,:,2] *= colors[d][2]
        mask = cv2.addWeighted(mask, 1, o_mask, 1, 0, dtype=cv2.CV_32F)

    o_mask = np.tile(np.expand_dims(converted_objects[-1], 2), (1,1,3))
    o_mask[:,:,0] *= colors[-1][0]
    o_mask[:,:,1] *= colors[-1][1]
    o_mask[:,:,2] *= colors[-1][2]
    mask = cv2.addWeighted(mask, 1, o_mask, 0.2, 0, dtype=cv2.CV_32F)

    upscale_mask = cv2.resize(mask, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_NEAREST)
    
    img = cv2.addWeighted(img, 0.6, upscale_mask, 0.4, 0, dtype=cv2.CV_32F)

    cv2.imshow("img", img/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ic(CLASSES)
    dataset = get_all_data()
    
    img_name = dataset["train"][2].iloc[10]["file"]
    print(img_name)
    img, objects = get_image_square_with_bounding_boxes(str(img_name).zfill(6), CLASSES)
    img_classes2 = convert_objects_to_np_array_with_background_full(img.shape, objects, CLASSES, (8,8))
    img_classes = convert_object_to_np_array_center_with_background_full(img.shape, objects, CLASSES, (8,8))

    ic(img_classes)
    ic(img_classes2)
    display_image(img, objects, img_classes2, CLASSES, CLASSES_COLORS)