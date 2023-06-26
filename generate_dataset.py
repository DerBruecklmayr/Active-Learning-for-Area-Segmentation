from typing import List, Dict, Tuple
import pandas as pd
from os.path import join, abspath
import numpy as np
from icecream import ic
import torch
from torchvision import transforms

TRANSFORMATION = transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(512),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

CLASSES = ["cow", "cat", "horse", "dog", "sheep"]

CLASSES_COLORS = [(0,0,255), (0,255,0), (255,0,0), (200,100,0), (0,100,200), (0,0,0)]

def get_all_data() -> Dict[str, List[pd.DataFrame]]:
    files = {"train": [], "trainval": [], "val": []}

    for c in CLASSES:
        for k in files.keys():
            df = pd.read_csv(abspath(join("VOCdevkit", "VOC2007", "ImageSets", "Main", f"{c}_{k}.txt" )),  delim_whitespace=True, names=["file", "include"])
            df = df[df["include"] == 1].drop(labels='include', axis=1)
            files[k].append(df)

    return files


def get_conjunced_subset(data: List[pd.DataFrame], percentage, seed) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if percentage == 1:
        return pd.concat(data)
    
    include, others = [], []
    for d in data:
        include.append(d.sample(frac=percentage, random_state=seed))
        others.append(d.iloc[d.index.isin(d.index.difference(include[-1].index))])

    return pd.concat(include), pd.concat(others) #pd.concat([d.sample(frac=percentage, random_state=seed) for d in data])


if __name__ == "__main__":
    ic(CLASSES)
    data = get_all_data()
    ic(data["train"][0])
    include, others = get_conjunced_subset(data['train'], 0.10, 4)
