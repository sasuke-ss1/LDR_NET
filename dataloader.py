import os
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from transforms import rotate_point


image_dir = "./datasheet001"
annotations_path = "./sampleDataset/input_sample_groundtruth/background00_gt/datasheet001.gt.xml"

tree = ET.parse(annotations_path)
root = tree.getroot()

sample_annotations = []
for neighbor in root.iter("frame"):
    coords = []
    for pt in neighbor.iter("point"):
        x = float(pt.get("x"))
        y = float(pt.get("y"))
        coords.append((x, y))
    sample_annotations.append(coords)


class DocData(Dataset):
    def __init__(self, img_dir, transforms=None, annotations_path=annotations_path):
        self.img_dir = img_dir
        self.annotations_path = annotations_path
        self.transforms = transforms
        self.get_annotations()

    def get_annotations(self):
        tree = ET.parse(self.annotations_path)
        root = tree.getroot()

        self.annotations = []
        for neighbor in root.iter("frame"):
            coords = []
            for pt in neighbor.iter("point"):
                x = float(pt.get("x"))
                y = float(pt.get("y"))
                coords.append(x / 1920)
                coords.append(y / 1080)
            self.annotations.append((np.array(coords, np.float32)))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        deg = None
        if isinstance(idx, tuple):
            idx, deg = idx

        img_name = os.path.join(self.img_dir, f"frame{idx + 1}.png")
        img = cv2.imread(img_name)
        # img = cv2.resize(img, (224, 224))
        coords = self.annotations[idx]

        if self.transforms:
            img = self.transforms(img)

        if deg:
            try:
                img = torch.from_numpy(img)
            except TypeError:
                pass

            img, coords = rotate_point(
                img.permute(2, 0, 1),
                torch.from_numpy(np.array(coords).reshape((4, 2))),
                deg * np.pi / 180,
                *img.shape[:2],
            )
            img = img.permute(1, 2, 0)

        self.annotations = torch.from_numpy(np.array(self.annotations))
        # print(img.shape)
        return (img - 127.5) / 255, coords


if __name__ == "__main__":
    data = DocData(img_dir=image_dir, annotations_path=annotations_path)
    sample, coords = data[0]
    plt.imshow(sample + 0.5)
    plt.show()
    print(coords)
    sample, coords = data[0, 90]
    plt.imshow(sample + 0.5)
    plt.show()
    print(coords)
