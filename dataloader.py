import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import transforms


class DocData(Dataset):
    def __init__(self, img_dir, annotations_path, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.annotations_path = annotations_path
        self.annotations = self.get_annotations()

    def get_annotations(self):
        tree = ET.parse(self.annotations_path)
        root = tree.getroot()
        annotations = []

        for neighbor in root.iter("frame"):
            coords = []
            for pt in neighbor.iter("point"):
                x = float(pt.get("x"))
                y = float(pt.get("y"))
                coords.append(x / 1920)
                coords.append(y / 1080)
            annotations.append(coords)

        return torch.tensor(annotations, dtype=torch.float32)

    def get(self, idx, deg, pad_ratio, crop_ratio):
        img_name = os.path.join(self.img_dir, f"frame{idx + 1}.png")
        img = cv2.imread(img_name)
        img = cv2.resize(img, (224, 224))
        img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        coords = self.annotations[idx].reshape((4, 2))

        if self.transforms:
            img = self.transforms(img)

        img, coords = transforms.rotate_with_points(img, coords, deg * np.pi / 180)
        img, coords = transforms.random_resize(img, coords, pad_ratio, mode="pad")
        img, coords = transforms.random_resize(img, coords, crop_ratio, mode="crop")

        return img / 255 - 0.5, coords.reshape(-1)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        deg = np.random.randint(0, 360)
        pad_ratio = np.random.uniform(0, 0.5, 4)
        crop_ratio = np.random.uniform(0, 0.1, 4)
        return self.get(idx, deg, pad_ratio, crop_ratio)


def display_img_coords(img, coords, title):
    print(coords.tolist())
    img = ((img + 0.5) * 255).permute(1, 2, 0)  # (C, H, W) -> (H, W, C), unnormalise
    img = np.ascontiguousarray(img, dtype=np.uint8)
    coords = np.array(coords.reshape((4, 2)) * 224, dtype=np.int32)
    cv2.imshow(title, cv2.polylines(img, [coords], True, (0, 255, 0), 8))
    cv2.waitKey(0)


if __name__ == "__main__":
    image_dir = "./datasheet001"
    annotations_path = "./sampleDataset/input_sample_groundtruth/background00_gt/datasheet001.gt.xml"
    data = DocData(img_dir=image_dir, annotations_path=annotations_path)

    idx = np.random.randint(0, len(data))

    img, coords = data.get(idx, 0, np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))
    display_img_coords(img, coords, "Original")

    img, coords = data[idx]
    display_img_coords(img, coords, "Random")
