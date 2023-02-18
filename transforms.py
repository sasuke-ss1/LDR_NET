import cv2
import numpy as np
import torch
from torchvision.transforms.functional import rotate
from torchvision.transforms import Resize


def rotate_with_points(img, points, radians):
    img_h, img_w = img.shape[1:3]  # (C, H, W)

    y = img_h * (points[:, 1] - 0.5)
    x = img_w * (points[:, 0] - 0.5)

    coordinates = torch.stack([y, x], axis=1)
    c, s = np.cos(radians), np.sin(radians)
    rot_mat = torch.tensor([[c, s], [-s, c]], dtype=torch.float32)

    new_coords = torch.matmul(rot_mat, coordinates.transpose(0, 1))

    x = new_coords[0, :] / img_h + 0.5
    y = new_coords[1, :] / img_w + 0.5

    return rotate(img, -radians * 180 / np.pi), torch.stack([y, x], axis=1)


def random_resize(img, coords, ratio, mode):
    img_c, img_h, img_w = img.shape  # (C, H, W)

    size_change = np.rint((ratio if mode == "pad" else -ratio) * np.float32([img_h, img_w] * 2)).astype(np.int32)
    new_h = img_h + size_change[0] + size_change[2]
    new_w = img_w + size_change[1] + size_change[3]

    if mode == "pad":
        img_padded = torch.zeros((img_c, new_h, new_w), dtype=torch.float32)
        img_padded[:, size_change[0] : size_change[0] + img_h, size_change[1] : size_change[1] + img_w] = img
        img = img_padded
    elif mode == "crop":
        img = img[:, -size_change[0] : -size_change[0] + new_h, -size_change[1] : -size_change[1] + new_w]
    else:
        raise ValueError("Invalid mode")

    coords[:, 0] = (coords[:, 0] * img_w + size_change[1]) / new_w
    coords[:, 1] = (coords[:, 1] * img_h + size_change[0]) / new_h
    img = Resize((224,224))(img)
    return img, coords


if __name__ == "__main__":
    img = cv2.imread("datasheet001/frame1.png")
    img = torch.from_numpy(cv2.resize(img, (224, 224))).permute(2, 0, 1)
    coord = np.array([0.3720, 0.7466, 0.3835, 0.1873, 0.6144, 0.1852, 0.6548, 0.7335], dtype=np.float32)
    img, new_coord = rotate_with_points(img, torch.from_numpy(coord).reshape((4, 2)), 0.5 * np.pi, 224, 224)

    new_coord = (new_coord * 224).numpy()
    new_coord = new_coord.astype(np.int32)
    img = img.permute(1, 2, 0)
    img = np.ascontiguousarray(img.numpy(), dtype=np.uint8)
    # print(new_coord)
    img = cv2.polylines(img, [new_coord], True, (0, 255, 0), 8)
    img = cv2.resize(img, (1920, 1080))

    cv2.imshow("test", img)
    cv2.waitKey(0)
