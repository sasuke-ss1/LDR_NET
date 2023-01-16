import cv2
import numpy as np
import torch
from torchvision.transforms.functional import rotate


def rotate_point(img, point, radians, image_h, image_w):
    y = image_h * (point[:, 1] - 0.5)
    x = image_w * (point[:, 0] - 0.5)

    coordinates = torch.stack([y, x], axis=1)
    c, s = np.cos(radians), np.sin(radians)
    rot_mat = torch.from_numpy(np.array([[c, s], [-s, c]], dtype=np.float32))

    new_coords = torch.matmul(rot_mat, coordinates.transpose(0, 1))

    x = new_coords[0, :] / image_h + 0.5
    y = new_coords[1, :] / image_w + 0.5
    # print(radians * 180 / np.pi)
    return rotate(img, -radians * 180 / np.pi), torch.stack([y, x], axis=1)


if __name__ == "__main__":
    img = cv2.imread("datasheet001/frame1.png")
    img = cv2.resize(img, (224, 224))
    coord = torch.from_numpy(np.array([0.3720, 0.7466, 0.3835, 0.1873, 0.6144, 0.1852, 0.6548, 0.7335]).reshape((4, 2)))
    img, new_coord = rotate_point(torch.from_numpy(img).permute(2, 0, 1), coord, 0.5 * np.pi, 224, 224)

    new_coord = (new_coord * 224).numpy()
    new_coord = new_coord.astype(np.int32)
    img = img.permute(1, 2, 0)
    img = np.ascontiguousarray(img.numpy(), dtype=np.uint8)
    # print(new_coord)
    img = cv2.polylines(img, [new_coord], True, (0, 255, 0), 8)
    img = cv2.resize(img, (1920, 1080))

    cv2.imshow("test", img)
    cv2.waitKey(0)
