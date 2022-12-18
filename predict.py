import cv2
import torch
import numpy as np
from LDRNet import LDRNet
from PIL import ImageDraw, Image


net = LDRNet(points_size=4)
net.load_state_dict(torch.load("./model.pth"))
net.eval()

img = cv2.imread("./datasheet001/frame1.png")
img_show = Image.open("./datasheet001/frame1.png")

img = img.astype(np.float32)
img2 = torch.permute(torch.from_numpy((img - 127.5) / 255.0), (2, 0, 1)).unsqueeze(0)
result = net(img2)[0].detach().numpy()
coord = [int(x * 224) for x in result[0]]

draw = ImageDraw.Draw(img_show)
draw.line(coord, fill=(255, 255, 0), width=5)
img_show.show()
