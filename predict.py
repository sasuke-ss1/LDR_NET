import cv2
import numpy as np
import torch

net = torch.load('./models/model.pth')
img = cv2.imread("./imgs/img1.png")
img = cv2.resize(img, (224, 224))
img_show = np.copy(img)
img = img.astype(np.float32)
img2 = img / 255.0 # Another part where not sure about whether we are scaling the image or not.
coord = net([img2]).numpy()[0]
coord = [int(x * 224) for x in coord] # Not sure about this part as it will depend on the output of the model whether the coordinates are scaled or not.
cv2.circle(img_show, coord[0], 3, (255, 0, 0),-1)
cv2.circle(img_show, coord[1], 3, (255, 0, 0),-1)
cv2.circle(img_show, coord[2], 3, (255, 0, 0),-1)
cv2.circle(img_show, coord[3], 3, (255, 0, 0),-1)
cv2.line(img_show,coord[0],coord[1],2,(0, 0, 255))
cv2.line(img_show,coord[1],coord[2],2,(0, 0, 255))
cv2.line(img_show,coord[2],coord[3],2,(0, 0, 255))
cv2.line(img_show,coord[3],coord[0],2,(0, 0, 255))
cv2.imshow("show", img_show)
cv2.waitKey(0)
