import cv2
import torch
import numpy as np

vid = cv2.VideoCapture("/home/sasuke/repos/LDR_NET/sampleDataset/input_sample/background00/datasheet001.avi")
net = torch.load("./model1.pth").to(torch.device('cpu')).eval()

while True:
    ret, frame = vid.read()
    img = cv2.resize(frame, (224, 224))
    img2 = torch.permute(torch.from_numpy((img.astype(np.float32)-127.5) / 255.0), (2, 0, 1)).unsqueeze(0)
    coord = net(img2)[0].detach().numpy().reshape((4,2))*224
    coord = coord.astype(np.int32)
    img = cv2.polylines(img, [coord], True, (0,255,0), 8)
    img = cv2.resize(img, (1920, 1080))

    cv2.imshow('vid', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()