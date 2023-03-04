import cv2
import numpy as np
import torch

vidpath = 'sampleDataset/input_sample/background00/datasheet001.avi'
vid = cv2.VideoCapture(vidpath)
net = torch.load("./model.pth").to(torch.device("cpu")).eval()

while True:
    ret, frame = vid.read()
    clone = frame.copy()
    img = cv2.resize(frame, (224, 224))
    img2 = torch.permute(torch.from_numpy((img.astype(np.float32) - 127.5) / 255.0), (2, 0, 1)).unsqueeze(0)
    border = net(img2)[1].detach().numpy().reshape((-1,2))
    coord = net(img2)[0].detach().numpy().reshape((-1, 2))
    x, y, _ = clone.shape
    coord[:,0] = coord[:,0]*y
    coord[:,1] = coord[:,1]*x
    border[:,0] = border[:,0]*y
    border[:,1] = border[:,1]*x
    coord = coord.astype(np.int32)
    for c in coord:
        clone = cv2.circle(clone, tuple(c), 10, (255,0, 0), 2)
    for b in border:
        print(b)
        clone = cv2.circle(clone, tuple(b), 10, (255,0,0), 2)
    img = cv2.polylines(clone, [coord], True, (0, 255, 0), 8)
    #img = cv2.resize(img, (1920, 1080))
    
    cv2.imshow("vid", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
