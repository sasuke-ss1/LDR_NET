import tensorflow as tf
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os

net = tf.keras.models.load_model("LDR_NET/model_file/model-icdar-1.4")

vidpath = '/media/web_slinger/Windows/Users/varad/Documents/Computer Vision/On Device DL/testDataset/background03/magazine002.avi'
vid = cv2.VideoCapture(vidpath)

output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*"XVID"),30, (1920,1080))
while True:
    _, frame = vid.read()
    img = frame.copy()
    x,y,_ = frame.shape
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img2 = img / 255.0
    result = net([img2], training=False).numpy()[0]
    coord = np.array(result[0:8]).reshape((-1,2))
    coord[:,0] = coord[:,0]*y
    coord[:,1] = coord[:,1]*x
    coord = coord.astype(np.int32)
    frame = cv2.polylines(frame , [coord], True, (0, 255, 0), 8)
    print(x,y)
    output.write(frame)
    cv2.imshow("show", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release()
output.release()
