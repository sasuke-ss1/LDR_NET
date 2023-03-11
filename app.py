#!/usr/bin/env python
from flask import Flask, render_template, Response
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# emulated camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera
import cv2

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
net = tf.keras.models.load_model("./model_file/model-icdar-1.4")

def gen(video):
    while True:
        success, image = video.read()
        img_show = np.copy(image)
        x, y , _ = img_show.shape
        img = cv2.resize(image, (224, 224))
        
        img = img.astype(np.float32)
        img2 = img / 255.0
        result = net([img2], training=False).numpy()[0]
        coord = result[0:8]
        coord[0::2] *= x
        coord[1::2] *= y
        coord = coord.astype(int)
        cv2.circle(img_show, (coord[0], coord[1]), 3, (0, 0, 255),-1)
        cv2.circle(img_show, (coord[2], coord[3]), 3, (0, 255, 255),-1)
        cv2.circle(img_show, (coord[4], coord[5]), 3, (255, 0, 0),-1)
        cv2.circle(img_show, (coord[6], coord[7]), 3, (0, 255, 0),-1)
        img_show = cv2.resize(img_show, (2*y, 2*x))
        img_show = cv2.flip(img_show, 1)
        # frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # frame_gray = cv2.equalizeHist(frame_gray)

        # faces = face_cascade.detectMultiScale(frame_gray)

        # for (x, y, w, h) in faces:
        #     center = (x + w//2, y + h//2)
        #     cv2.putText(image, "X: " + str(center[0]) + " Y: " + str(center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        #     image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #     faceROI = frame_gray[y:y+h, x:x+w]
        ret, jpeg = cv2.imencode('.jpg', img_show)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    video = cv2.VideoCapture(0)
    app.run(host='0.0.0.0', debug=False, threaded=True)