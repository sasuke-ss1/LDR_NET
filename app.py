from flask import Flask, render_template, Response

import numpy as np
import cv2
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

app = Flask(__name__)


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


net = tf.keras.models.load_model("./model_file/model-icdar-1.4")


def gen(video):
    while True:
        _, image = video.read()
        img_show = np.copy(image)
        x, y, _ = img_show.shape
        img = cv2.resize(image, (224, 224))

        img = img.astype(np.float32)
        img2 = img / 255.0
        result = net([img2], training=False).numpy()[0]
        coord = result[0:8]
        coord[0::2] *= x
        coord[1::2] *= y
        coord = coord.astype(int)
        cv2.circle(img_show, (coord[0], coord[1]), 3, (0, 0, 255), -1)
        cv2.circle(img_show, (coord[2], coord[3]), 3, (0, 255, 255), -1)
        cv2.circle(img_show, (coord[4], coord[5]), 3, (255, 0, 0), -1)
        cv2.circle(img_show, (coord[6], coord[7]), 3, (0, 255, 0), -1)
        img_show = cv2.resize(img_show, (2 * y, 2 * x))
        img_show = cv2.flip(img_show, 1)
        _, jpeg = cv2.imencode(".jpg", img_show)

        frame = jpeg.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(video), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    video = cv2.VideoCapture(0)
    app.run(host="0.0.0.0", debug=False, threaded=True)
