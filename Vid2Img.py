import cv2 
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', "-p", required=True, help="Relative Path to video Directory")
args = parser.parse_args()
path = args.path
Direc_Name = path.split("/")[-1].split(".")[0]
vid = cv2.VideoCapture(path)
try:

    if not os.path.exists(Direc_Name):
        os.makedirs(Direc_Name)
        print("Directory was created")
except OSError:
    print("Error in creating Directory")

frame = 1
while(True):
    ret, f = vid.read()
    if ret:
        name = f'./{Direc_Name}/frame' + str(frame) + ".png"
        cv2.imwrite(name, f)
        frame += 1
    else:
        break
vid.release()
cv2.destroyAllWindows()