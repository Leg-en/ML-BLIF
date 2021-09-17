import cv2
import os

path = "/home/emily/Schreibtisch/BA/PNG/Trimaps"
dir = os.listdir(path)

for i in dir:
    img = cv2.imread(os.path.join(path,i))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(path,i),img)