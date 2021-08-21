import csv
import os
import itertools
import cv2
import numpy
import numpy as np
import pandas
from joblib import dump
# from numba import jit
from sklearn.neural_network import MLPClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
import logging
import tensorflow as tf

img_scale = 0.25


# @jit
def buildKernels(ksize):
    kernels = []
    for i in range(ksize * ksize):
        x = np.zeros((ksize, ksize))
        x[i % ksize][(ksize - 1) - i % ksize] = 1
        kernels.append(x)
    return kernels


# @jit(parallel=True)
def buildIMGS(kernels, img):
    images = []
    for kernel in kernels:
        cv_filter = cv2.filter2D(img, -1, kernel)
        images.append(cv_filter)
    return np.array(images)


def preprocess(Image: str, csv, ksize,epochs) -> None:
    print("Processing Started")
    frame = pandas.read_csv(csv)
    kernels = buildKernels(ksize)
    images = {}
    for index, row in frame.iterrows():
        if row["filename"] in images:
            images[row["filename"]].append(
                {
                    "class": row["class"],
                    "coords": [row["xmin"], row["xmax"], row["ymin"], row["ymax"]],
                    "size": [row["width"], row["height"]]
                }
            )
        else:
            images[row["filename"]] = [{
                "class":row["class"],
                "coords":[row["xmin"], row["xmax"], row["ymin"], row["ymax"]],
                "size":[row["width"], row["height"]]
            }]
    for i in images:
        path = os.path.join(Image, i)
        img = cv2.imread(path)
        imgs = buildIMGS(kernels, img)
        coords = np.empty((img.shape[0],img.shape[1],1))
        coords[:,:,:] = 3
        for j in images[i]:
            xmin,ymin,xmax,ymax = j["coords"][0],j["coords"][1],j["coords"][2],j["coords"][3]
            if j["class"] == "Wasser":
                coords[ymin:ymax, xmin:xmax,:] = 0
            elif j["class"] == "Strand":
                coords[ymin:ymax, xmin:xmax,:] = 1
            elif j["class"] == "Himmel":
                coords[ymin:ymax, xmin:xmax,:] = 2
            else:
                raise ValueError("Irgendwas stimmt hier nicht")
        for j in imgs:
            c = coords.reshape((j.shape[0]*j.shape[1],1))
            _img = j.reshape((j.shape[0]*j.shape[1],3))
            div = divCheck(_img.shape[0])
            #Smalling for GPU Use
            c_small = np.split(c, div)
            _img_small = np.split(_img,div)
            for i in range(len(_img_small)):
                yield (_img_small[i], c_small[i])
            #yield(_img,c)


def divCheck(shape, div_base=100):
    div = div_base
    while True:
        if shape % div == 0:
            return div
        else:
            div +=1

def calculateSteps(path):
    #todo implement dynamic calculation
    x = 3000
    y = 4000
    imgs_per_img = 25
    imgs = 105

    res = x*y*105*25
    return res



    frame = pandas.read_csv(path)
    images = {}
    for index, row in frame.iterrows():
        if row["filename"] in images:
            images[row["filename"]].append(
                {
                    "class": row["class"],
                    "coords": [row["xmin"], row["xmax"], row["ymin"], row["ymax"]],
                    "size": [row["width"], row["height"]]
                }
            )
        else:
            images[row["filename"]] = [{
                "class":row["class"],
                "coords":[row["xmin"], row["xmax"], row["ymin"], row["ymax"]],
                "size":[row["width"], row["height"]]
            }]
    #for i in images:

def pcn(Image: str, csv, ksize, epochs):
    model = Sequential()
    model.add(Dense(10,activation='relu',input_shape=(3,)))
    model.add(Dense(4, activation="softmax"))
    opt = Adam()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
    gen = preprocess(Image, csv, ksize, epochs)
    #model.fit(gen,epochs=1, steps_per_epoch=57151)
    model.fit(gen, steps_per_epoch=calculateSteps(csv))
    model.save("Model")
    print("Fitted")

if __name__ == '__main__':
    pcn(r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG", r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\preprocess\out.csv", 5, 10)
    #pcn(r"/home/azureuser/Bachelor/convertet_png/", "/home/azureuser/Bachelor/Code/image_data.csv", 5, 10)
