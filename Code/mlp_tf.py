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
    for i in range(epochs):
        for index, row in frame.iterrows():
            path = os.path.join(Image, row["filename"])
            img = cv2.imread(path)

            # img = cv2.resize(img, (int(img.shape[1]*img_scale), int(img.shape[0]*img_scale)))

            imgs = buildIMGS(kernels, img)
            xmin, xmax, ymin, ymax = row["xmin"], row["xmax"], row["ymin"], row["ymax"]

            # xmin = xmin*img_scale
            # xmin = xmax*img_scale
            # xmax = int((xmax-xmin)*img_scale + xmin)
            # ymax = int((ymax-ymin)*img_scale + ymin)

            img_ = imgs[:, ymin:ymax, xmin:xmax, :]
            img_ = img_.reshape(img_.shape[0] * img_.shape[1] * img_.shape[2], 3)
            div = divCheck(img_.shape[0])
            for i in numpy.split(img_, div):
                Y = np.empty(len(i))
                if row["class"] == "Wasser":
                    y = 0
                elif row["class"] == "Strand":
                    y = 1
                elif row["class"] == "Himmel":
                    y = 3
                else:
                    y = 4
                Y[:] = y
                yield (i, Y)

            #print("preprocessed: ", row["filename"])


def divCheck(shape, div_base=100):
    div = div_base
    while True:
        if shape % div == 0:
            return div
        else:
            div +=1


def pcn(Image: str, csv, ksize, epochs):
    model = Sequential()
    model.add(Dense(10,activation='relu'))
    model.add(Dense(4, activation="softmax"))
    opt = Adam()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
    gen = preprocess(Image, csv, ksize, epochs)
    model.fit(gen,epochs=epochs, steps_per_epoch=57151)
    model.save("Model")
    print("Fitted")

if __name__ == '__main__':
    #pcn(r"C:\Users\Emily\Documents\Bachelor\convertet_png", r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\out.csv", 5, 10)
    pcn(r"/home/azureuser/Bachelor/convertet_png/", "/home/azureuser/Bachelor/Code/out.csv", 5, 10)