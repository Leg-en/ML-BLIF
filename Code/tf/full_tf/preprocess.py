import csv
import os
import itertools
import uuid

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


def generateKernelImages(path,ksize, targetpath):
    kernels = buildKernels(ksize)
    for i in os.listdir(path):
        image_path = os.path.join(path,i)
        img = cv2.imread(image_path)
        imgs = buildIMGS(kernels, img)
        for j in imgs:
            cv2.imwrite(os.path.join(targetpath, str(uuid.uuid1())+".png"),j)



if __name__ == '__main__':
    path = r"C:\Users\Emily\Documents\Bachelor\modified"
    targetpath = r"C:\Users\Emily\Documents\Bachelor\classifield_pngs_with_kernel"
    for i in os.listdir(path):
        generateKernelImages(os.path.join(path,i), 5, os.path.join(targetpath,i))
