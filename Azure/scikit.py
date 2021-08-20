import csv
import os

import cv2
import numpy as np
import pandas
from joblib import dump
# from numba import jit
from sklearn.neural_network import MLPClassifier
import sys


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


def preprocess(Image: str, csv, ksize) -> None:
    print("Processing Started")
    frame = pandas.read_csv(csv)
    kernels = buildKernels(ksize)
    for index, row in frame.iterrows():
        path = os.path.join(Image, row["filename"])
        img = cv2.imread(path)
        imgs = buildIMGS(kernels, img)
        xmin, xmax, ymin, ymax = row["xmin"], row["xmax"], row["ymin"], row["ymax"]

        img_ = imgs[:, ymin:ymax, xmin:xmax, :]
        img_ = img_.reshape(img_.shape[0] * img_.shape[1] * img_.shape[2], 3)
        print("preprocessed: ", row["filename"])
        yield [img_, row["class"]]




def pcn(Image: str, csv, ksize):
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(3,150,150), verbose=True)
    gen = preprocess(Image, csv, ksize)
    for i in gen:
        X = i[0]
        Y = np.chararray(len(X), itemsize=10, unicode=True)
        Y[:] = i[1]
        clf.partial_fit(X, Y, classes=["Wasser", "Strand", "Himmel"])
    dump(clf,os.path.join(sys.argv[2], "filename.joblib"))


if __name__ == '__main__':
    #pcn(r"Users/marius100100/datasets", r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\out.csv", 5)
    with open(os.path.join(sys.argv[2], "test.txt"), "w") as file:
        file.write("ABC")
    datasets = sys.argv[1]
    pcn(datasets, r"out.csv", 5)
    print("Complete")
