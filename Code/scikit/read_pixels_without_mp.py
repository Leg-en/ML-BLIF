import csv
import os

import cv2
import numpy as np
import pandas
from joblib import dump
# from numba import jit
from sklearn.neural_network import MLPClassifier

img_scale = 0.25


# @jit
def buildKernels(ksize):
    """
        Baut ein Kernel welches mit 0 gefüllt ist
        :param ksize: Die Kernelsize
        :return: Eine liste mit Kernels
        """
    kernels = []
    for i in range(ksize * ksize):
        x = np.zeros((ksize, ksize))
        x[i % ksize][(ksize - 1) - i % ksize] = 1
        kernels.append(x)
    return kernels


# @jit(parallel=True)
def buildIMGS(kernels, img):
    """
        Nimmt ein Bild und wendet das Kernel darauf an.
        :param kernels:
        :param img:
        :return:
        """
    images = []
    for kernel in kernels:
        cv_filter = cv2.filter2D(img, -1, kernel)
        images.append(cv_filter)
    return np.array(images)


def preprocess(Image: str, csv, ksize) -> None:
    """
        Python Generator der der die Bilder mit den entsprechenden Kernels als np nd array yielded
        :param Image: Pfad zu den bildern
        :param csv: Pfad zu der CSV Datei
        :param ksize: Kernel Size
        :yields: np nd arrays mit den bildern
        """
    print("Processing Started")
    frame = pandas.read_csv(csv)
    kernels = buildKernels(ksize)
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
        print("preprocessed: ", row["filename"])
        yield [img_, row["class"]]




def pcn(Image: str, csv, ksize):
    """
    Hier wird das modell Trainiert und gespeichert
    :param Image: Bildpfad
    :param csv: Csvpfad
    :param ksize: Kernelsize
    :return:
    """
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5,3), random_state=1)
    gen = preprocess(Image, csv, ksize)
    for i in gen:
        X = i[0]
        Y = np.chararray(len(X), itemsize=10, unicode=True)
        Y[:] = i[1]
        clf.partial_fit(X, Y, classes=["Wasser", "Strand", "Himmel"])
    dump(clf, 'filename.joblib')


if __name__ == '__main__':
    pcn(r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG", r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Artefakte\image_data.csv", 5)
    #pcn(r"./datasets", r"image_data.csv", 5)
    print("Complete")
