import csv
import multiprocessing as mp
import os

import cv2
import numpy as np
import pandas
from joblib import dump
#from numba import jit
from sklearn.neural_network import MLPClassifier


#@jit
def buildKernels(ksize):
    kernels = []
    for i in range(ksize * ksize):
        x = np.zeros((ksize, ksize))
        x[i % ksize][(ksize - 1) - i % ksize] = 1
        kernels.append(x)
    return kernels


#@jit(parallel=True)
def buildIMGS(kernels, img):
    images = []
    for kernel in kernels:
        cv_filter = cv2.filter2D(img, -1, kernel)
        images.append(cv_filter)
    return np.array(images)


def preprocess(Image: str, csv, ksize, conn) -> None:
    print("Processing Started")
    frame = pandas.read_csv(csv)
    kernels = buildKernels(ksize)
    for index, row in frame.iterrows():
        path = os.path.join(Image, row["filename"])
        img = cv2.imread(path)
        imgs = buildIMGS(kernels, img)
        xmin, xmax, ymin, ymax = row["xmin"], row["xmax"], row["ymin"], row["ymax"]
        img_ = imgs[:,ymin:ymax, xmin:xmax,:]
        img_ = img_.reshape(img_.shape[0]*img_.shape[1]*img_.shape[2],3)
        conn.send([img_, row["class"]])
        print("preprocessed: ", row["filename"])
    conn.close()
    print("Image Processing Finished")
    return


def print_rgb(conn):
    with open("test.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        while True:
            try:
                x = conn.recv()
                writer.writerow(x)
            except EOFError:
                return


def pcn(conn):
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10), random_state=1)
    print("PCN Training started")
    while True:
        try:
            x = conn.recv()
            data = x[0]
            y = np.chararray(len(data), itemsize=10, unicode=True)
            y[:] = x[1]
            clf.partial_fit(data, y, classes=["Wasser", "Strand", "Himmel"])
            print("Train iteration complete")
        except EOFError:
            dump(clf, 'filename.joblib')
            print("PCN Trained and Saved")
            return


if __name__ == '__main__':
    parent_conn, child_conn = mp.Pipe()
    #t = mp.Process(target=preprocess, args=(r'/home/pi/Desktop/convertet_png',
    #                                        r"/home/pi/Desktop/out.csv", 5, parent_conn,))
    t = mp.Process(target=preprocess, args=(r"C:\Users\Emily\Documents\Bachelor\convertet_png",
                                            r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\out.csv", 5, parent_conn,))
    t.start()
    # t2 = mp.Process(target=print_rgb, args=(child_conn,))
    t2 = mp.Process(target=pcn, args=(child_conn,))
    t2.start()
    t.join()
    t2.join()
    print("Complete")
