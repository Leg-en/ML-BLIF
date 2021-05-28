import sys
import time
import cv2
import os
import numpy as np
import numba
from numba import jit
import pandas
import multiprocessing as mp
import csv
import queue



@jit
def buildKernels(ksize):
    kernels = []
    for i in range(ksize*ksize):
        x = np.zeros((ksize, ksize))
        x[i%ksize][(ksize-1) - i%ksize] = 1
        kernels.append(x)
    return kernels

@jit(parallel=True)
def buildIMGS(kernels, img):
    images = []
    for kernel in kernels:
        cv_filter = cv2.filter2D(img, -1, kernel)
        images.append(cv_filter)
    return images



@jit(parallel=True)
def preprocess(Image: str, csv, ksize, conn) -> None:
    frame = pandas.read_csv(csv)
    kernels = buildKernels(ksize)
    for index, row in frame.iterrows():
        path = os.path.join(Image, row["filename"])
        img = cv2.imread(path)
        imgs = buildIMGS(kernels, img)
        xmin,xmax,ymin,ymax = row["xmin"],row["xmax"],row["ymin"],row["ymax"]
        while ymin <= ymax:
            while xmin <= xmax:
                r, g, b = [], [], []
                for i in imgs:
                    r.append(i[ymin, xmin, 0])
                    g.append(i[ymin, xmin, 1])
                    b.append(i[ymin, xmin, 2])
                #r, g und b sind listen die für jeden pixel alle 25x25 RGB werte enthalten
                conn.send([r,g,b])
                xmin += 1
            ymin += 1
    conn.close()


def print_rgb(conn):
    with open("test.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        while True:
            try:
                x = conn.recv()
                writer.writerow(x)
            except EOFError:
                return



if __name__ == '__main__':
    parent_conn, child_conn = mp.Pipe()
    t = mp.Process(target=preprocess, args=(r'C:\Users\Emily\Documents\Bachelor\convertet_png',
               r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\out.csv", 5, parent_conn,))
    t.start()
    t2 = mp.Process(target=print_rgb, args=(child_conn,))
    t2.start()
    t.join()
    t2.join()
