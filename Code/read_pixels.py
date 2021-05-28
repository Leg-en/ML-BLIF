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




@jit
def buildKernels(ksize):
    kernels = []
    for i in range(ksize):
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
def preprocess(Image: str, csv, ksize, queue) -> None:
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
                queue.put([r,g,b])
                xmin += 1
            ymin += 1


def print_rgb(queue):
    with open("test.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for job in iter(queue.get, None):
            writer.writerow(job)



if __name__ == '__main__':
    q = mp.Queue()
    t = mp.Process(target=preprocess, args=(r'C:\Users\Emily\Documents\Bachelor\convertet_png',
               r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\out.csv", 5, q,))
    #preprocess(r'C:\Users\Emily\Documents\Bachelor\convertet_png',
    #           r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\out.csv", 5, q)
    t.start()
    t2 = mp.Process(target=print_rgb, args=(q,))
    t2.start()
    t.join()
    time.sleep()
    t2.terminate()
