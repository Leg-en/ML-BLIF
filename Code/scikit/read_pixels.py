import csv
import multiprocessing as mp
import os

import cv2
import numpy as np
import pandas
from joblib import dump
#from numba import jit
from sklearn.neural_network import MLPClassifier

img_scale = 0.25

#@jit
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


#@jit(parallel=True)
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


def preprocess(Image: str, csv, ksize, conn) -> None:
    """
    Präprozessiert das Image und übergibt es an den Thread mit dem perzeptron
    :param Image: Pfad zu den bildern
    :param csv: Pfad zu der CSV Datei
    :param ksize: Kernel Size
    :param conn: Die connection pipe zu dem anderen Thread
    :return:
    """
    print("Processing Started")
    frame = pandas.read_csv(csv)
    kernels = buildKernels(ksize)
    for index, row in frame.iterrows():
        path = os.path.join(Image, row["filename"])
        img = cv2.imread(path)

        #img = cv2.resize(img, (int(img.shape[1]*img_scale), int(img.shape[0]*img_scale)))

        imgs = buildIMGS(kernels, img)
        xmin, xmax, ymin, ymax = row["xmin"], row["xmax"], row["ymin"], row["ymax"]

        #xmin = xmin*img_scale
        #xmin = xmax*img_scale
        #xmax = int((xmax-xmin)*img_scale + xmin)
        #ymax = int((ymax-ymin)*img_scale + ymin)

        img_ = imgs[:,ymin:ymax, xmin:xmax,:]
        img_ = img_.reshape(img_.shape[0]*img_.shape[1]*img_.shape[2],3)
        conn.send([img_, row["class"]])
        print("preprocessed: ", row["filename"])
    conn.send(["Finished"])
    conn.close()
    print("Image Processing Finished")
    return




def pcn(conn):
    """
    Hier wird das MLP Trainiert und am ende gespeichert.
    :param conn: Connection pipe zu dem anderen Thread
    :return:
    """
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10), random_state=1)
    print("PCN Training started")
    while True:
        try:
            x = conn.recv()
            if str(x[0]) == "Finished":
                dump(clf, 'filename.joblib')
                print("PCN Trained and Saved")
                return
            else:
                data = x[0]
                y = np.chararray(len(data), itemsize=10, unicode=True)
                y[:] = x[1]
                clf.partial_fit(data, y, classes=["Wasser", "Strand", "Himmel"])
                print("Train iteration complete")
        except EOFError:
            dump(clf, 'filename.joblib')
            print("PCN Trained and Saved")
            return
        except KeyboardInterrupt:
            dump(clf, 'filename.joblib')
            print("PCN Trained and Saved")
            return


if __name__ == '__main__':
    parent_conn, child_conn = mp.Pipe()
    t = mp.Process(target=preprocess, args=(r'/home/pi/Desktop/convertet_png',
                                            r"/home/pi/Desktop/image_data.csv", 5, parent_conn,))
    t.start()
    t2 = mp.Process(target=pcn, args=(child_conn,))
    t2.start()
    t.join()
    t2.join()
    print("Complete")
