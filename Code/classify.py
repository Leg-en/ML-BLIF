from joblib import dump
import numpy as np
import cv2
from joblib import load
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed


clf = load('filename.joblib')

ksize = 5
img = cv2.imread('../../Drohnenbilder_convertet/DJI_0001.png')
img_width = img.shape[1]
img_height = img.shape[0]

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


def predict_row(row_data, row):
    predict_image = np.empty((img_width, 1, 3))
    for x in range(img_width):
        rgb = row_data[x].reshape(1,-1)
        prediction_string = clf.predict(rgb)

        if prediction_string == 'Wasser':
            prediction_rgb = (0, 0, 255)
        elif prediction_string == 'Himmel':
            prediction_rgb = (255, 0, 255)
        elif prediction_string == 'Strand':
            prediction_rgb = (0, 255, 0)
        else:
            prediction_rgb = (255, 255, 0)

        predict_image[x] = prediction_rgb
        return predict_image

results = Parallel(n_jobs=-1)(
    delayed(predict_row)(img[row, :], row)
    for row in range(img_height))

results = np.asarray(results)[:, :, -1]
resized = cv2.resize(results, (int(img_width/4), int(img_height/4)), interpolation=cv2.INTER_CUBIC)
cv2.imshow('frame', resized)
cv2.waitKey(5000)
