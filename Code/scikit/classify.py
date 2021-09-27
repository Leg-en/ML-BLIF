"""
Lädt ein Scikit Learn Model und macht eine Klassifikation
"""

from joblib import dump
import numpy as np
import cv2
from joblib import load
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed
from PIL import Image


clf = load(r'C:\Users\Emily\Documents\GitHub\ML-BLIF\Artefakte\modelle\filename2.joblib')
img = cv2.imread(r'C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG\DJI_0092.png')



def optimized_modelling():
    img_reshape = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    prediction_strings = clf.predict(img_reshape)
    #prediction_strings_reshaped = prediction_strings.reshape(img.shape[0],img.shape[1])
    new_img = np.empty((img.shape[0] * img.shape[1], img.shape[2]))
    prediction_list = prediction_strings.tolist()
    mapping = map(map_func, prediction_list)
    map_np = np.array(list(mapping))
    reshaped = map_np.reshape((img.shape[0], img.shape[1], 3))
    cv2.imwrite(r'C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\output.png', reshaped)





def map_func(element):
    if element == 'Wasser':
        return [255, 0, 0]
    elif element == 'Himmel':
        return [255, 255, 255]
    elif element == 'Strand':
        return [0, 255, 255]
    else:
        raise ValueError("Ungueltige Eingabe")
        #prediction_rgb = [0, 0, 0]


if __name__ == '__main__':
    optimized_modelling()