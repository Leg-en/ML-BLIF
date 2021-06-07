from joblib import dump
import numpy as np
import cv2
from joblib import load
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed
from PIL import Image
from tensorflow import keras


model = keras.models.load_model(r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\tf\Modelle\Basic_model")

ksize = 5
img = cv2.imread(r'C:\Users\Emily\Documents\Bachelor\convertet_png\DJI_0001.png')



def predict():
    predictions = model.predict(img)
    result = np.empty(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = np.argmax(predictions[i,j])
            if x == 0: #Wasser
                prediction_bgr = [255, 0, 0] #Rot
            elif x == 2: #Himmel
                prediction_bgr = [255, 255, 255] #Weiß
            elif x == 1: # Strand
                prediction_bgr = [0, 255, 255] #Blau
            else:
                prediction_bgr = [0,0,0] #Schwarz
            result[i,j] = prediction_bgr
    return result

results = predict()


results = np.asarray(results)
print(results.shape)
cv2.imwrite(r'C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\tf\predictions\out.png', results)
