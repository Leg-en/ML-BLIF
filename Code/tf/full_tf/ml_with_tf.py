import csv
import os
import itertools
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

def pcn(Image: str, csv, ksize, epochs):
    model = Sequential()
    model.add(Dense(10,activation='relu',input_shape=(3,)))
    model.add(Dense(4, activation="softmax"))
    opt = Adam()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
    gen = preprocess(Image, csv, ksize, epochs)
    #model.fit(gen,epochs=1, steps_per_epoch=57151)
    model.fit(gen, steps_per_epoch=calculateSteps(csv))
    model.save("Model")
    print("Fitted")
