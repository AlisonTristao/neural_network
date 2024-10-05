import tensorflow as tf
from keras import layers, models
import numpy as np

IMG_H = 84
IMG_W = 84   
LEN_GIF = 4  

def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(LEN_GIF, IMG_H, IMG_W)))
    model.add(layers.Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))

    model.add(layers.Flatten()) 
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(2, activation='tanh'))

    return model

model = create_model()
model.summary()
model.save('model.keras')
