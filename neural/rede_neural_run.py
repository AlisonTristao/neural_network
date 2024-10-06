from simulation import *   
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model  # type: ignore

import threading
import numpy as np
import cv2
from queue import Queue

import matplotlib.pyplot as plt

from data_logger import *

input = np.array([0.0, 0.0, 0.0])

LEN_GIF = 4

car = CarRacing(render_mode="human")
counter_loop = 0
counter = 0

model = load_model('model_trained.keras')

for _ in range(1):
    # init display with default values
    car.reset(seed=0, options={"randomize": False})
    screen, close = run_simluation(car, input)

    # save last frames
    last_frames = []

    while True:
        quit, input_nao_usado = register_input(input)

        # Pré-processar imagem
        screen = pre_processes.crop_image_to_84x84(screen)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        '''# Adiciona a imagem à lista de quadros
        last_frames.append(screen)

        # Mantém apenas os últimos 4 quadros
        if len(last_frames) > LEN_GIF:
            last_frames.pop(0)'''

        # Quando temos pelo menos 4 quadros, fazemos a previsão
        #if len(last_frames) == LEN_GIF:
            # Normalizando e preparando o array para o modelo
        #imagens_por_linha = np.array(last_frames, dtype=np.float32) / 255.0
        imagens_por_linha = np.expand_dims(screen, axis=0)/ 255.0  # Adiciona dimensão do batch
        imagens_por_linha = np.expand_dims(imagens_por_linha, axis=0)  # Adiciona dimensão do batch

        # predict the input
        input_pred = model.predict(imagens_por_linha, verbose=False)
        print("input pred before:", input_pred)
        input_pred = np.round(input_pred)
        print("input pred after:", input_pred)
        input[0] = input_pred[0][0]
        #input[1] = input_pred[0][1]
        input[1] = input_nao_usado[1]
        # run the simulation
        screen, close = run_simluation(car, input)

        counter_loop += 1
        if close or quit:
            break