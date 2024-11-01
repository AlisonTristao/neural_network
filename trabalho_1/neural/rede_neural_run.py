from simulation import *   
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model  # type: ignore
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data_logger import *
import time

input = np.array([0.0, 0.0, 0.0])

LEN_GIF = 1

car = CarRacing(render_mode="human")
counter_loop = 0

model = load_model('model_trained_random.keras')

for _ in range(1):
    # init display with default values
    car.reset(seed=1)
    screen, close = run_simluation(car, input)

    # save last frames
    last_frames = []

    while True:
        quit, input_nao_usado = register_input(input)

        # Pré-processar imagem
        screen = pre_processes.crop_image_to_84x84(screen)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        # Adiciona a imagem à lista de quadros
        last_frames.append(screen)

        # Mantém apenas os últimos 4 quadros
        if len(last_frames) > LEN_GIF:
            last_frames.pop(0)

        # Quando temos pelo menos 4 quadros, fazemos a predição
        if len(last_frames) == LEN_GIF:
            # Normalizando e preparando o array para o modelo
            #imagens_por_linha = np.array(last_frames, dtype=np.float32) / 255.0
            imagens_por_linha = np.expand_dims(screen, axis=0)/ 255.0 
            imagens_por_linha = np.expand_dims(imagens_por_linha, axis=0)  # Adiciona dimensão do batch

            #comente do if até o else quando for com velocidade fixa em 0
            if LEN_GIF == 1:
                speed = np.array([car.true_speed/160], dtype=np.float32)
            else:
                speed = 0.0

            speed = np.expand_dims(speed, axis=0)

            # predict the input
            input_pred = np.round(model.predict([imagens_por_linha, speed], verbose=False))
            input[0] = input_pred[0][0]
            input[1] = input_pred[0][1]

        # run the simulation
        screen, close = run_simluation(car, input)

        counter_loop += 1
        if close or quit:
            time.sleep(5)  # Sleep for 1 second
            break