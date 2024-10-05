import cv2
import os
import csv

from controle import *

import tensorflow as tf

pre_processes = RecognizesRoad()

def contar_arquivos(name_folder):
    arquivos = os.listdir(name_folder)
        
    # Contar apenas os arquivos (não subpastas)
    return sum(os.path.isfile(os.path.join(name_folder, f)) for f in arquivos)

arr_imagens = []
arr_inputs = []
name_folder = ''

def create_folder(folder):
    global name_folder, arr_imagens, arr_inputs
    if not os.path.exists('data/' + str(folder)):
        os.makedirs('data/' + str(folder))
        os.makedirs('data/' + str(folder) + '/pictures')
        print(f'Pasta "data/{str(folder)}" criada com sucesso!')
    else:
        print(f'A pasta "data/{str(folder)}" já existe.')

    name_folder = folder

def add_image_and_input_to_array(image, input):
    # adiciona a imagem e o input ao array
    image = pre_processes.crop_image_to_84x84(image)
    arr_imagens.append(image)
    arr_inputs.append(input.copy())

def save_data():
    global name_folder, arr_imagens, arr_inputs

    offset = contar_arquivos('data/' + str(name_folder) + '/pictures')
    
    print("contador iniciando em:", offset)

    arquivo = open('data/' + str(name_folder) + '/input_.csv', 'a')
    for i in range(len(arr_imagens)):

        # Salvar a imagem
        cv2.imwrite('data/' + str(name_folder) + '/pictures/imagem_' + str(i + offset) + '.png', arr_imagens[i])

        # Salvar o input
        #arquivo.write(str(arr_inputs[i][0]) + "," + str(arr_inputs[i][1]) + "\n")
        arquivo.write('imagem_' + str(i + offset) + ',' + str(arr_inputs[i][0]) + "," + str(arr_inputs[i][1]) + "\n")

    # zera os arrays
    arr_imagens = []
    arr_inputs = []
    
    print('Dados salvos com sucesso!')

def load_images(pasta_imagens, len_gif=4):
    imagens = []
    nomes_imagens = [f for f in os.listdir(pasta_imagens) if f.startswith('imagem_') and f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    nomes_imagens.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for nome_arquivo in nomes_imagens:
        caminho_completo = os.path.join(pasta_imagens, nome_arquivo)
        img = cv2.imread(caminho_completo)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imagens.append(img.copy())
        
    imagens_por_linha = [imagens[i:i + len_gif] for i in range(0, len(imagens), len_gif)]
    
    return np.array(imagens_por_linha, dtype=np.float32)/255.0

def load_csv(arquivo_csv, len_gif=4):
    input = []
    with open(arquivo_csv, newline='', encoding='utf-8') as csvfile:
        leitor = csv.reader(csvfile)
        
        for i, linha in enumerate(leitor):
            input.append([linha[1], linha[2]])

    return np.array([input[i] for i in range(0, len(input), len_gif)], dtype=np.float32)

def accuracy(y_true, y_pred):  
    # Condição 1: y_pred[:, 0] > 1/3 e y_true[:, 0] == 1, idem para o segundo valor
    cond1 = tf.logical_and(tf.greater(y_pred[:, 0], 1/3), tf.equal(y_true[:, 0], 1))
    cond1 = tf.logical_and(cond1, tf.greater(y_pred[:, 1], 1/3))
    cond1 = tf.logical_and(cond1, tf.equal(y_true[:, 1], 1))

    # Condição 2: y_pred[:, 0] < -1/3 e y_true[:, 0] == -1, idem para o segundo valor
    cond2 = tf.logical_and(tf.less(y_pred[:, 0], -1/3), tf.equal(y_true[:, 0], -1))
    cond2 = tf.logical_and(cond2, tf.less(y_pred[:, 1], -1/3))
    cond2 = tf.logical_and(cond2, tf.equal(y_true[:, 1], -1))

    # Condição 3: -1/3 < y_pred[:, 0] < 1/3 e y_true[:, 0] == 0, idem para o segundo valor
    cond3 = tf.logical_and(tf.greater(y_pred[:, 0], -1/3), tf.less(y_pred[:, 0], 1/3))
    cond3 = tf.logical_and(cond3, tf.equal(y_true[:, 0], 0))
    cond3 = tf.logical_and(cond3, tf.greater(y_pred[:, 1], -1/3))
    cond3 = tf.logical_and(cond3, tf.less(y_pred[:, 1], 1/3))
    cond3 = tf.logical_and(cond3, tf.equal(y_true[:, 1], 0))

    correct_predictions = tf.logical_or(tf.logical_or(cond1, cond2), cond3)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy