import os
import numpy as np
import cv2
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow import keras as k

print(tf.test.gpu_device_name())

# Lista todas as GPUs visíveis pelo TensorFlow
gpus = tf.config.list_physical_devices('GPU')

# Exibe a quantidade de GPUs detectadas
print(f"GPUs detectadas: {len(gpus)}")

# Exibe informações sobre cada GPU detectada
for gpu in gpus:
    print(gpu)

from keras import Model, Sequential
from keras.utils import to_categorical                                  # type: ignore
from keras.losses import categorical_crossentropy                       # type: ignore
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout  # type: ignore

class CarCloneNet(Sequential):
    def __init__(self, input_shape):
        super().__init__()

        self.add(Conv2D(32, kernel_size=(8,8), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal'))

        self.add(Conv2D(64, kernel_size=(4,4), strides= 2,
                        padding= 'valid', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(Conv2D(64, kernel_size=(3,3), strides= 2,
                        padding= 'valid', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        self.add(Flatten())
        self.add(Dense(512, activation= 'relu'))
        self.add(Dense(512, activation= 'relu'))
        self.add(Dense(2, activation= 'tanh'))

        self.add(Dropout(0.5))  # Adicionando uma camada Dropout
        self.add(Dense(2, activation='softmax'))  # Usando softmax na camada de saída
        
        self.compile(optimizer= tf.keras.optimizers.Adam(0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
model = CarCloneNet([84, 84, 1])

# training parameters
EPOCHS = 100
BATCH_SIZE = 2
image_height = 84
image_width = 84

# Definir o caminho para a pasta com as imagens e input
pasta_imagens = 'data/teste_1/pictures'
arquivo_csv = 'data/teste_1/input_.csv'

pasta_valid_img = 'data/teste_1/pictures'
arquivo_valid_csv = 'data/teste_1/input_.csv'

# Lista para armazenar as imagens
imagens = []
imagens_valid = []
input = []
input_valid = []

# Iterar sobre os arquivos na pasta
for nome_arquivo in os.listdir(pasta_imagens):
    # Verificar se o arquivo é uma imagem (png, jpg, jpeg, bmp, etc.)
    if nome_arquivo.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        caminho_completo = os.path.join(pasta_imagens, nome_arquivo)
        
        # Carregar a imagem usando cv2
        img = cv2.imread(caminho_completo)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Verificar se a imagem foi carregada corretamente
        if img is not None:
            # Adicionar a imagem (como array NumPy) à lista de imagens
            imagens.append(img)

# Iterar sobre os arquivos na pasta
for nome_arquivo in os.listdir(pasta_valid_img):
    # Verificar se o arquivo é uma imagem (png, jpg, jpeg, bmp, etc.)
    if nome_arquivo.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        caminho_completo = os.path.join(pasta_valid_img, nome_arquivo)
        
        # Carregar a imagem usando cv2
        img = cv2.imread(caminho_completo)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Verificar se a imagem foi carregada corretamente
        if img is not None:
            # Adicionar a imagem (como array NumPy) à lista de imagens
            imagens_valid.append(img)

# Abrir e ler o arquivo CSV
with open(arquivo_csv, newline='', encoding='utf-8') as csvfile:
    leitor_csv = csv.reader(csvfile)
    
    # Iterar sobre as cols do arquivo CSV
    for i, col in enumerate(leitor_csv):
        # Adicionar os valores das col 1 e 2 nas listas
        input.append([col[1], col[2]])

# Abrir e ler o arquivo CSV
with open(arquivo_valid_csv, newline='', encoding='utf-8') as csvfile:
    leitor_csv = csv.reader(csvfile)
    
    # Iterar sobre as cols do arquivo CSV
    for i, col in enumerate(leitor_csv):
        # Adicionar os valores das col 1 e 2 nas listas
        input_valid.append([col[1], col[2]])

# Se imagens for uma lista de strings, converta para um formato numérico apropriado
imagens = np.array(imagens, dtype=np.uint8)  # ou dtype adequado ao seu caso
input = np.array(input, dtype=np.uint8)  # ajuste conforme necessário

imagens_valid = np.array(imagens, dtype=np.uint8)  # ou dtype adequado ao seu caso
input_valid = np.array(input, dtype=np.uint8)  # ajuste conforme necessário

var = model.fit(x=imagens, y=input, validation_data=(imagens_valid, input_valid), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

model.summary()

print("ue")
