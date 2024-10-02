import tensorflow as tf
from keras import layers, models
import numpy as np
import csv
import cv2
import os

# Parâmetros da imagem
IMG_HEIGHT = 84
IMG_WIDTH = 84
IMG_CHANNELS = 1

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
    layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),  # Usando Conv3D
    layers.MaxPooling2D(pool_size=(2, 2)),
    #layers.Conv2D(32, (4, 4), strides=(2, 2), activation='tanh'),
    layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu'),
    layers.Flatten(),
    #layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    #layers.Dropout(0.5), 
    #layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='tanh'),
    #layers.Dense(9, activation='softmax')
])

# Compilando o modelo
# Compilar o modelo com Adam e taxa de aprendizado ajustada
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

# Definir o caminho para a pasta com as imagens e input
pasta_imagens = 'data/teste_mamada/pictures'
arquivo_csv = 'data/teste_mamada/input_.csv'

imagens = []
input = []

# Iterar sobre os arquivos na pasta
for nome_arquivo in os.listdir(pasta_imagens):
    if nome_arquivo.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        caminho_completo = os.path.join(pasta_imagens, nome_arquivo)
        img = cv2.imread(caminho_completo)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imagens.append(img)

# Abrir e ler o arquivo CSV de validação
with open(arquivo_csv, newline='', encoding='utf-8') as csvfile:
    leitor = csv.reader(csvfile)
    
    # Para pular as linhas, vamos contar as linhas lidas
    for i, linha in enumerate(leitor):
        input.append([linha[1], linha[2]])

# Convertendo para arrays NumPy e normalizando
imagens = np.array(imagens, dtype=np.float32)/255.0
input = np.array(input, dtype=np.float32)  # Rótulos como inteiros

model.fit(imagens, input, epochs=2000, batch_size=32)

model.save('tanh_max.keras')
