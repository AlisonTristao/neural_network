import tensorflow as tf
from keras import layers, models
import numpy as np
import csv
import cv2
import os

# Parâmetros da imagem
QTD_IMAGENS = 6
IMG_HEIGHT = 84
IMG_WIDTH = 84
IMG_CHANNELS = 1

model = models.Sequential([
    layers.Input(shape=(QTD_IMAGENS, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
    layers.Conv3D(32, (1, 8, 8), strides=(1, 4, 4), activation='relu'),  # Usando Conv3D
    layers.MaxPooling3D(pool_size=(1, 2, 2)),
    layers.Conv3D(64, (1, 4, 4), strides=(1, 2, 2), activation='relu'),
    layers.Conv3D(64, (1, 3, 3), strides=(1, 2, 2), activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.5), 
    layers.Dense(512, activation='relu', kernel_regularizer='l2'),
    #layers.Dense(2, activation='relu'),
    layers.Dense(9, activation='softmax')
])

# Compilando o modelo
# Compilar o modelo com Adam e taxa de aprendizado ajustada
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Definir o caminho para a pasta com as imagens e input
pasta_imagens = 'data/teste_0/pictures'
arquivo_csv = 'data/teste_0/input_.csv'

imagens = []
input = []

# Iterar sobre os arquivos na pasta
for nome_arquivo in os.listdir(pasta_imagens):
    if nome_arquivo.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        caminho_completo = os.path.join(pasta_imagens, nome_arquivo)
        img = cv2.imread(caminho_completo)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar a imagem
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        if img is not None:
            imagens.append(img)

# Abrir e ler o arquivo CSV de validação
with open(arquivo_csv, newline='', encoding='utf-8') as csvfile:
    leitor = csv.reader(csvfile)
    
    # Para pular as linhas, vamos contar as linhas lidas
    for i, linha in enumerate(leitor):
        # Adiciona ao array apenas quando a linha é múltipla de 4 (ou seja, 0, 4, 8, ...)
        if (i+1) % QTD_IMAGENS == 0:
            input.append([linha[1], linha[2]])

# Convertendo para arrays NumPy e normalizando
imagens = np.array(imagens, dtype=np.float32).reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS) / 255.0
input = np.array(input, dtype=np.float32)  # Rótulos como inteiros

print(len(imagens))

states = []
for i in range(len(input)):
    saida = 0
    if input[i][0] == 1 and input[i][1] == 1:
        saida = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif input[i][0] == 1 and input[i][1] == 0:
        saida = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif input[i][0] == 1 and input[i][1] == -1:
        saida = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif input[i][0] == 0 and input[i][1] == 1:
        saida = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif input[i][0] == 0 and input[i][1] == 0:
        saida = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif input[i][0] == 0 and input[i][1] == -1:
        saida = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif input[i][0] == -1 and input[i][1] == 1:
        saida = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif input[i][0] == -1 and input[i][1] == 0:
        saida = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif input[i][0] == -1 and input[i][1] == -1:
        saida = [1, 0, 0, 0, 0, 0, 0, 0, 0]

    states.append(saida)

states = np.array(states, dtype=np.float32)

# Reshape: calculamos o novo número de linhas baseado na quantidade total de imagens
num_imagens = imagens.shape[0]
novas_linhas = num_imagens // QTD_IMAGENS + (num_imagens % QTD_IMAGENS > 0)  # Adiciona uma linha extra se houver imagens restantes

# Reshape para 4 imagens por linha
imagens_reformuladas = imagens.reshape(novas_linhas, QTD_IMAGENS, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model.fit(imagens_reformuladas, states, epochs=10000, batch_size=500)

model.save('40k.keras')
