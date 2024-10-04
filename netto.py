import tensorflow as tf
from keras import layers, models, applications
import numpy as np
import csv
import cv2
import os

# Parâmetros da imagem
IMG_HEIGHT = 84
IMG_WIDTH = 84
IMG_CHANNELS = 1

base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
for layer in base_model.layers:
        layer.trainable = True
        
model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2'),
        #tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.Dense(6, activation='softmax')  
    ])

# Compilando o modelo
# Compilar o modelo com Adam e taxa de aprendizado ajustada
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Definir o caminho para a pasta com as imagens e input
pasta_imagens = 'data/teste_030030/pictures'
arquivo_csv = 'data/teste_030030/input_.csv'

imagens = []
input = []

# Iterar sobre os arquivos na pasta
for nome_arquivo in os.listdir(pasta_imagens):
    if nome_arquivo.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        caminho_completo = os.path.join(pasta_imagens, nome_arquivo)
        img = cv2.imread(caminho_completo)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

states = []
for i in range(len(input)):
    saida = 0
    if input[i][0] == 1 and input[i][1] == 1:
        saida = [0, 0, 0, 0, 0, 1]
    elif input[i][0] == 1 and input[i][1] == 0:
        saida = [0, 0, 0, 0, 1, 0]
    elif input[i][0] == 0 and input[i][1] == 1:
        saida = [0, 0, 0, 1, 0, 0]
    elif input[i][0] == 0 and input[i][1] == 0:
        saida = [0, 0, 1, 0, 0, 0]
    elif input[i][0] == -1 and input[i][1] == 1:
        saida = [0, 1, 0, 0, 0, 0]
    elif input[i][0] == -1 and input[i][1] == 0:
        saida = [1, 0, 0, 0, 0, 0]

    states.append(saida)

states = np.array(states, dtype=np.float32)

#model = tf.keras.models.load_model('model_soft.keras')

for i in range(100):
    model.fit(imagens, states, epochs=250, batch_size=10)
    model.save('model_soft.keras')