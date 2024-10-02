import tensorflow as tf

import cv2
import csv
import os

import numpy as np

new_model = tf.keras.models.load_model('model.keras')

# Parâmetros da imagem
QTD_IMAGENS = 6
IMG_HEIGHT = 84
IMG_WIDTH = 84
IMG_CHANNELS = 1

# Show the model architecture
new_model.summary()

# Definir o caminho para a pasta com as imagens e input
pasta_imagens = 'data/teste_4/pictures'
arquivo_csv = 'data/teste_4/input_.csv'

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

'''# Abrir e ler o arquivo CSV de validação
with open(arquivo_csv, newline='', encoding='utf-8') as csvfile:
    leitor = csv.reader(csvfile)
    
    # Para pular as linhas, vamos contar as linhas lidas
    for i, linha in enumerate(leitor):
        # Adiciona ao array apenas quando a linha é múltipla de 4 (ou seja, 0, 4, 8, ...)
        if (i+1) % QTD_IMAGENS == 0:
            input.append([linha[1], linha[2]])'''

# Abrir e ler o arquivo CSV de validação
with open(arquivo_csv, newline='', encoding='utf-8') as csvfile:
    leitor = csv.reader(csvfile)
    
    # Para pular as linhas, vamos contar as linhas lidas
    for i, linha in enumerate(leitor):
        # Adiciona ao array apenas quando a linha é múltipla de 4 (ou seja, 0, 4, 8, ...)
        if (i+1) % QTD_IMAGENS == 0:
            input.append([linha[1], linha[2]])

# Convertendo para arrays NumPy e normalizando
print(len(imagens))
imagens = np.array(imagens, dtype=np.float32).reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS) / 255.0
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

# Reshape: calculamos o novo número de linhas baseado na quantidade total de imagens
num_imagens = imagens.shape[0]
novas_linhas = num_imagens // QTD_IMAGENS + (num_imagens % QTD_IMAGENS > 0)  # Adiciona uma linha extra se houver imagens restantes

# Reshape para 4 imagens por linha
imagens_reformuladas = imagens.reshape(novas_linhas, QTD_IMAGENS, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

print(imagens_reformuladas.shape)

new_model.evaluate(imagens_reformuladas, states)

#imagens_reformuladas = imagens_reformuladas.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

test = imagens_reformuladas[0:1]

print(test.shape)

# Prever as saídas com apenas umaimgaem
pred = new_model.predict(test)

for i in range(len(pred)):
    # retorna o argmax da pred
    print('Predição: {:.2f} Estado: {:.2f}'.format(np.argmax(pred[i]), np.argmax(states[i])))