import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
import csv
import os

import numpy as np

epoch = 1000

# Definir o caminho para a pasta com as imagens e input
pasta_imagens = 'data/teste_030030/pictures'
arquivo_csv = 'data/teste_030030/input_.csv'
pasta_imagens_valid = 'data/teste_030000/pictures'
arquivo_csv_valid = 'data/teste_030000/input_.csv'

imagens = []
input = []

imagens_valid = []
input_valid = []

# Iterar sobre os arquivos na pasta
for nome_arquivo in os.listdir(pasta_imagens):
    if nome_arquivo.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        caminho_completo = os.path.join(pasta_imagens, nome_arquivo)
        img = cv2.imread(caminho_completo)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imagens.append(img.copy())

# Abrir e ler o arquivo CSV de validação
with open(arquivo_csv, newline='', encoding='utf-8') as csvfile:
    leitor = csv.reader(csvfile)
    
    # Para pular as linhas, vamos contar as linhas lidas
    for i, linha in enumerate(leitor):
        input.append([linha[1], linha[2]])

        # Iterar sobre os arquivos na pasta
for nome_arquivo in os.listdir(pasta_imagens_valid):
    if nome_arquivo.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        caminho_completo = os.path.join(pasta_imagens_valid, nome_arquivo)
        img = cv2.imread(caminho_completo)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imagens_valid.append(img.copy())

# Abrir e ler o arquivo CSV de validação
with open(arquivo_csv_valid, newline='', encoding='utf-8') as csvfile:
    leitor = csv.reader(csvfile)
    
    # Para pular as linhas, vamos contar as linhas lidas
    for i, linha in enumerate(leitor):
        input_valid.append([linha[1], linha[2]])

# Convertendo para arrays NumPy e normalizando
imagens = np.array(imagens, dtype=np.float32)/255.0
input = np.array(input, dtype=np.float32)  # Rótulos como inteiros

imagens_valid = np.array(imagens_valid, dtype=np.float32)/255.0
input_valid = np.array(input_valid, dtype=np.float32)  # Rótulos como inteiros

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

states_valid = []
for i in range(len(input_valid)):
    saida = 0
    if input_valid[i][0] == 1 and input_valid[i][1] == 1:
        saida = [0, 0, 0, 0, 0, 1]
    elif input_valid[i][0] == 1 and input_valid[i][1] == 0:
        saida = [0, 0, 0, 0, 1, 0]
    elif input_valid[i][0] == 0 and input_valid[i][1] == 1:
        saida = [0, 0, 0, 1, 0, 0]
    elif input_valid[i][0] == 0 and input_valid[i][1] == 0:
        saida = [0, 0, 1, 0, 0, 0]
    elif input_valid[i][0] == -1 and input_valid[i][1] == 1:
        saida = [0, 1, 0, 0, 0, 0]
    elif input_valid[i][0] == -1 and input_valid[i][1] == 0:
        saida = [1, 0, 0, 0, 0, 0]

    states_valid.append(saida)

states = np.array(states, dtype=np.float32)
states_valid = np.array(states_valid, dtype=np.float32)

new_model = tf.keras.models.load_model('model_soft.keras')

new_model.summary()

# Treinamento do modelo com 20% dos dados de validação
history = new_model.fit(imagens, states, epochs=epoch, validation_data=(imagens_valid, states_valid))

# Plotando a acurácia do treinamento e da validação ao longo das épocas
plt.plot(history.history['accuracy'], label='Acurácia de Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

'''
new_model.evaluate(imagens, states)

# calcula quantos acertou 
predictions = new_model.predict(imagens)

acertos = 0

for i in range(len(predictions)):
    if np.argmax(predictions[i]) == np.argmax(states[i]):
        acertos += 1

print('Acertos:', acertos)
print('Acurácia:', acertos / len(predictions) * 100)'''