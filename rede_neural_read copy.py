import tensorflow as tf

import cv2
import csv
import os

import numpy as np

new_model = tf.keras.models.load_model('tanh_max.keras')

IMG_HEIGHT = 84
IMG_WIDTH = 84
IMG_CHANNELS = 1

# Custom callback to stop training when accuracy gets too high
class StopOnHighAccuracy(tf.keras.callbacks.Callback):
    def _init_(self, threshold=0.95):  # Set your accuracy threshold
        super(StopOnHighAccuracy, self)._init_()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        # Check the training or validation accuracy
        accuracy = logs.get('accuracy')  # Use 'val_accuracy' for validation accuracy
        if accuracy >= self.threshold:
            print(f"\nStopping training as accuracy reached {accuracy:.4f}")
            self.model.stop_training = True

stop_on_high_accuracy = StopOnHighAccuracy(threshold=0.95)

# Show the model architecture
new_model.summary()

# Definir o caminho para a pasta com as imagens e input
pasta_imagens = 'data/teste_mamada_2/pictures'
arquivo_csv = 'data/teste_mamada_2/input_.csv'

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
        # Adiciona ao array apenas quando a linha é múltipla de 4 (ou seja, 0, 4, 8, ...)
        input.append([linha[1], linha[2]])

# Convertendo para arrays NumPy e normalizando
imagens = np.array(imagens, dtype=np.float32)/255.0
input = np.array(input, dtype=np.float32)  # Rótulos como inteiros

new_model.evaluate(imagens, input)

# Predict the output
predictions = new_model.predict(imagens)

def categorize_value(value):
    if value < -0.10:
        return -1
    elif value > 0.10:
        return 1
    else:
        return 0

# Inicializando o contador de previsões corretas
correct_predictions = 0

for i, pred in enumerate(predictions):
    # Aplica a categorização tanto para as previsões quanto para os valores reais
    pred_categorized = [categorize_value(pred[0]), categorize_value(pred[1])]
    real_categorized = [categorize_value(input[i][0]), categorize_value(input[i][1])]

    # Verifica se as previsões categorizadas são iguais aos valores reais categorizados
    match = pred_categorized == real_categorized

    # Incrementa o contador se os valores forem iguais
    if match:
        correct_predictions += 1

    # Print formatado com largura fixa e adicionando o resultado da comparação
    print(f'Predicted: {pred_categorized[0]:>3} {pred_categorized[1]:>3}   Real: {real_categorized[0]:>3} {real_categorized[1]:>3}   {"True" if match else "False"}')

# Print do total de previsões corretas
print(f'\nTotal de previsões corretas: {correct_predictions}')
print(f'\nAccuracy: {correct_predictions/len(predictions)*100:.2f}%')