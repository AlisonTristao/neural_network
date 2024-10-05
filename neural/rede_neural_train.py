import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model  # type: ignore

from data_logger import *

import numpy as np

x_train = 'data/teste_5/pictures'
y_train = 'data/teste_5/input_.csv'

x_valid = 'data/teste_6/pictures'
y_valid = 'data/teste_6/input_.csv'

# Dados para treino e validacao
imagens = load_images(x_train)
input = load_csv(y_train)
imagens_valid = load_images(x_valid)
input_valid = load_csv(y_valid)

# carrega o modelo e compila
model = load_model('model_trained_bom.keras')
model.compile(optimizer='adam', loss='mean_squared_error')

predictions = model.predict(imagens_valid)

# Converter predictions para valores discretos (-1, 0, ou 1)
predictions_discretas = np.round(predictions)

# Comparar predictions_discretas com input_valid
correspondencias = (predictions_discretas == input_valid)

# Calcular a porcentagem de correspondências
percentual_correspondencia = np.mean(correspondencias) * 100

print(f'As previsões correspondem aos valores de input_valid em {percentual_correspondencia:.2f}% dos casos.')

'''# treina o modelo
history = model.fit(imagens, input, epochs=200, batch_size=32, validation_data=(imagens_valid, input_valid))

# salva o modelo
model.save('model_trained_20FPS.keras')

# exibe os resultados
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
ax1.plot(history.history['loss'], label='Treinamento', color='blue')
ax1.plot(history.history['val_loss'], label='Validação', color='orange')
ax1.set_title('Loss')
ax1.set_xlabel('Épocas')
ax1.legend()
ax1.grid(True)
ax2.plot(history.history['accuracy'], label='Treinamento', color='green')
ax2.plot(history.history['val_accuracy'], label='Validação', color='red')
ax2.set_title('Accuracy')
ax2.set_xlabel('Épocas')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()'''