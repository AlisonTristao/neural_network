import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model  # type: ignore

from data_logger import *

import numpy as np

x_train = 'data/teste_1/pictures'
y_train = 'data/teste_1/input_.csv'

x_valid = 'data/teste_0/pictures'
y_valid = 'data/teste_0/input_.csv'

# Dados para treino e validacao
imagens = load_images(x_train, 1)
input = load_csv(y_train, 1)
imagens_valid = load_images(x_valid, 1)
input_valid = load_csv(y_valid, 1)

# carrega o modelo e compila
model = load_model('model.keras')
model.compile(optimizer='adam', loss='mean_squared_error')

'''predictions = model.predict(imagens_valid)

# Converter predictions para valores discretos (-1, 0, ou 1)
predictions_discretas = np.round(predictions)

# Comparar predictions_discretas com input_valid
correspondencias = (predictions_discretas == input_valid)

# Calcular a porcentagem de correspondências
percentual_correspondencia = np.mean(correspondencias) * 100

print(f'As previsões correspondem aos valores de input_valid em {percentual_correspondencia:.2f}% dos casos.')'''

# treina o modelo
history = model.fit(imagens, input, epochs=1000, batch_size=32, validation_data=(imagens_valid, input_valid))

# salva o modelo
model.save('model_trained.keras')

# exibe os resultados
plt.plot(history.history['loss'], label='Treinamento', color='blue')
plt.plot(history.history['val_loss'], label='Validação', color='orange')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()