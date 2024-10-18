import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model  # type: ignore
from data_logger import *
import numpy as np
from CustomAccuracy import *

LEN_GIF = 4  

x_train = 'data/teste_seed_1/pictures'
y_train = 'data/teste_seed_1/input_.csv'

x_valid = 'data/teste_seed_2/pictures'
y_valid = 'data/teste_seed_2/input_.csv'

# dados para treino e validacao
imagens = load_images(x_train, LEN_GIF)
input, speed = load_csv(y_train, LEN_GIF)
imagens_valid = load_images(x_valid, LEN_GIF)
input_valid, speed_valid = load_csv(y_valid, LEN_GIF)

# carrega o modelo e compila
model = load_model('model.keras', custom_objects={'CustomAccuracy': CustomAccuracy})
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[CustomAccuracy()])

# treina o modelo
history = model.fit([imagens, speed], input, epochs=100, batch_size=32, validation_data=([imagens_valid, speed_valid], input_valid)).history

# salva o modelo
model.save('model_trained_4frames.keras')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# primeiro gráfico: Loss
ax1.plot(history['loss'], label='Loss-train', color='blue')
ax1.plot(history['val_loss'], label='Loss-valid', color='orange')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# segundo gráfico: Accuracy
ax2.plot(history['custom_accuracy'], label='Accuracy-train', color='green')
ax2.plot(history['val_custom_accuracy'], label='Accuracy-valid', color='red')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('plots/treinamento_dividido_1.png')
plt.show()