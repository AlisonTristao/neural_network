import tensorflow as tf
from keras import layers, models

# Parâmetros da imagem
IMG_HEIGHT = 86
IMG_WIDTH = 86
IMG_CHANNELS = 1

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
    layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),  
    #layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    #layers.Dense(2, activation='tanh'),
    layers.Dense(6, activation='softmax')
])

# Compilando o modelo
# Compilar o modelo com Adam e taxa de aprendizado ajustada
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.save('model_soft.keras')

