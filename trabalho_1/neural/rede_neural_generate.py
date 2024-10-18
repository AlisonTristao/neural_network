import tensorflow as tf
from keras import layers, models, layers, models, Input # type: ignore

IMG_H = 84
IMG_W = 84   
ADITIONAL_INPUT = 1
LEN_GIF = 4  

def create_model():
    # rede convolucionar
    input1 = Input(shape=(LEN_GIF, IMG_H, IMG_W)) 
    x1 = layers.Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='same', activation='sigmoid')(input1)
    x1 = layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='sigmoid')(x1)
    x1 = layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x1)
    x1 = layers.Flatten()(x1)
    
    # input auxiliar
    input2 = Input(shape=(ADITIONAL_INPUT,)) 

    # conbinacao das convolucoes com o input auxiliar
    combined = layers.Add()([x1, input2])

    # rede densa
    x = layers.Dense(512, activation='sigmoid')(combined)
    x = layers.Dense(512, activation='sigmoid')(x)
    output = layers.Dense(2, activation='tanh')(x)
    
    # output
    model = models.Model(inputs=[input1, input2], outputs=output)
    return model

model = create_model()
model.summary()
model.save('model.keras')
