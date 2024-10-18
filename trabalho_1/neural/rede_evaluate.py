import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from CustomAccuracy import *   
from data_logger import *

x_valid = 'data/teste_validacao_seed2/pictures'
y_valid = 'data/teste_validacao_seed2/input_.csv'

imagens_valid = load_images(x_valid, 1)
input_valid = load_csv(y_valid, 1)
# Load the trained model
model = load_model('model_trained.keras', custom_objects={'CustomAccuracy': CustomAccuracy})

imagens_valid = np.expand_dims(imagens_valid, axis=1) 
# Evaluate the model
#loss, accuracy = model.evaluate(imagens_valid, input_valid, verbose=0)
#print(f'Test Loss: {loss}')
#print(f'Test Accuracy: {accuracy}')

# minha avaliação
acertos = 0
for i in range(len(imagens_valid)):
    model_prediction = model.predict(imagens_valid[i])
   # print("prediction shape:", model_prediction.shape)
   # print("Probabilities for neuron1:", model_prediction[0][0]) #direcao
   # print("Probabilities for neuron2", model_prediction[0][1]) #acelerador
    actual_input = input_valid[i]
    #print("model prediction not normalized:", model_prediction)
    #aplica a threshold pros valores discretos
    if model_prediction[0][0] < -1/3:
        model_prediction[0][0] = -1.0
    elif model_prediction[0][0] > 1/3:
        model_prediction[0][0] = 1.0
    else:
        model_prediction[0][0] = 0.0

    if model_prediction[0][1] < -1/3:
        model_prediction[0][1] = -1.0
    elif model_prediction[0][1] > 1/3:
        model_prediction[0][1] = 1.0
    else:
        model_prediction[0][1] = 0.0

    #print("model prediction shape:",model_prediction.shape)
    #print("actual input shape:", actual_input.shape)
    if(model_prediction[0][0] == actual_input[0] and model_prediction[0][1] == actual_input[1]):
        #print("acertou")
        acertos+=1
    ##print("model prediction normalized:", model_prediction)
    #print("actual input:", actual_input)
acuracia = acertos/len(imagens_valid)
print(f'Minha Acuracia: {acuracia}')