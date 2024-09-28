import cv2
import os

from controle import *

pre_processes = RecognizesRoad()

offset = 5087

arr_imagens = []
arr_inputs = []
name_folder = ''

def create_folder(folder):
    global name_folder
    if not os.path.exists('data/' + str(folder)):
        os.makedirs('data/' + str(folder))
        os.makedirs('data/' + str(folder) + '/pictures')
        print(f'Pasta "data/{str(folder)}" criada com sucesso!')
    else:
        print(f'A pasta "data/{str(folder)}" já existe.')

    name_folder = folder

def add_image_and_input_to_array(image, input):
    # adiciona a imagem e o input ao array
    image = pre_processes.crop_image_to_86x86(image)
    arr_imagens.append(image)
    arr_inputs.append(input.copy())

def save_data():
    arquivo = open('data/' + str(name_folder) + '/input_.csv', 'a')
    for i in range(len(arr_imagens)):

        # Salvar a imagem
        cv2.imwrite('data/' + str(name_folder) + '/pictures/imagem_' + str(i + offset) + '.png', arr_imagens[i])

        # adiciona o freio na mesma coluna
        if arr_inputs[i][2] == 1:
            arr_inputs[i][1] = -1

        # Salvar o input
        #arquivo.write(str(arr_inputs[i][0]) + "," + str(arr_inputs[i][1]) + "\n")
        arquivo.write('imagem_' + str(i + offset) + ',' + str(arr_inputs[i][0]) + "," + str(arr_inputs[i][1]) + "\n")

    print('Dados salvos com sucesso!')