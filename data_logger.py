import cv2
import os

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
    arr_imagens.append(image)
    arr_inputs.append(input.copy())

def save_data():
    arquivo = open('data/' + str(name_folder) + '/input_.csv', 'w')
    for i in range(len(arr_imagens)):

        # Salvar a imagem
        cv2.imwrite('data/' + str(name_folder) + '/pictures/imagem_' + str(i) + '.png', arr_imagens[i])

        # Salvar o input
        #arquivo.write(str(arr_inputs[i][0]) + "," + str(arr_inputs[i][1]) + "\n")
        arquivo.write('imagem_' + str(i)+ ',' + str(arr_inputs[i][0]) + "," + str(arr_inputs[i][1]) + "," + str(arr_inputs[i][2]) + "\n")

    print('Dados salvos com sucesso!')