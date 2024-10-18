import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

def load_images(pasta_imagens, len_gif=4):
    imagens = []
    
    # Listar arquivos e filtrar apenas aqueles que começam com 'imagem_' e terminam com uma extensão de imagem
    nomes_imagens = [f for f in os.listdir(pasta_imagens) if f.startswith('imagem_') and f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Ordenar os nomes das imagens de acordo com o número no final do nome
    nomes_imagens.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    for nome_arquivo in nomes_imagens:
        caminho_completo = os.path.join(pasta_imagens, nome_arquivo)
        img = cv2.imread(caminho_completo)
        
        if img is not None:  # Verifica se a imagem foi carregada corretamente
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imagens.append(img.copy())
    
    # Dividir as imagens em sublistas de len_gif
    imagens_por_linha = [imagens[i:i + len_gif] for i in range(0, len(imagens), len_gif)]
    
    return np.array(imagens_por_linha, dtype=np.float32)

def load_csv(arquivo_csv, len_gif=4):
    input_data = []
    
    with open(arquivo_csv, newline='', encoding='utf-8') as csvfile:
        leitor = csv.reader(csvfile)
        
        for i, linha in enumerate(leitor):
            input_data.append([linha[1], linha[2]])

    # Agrupar input_data em sublistas de len_gif
    input_array = np.array([input_data[i] for i in range(0, len(input_data), len_gif)], dtype=object)
    
    return input_array

def main(pasta_imagens, arquivo_csv, len_gif=4):
    # Carregar imagens
    imagens = load_images(pasta_imagens, len_gif)
    
    # Carregar dados do CSV
    dados_csv = load_csv(arquivo_csv, len_gif)
    
    # Exibir resultados
    for i in range(len(imagens)):
        # Exibir imagens
        plt.figure(figsize=(10, 5))
        for j in range(len(imagens[i])):
            plt.subplot(1, len(imagens[i]), j + 1)
            plt.imshow(imagens[i][j], cmap='gray')
            plt.axis('off')  # Desliga os eixos
        plt.suptitle(f"Imagens {i}")
        plt.show()
        
        # Imprimir dados do CSV no terminal
        print(f"Dados do CSV {i}: {dados_csv[i]}")

# Exemplo de uso
main('data/teste_1/pictures', 'data/teste_1/input_.csv', len_gif=4)
