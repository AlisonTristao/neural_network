import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar o Modelo
model = tf.keras.models.load_model('model_trained.keras')

# 2. Preparar a Imagem e o Valor Float
# Exemplo de uma imagem (1x81x84) - substitua com sua imagem real
image = np.random.rand(1, 1, 84, 84).astype(np.float32)  # Substitua com sua imagem real
image /= 255.0  # Normalização

# Exemplo de um valor float - substitua com seu valor real
float_value = np.array([[0.5]], dtype=np.float32)  # Substitua com seu valor real

# Converter a imagem e o valor float para tensores do TensorFlow
image_tensor = tf.convert_to_tensor(image)
float_tensor = tf.convert_to_tensor(float_value)

# 3. Função para Calcular o Mapa de Saliência
def get_saliency_map(model, image, float_value, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(image)
        tape.watch(float_value)
        
        # Crie um tensor que combine a imagem e o valor float
        input_data = [image, float_value]
        predictions = model(input_data)
        
        # Selecione a saída desejada com base no índice da classe
        loss = predictions[:, class_idx]
    
    # Gradientes em relação à imagem
    gradient = tape.gradient(loss, image)
    
    # Maximiza o gradiente ao longo dos canais
    gradient = tf.reduce_max(gradient, axis=-1)  # Reduzindo ao longo do eixo do canal (neste caso não tem canais)
    
    # Convertendo para numpy
    gradient = gradient.numpy()
    
    # Normalizando entre 0 e 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + tf.keras.backend.epsilon())
    
    return smap[0]  # Retornando apenas a primeira dimensão (81x84)

# 4. Gerar e Visualizar Saliency Maps para Todas as Classes
num_classes = 2  # Defina o número total de classes do seu modelo

# Criar subplots para as classes
plt.figure(figsize=(15, 5))

for class_idx in range(num_classes):
    # Gerar o mapa de saliência para a classe atual
    saliency_map = get_saliency_map(model, image_tensor, float_tensor, class_idx)
    
    # Plotar o mapa de saliência
    plt.subplot(1, num_classes, class_idx + 1)  # Ajustar para o número de classes
    plt.imshow(saliency_map, cmap='hot', aspect='auto')  # Garantindo a proporção correta
    plt.title(f'Classe {class_idx}')
    plt.axis('off')

plt.show()
