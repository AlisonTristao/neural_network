import numpy as np

class RecognizesRoad:
    def __init__(self):
        self.last_value = 0
    
    # converte a imagem RGB gerada para uma matrix de 0 a 1
    def rgb_to_bw_binary(self, image_array, threshold=128):
        gray_image = np.mean(image_array, axis=-1)
        return abs(1 - ((gray_image > threshold).astype(np.uint8)))  # 1 para preto e 0 para branco
    
    # recebe uma imagem de 96x96
    def crop_image_to_86x86(self, image_array):
        return image_array[6:90, 6:90]

    # captura a parte da imagem que contém a pista levemente a frente do carro
    def crop_image(self, binary_image, top_left, size):
        x, y = top_left
        width = size
        futere_track = binary_image[y-50:y-49, x+13:x+14].flatten()
        return binary_image[y-10:y-9, x:x+width].flatten(), futere_track

    # calcula a média ponderada do vetor para saber a posição do carro na pista
    def weighted_mean(self, vector):
        # Cria um array de pesos que são as posições (começando em 1)
        weights = np.arange(1, len(vector) + 1) * 200
        
        # media
        weighted_sum = np.sum(vector * weights)
        total_weight = np.sum(vector)
        
        # satura a media
        if total_weight != 0:
            self.last_value = ((weighted_sum / total_weight) / (len(vector) + 1)) - 100
        else:
            if self.last_value <= 0:
                self.last_value = -100
            else:
                self.last_value = 100

        return self.last_value

    def calculate_position(self, screen):
        binary_image = self.rgb_to_bw_binary(screen)
        # caso mude o tamanho ta tela, talvez tenha q mudar isso aqui
        cropped_image, future_track = self.crop_image(binary_image, top_left=(35, 52), size=26)
        return self.weighted_mean(cropped_image), future_track

class Control:
    def __init__(self, kp = 0, ki = 0, kd = 0, freq = 0):
        self.pd_consts = np.array([kp, ki, kd])
        self.freq = freq
        self.error_ant = 0.0
        self.integral = 0.0

    def pd_control(self, error):
        kp = self.pd_consts[0]
        kd = self.pd_consts[2]

        control = (kp * error) + (kd * (error - self.error_ant) * self.freq)

        if control > 100:
            control = 100
        elif control < -100:
            control = -100

        self.error_ant = error
        return control
    
    def pi_control(self, error):
        kp = self.pd_consts[0]
        ki = self.pd_consts[1]

        self.integral += error

        control = (kp * error) + (ki * self.integral)

        if control > 100:
            control = 100
        elif control < -100:
            control = -100

        return control
    
class FrequenceSignal:
    def __init__(self, freq):
        self.freq = freq
        self.signal = 0.0
        self.counter = 0

    def set_signal(self, sig):
        self.signal = round((sig/10 * self.freq))
        ##print("sig {:.3f}, self.freq: {:.3f},self.signal: {:.3f}".format(sig,self.freq,self.signal))
    
    def get_signal(self):
        return self.signal

    def calculate_signal(self):
        # count loop
        self.counter += 1
        
        # reset counter
        if self.counter >= self.freq:
            self.counter = 0
        
        # return signal
        if self.counter >= abs(self.signal):
            return 0
        else:
            if self.signal < 0:
                return -1
            else:   
                return +1