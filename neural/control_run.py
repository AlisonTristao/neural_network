from controle import *
from data_logger import *
#from neural import *
from simulation import *   
import tensorflow as tf

from keras.models import load_model  # type: ignore

control_time = 0.01 # 10ms
control_freq = control_time * FPS
counter_loop = 0
input = np.array([0.0, 0.0, 0.0])

sensor_theta = RecognizesRoad()
control_omega = Control(kp=0.9, kd=0.2, freq=FPS)
control_trans = Control(kp=5, ki=0.01, freq=FPS) 

signal_freq_omega = FrequenceSignal(control_freq)
signal_freq_trans = FrequenceSignal(control_freq)

car = CarRacing(render_mode="human")

LEN_GIF = 4

# create the folder to save the data
create_folder("teste_validacao_seed4")

#model = load_model('model_trained.keras')

for _ in range(1):
    # init display with default values
    car.reset()
    screen, close = run_simluation(car, input)

    # save frames count
    counter = 0

    last_frames = []

    while True:
        quit, input = register_input(input)

        position, future_track = sensor_theta.calculate_position(screen)

        # Pré-processar imagem
        screen_2 = pre_processes.crop_image_to_84x84(screen.copy())
        screen_2 = cv2.cvtColor(screen_2, cv2.COLOR_BGR2GRAY)
        
        # Adiciona a imagem à lista de quadros
        last_frames.append(screen_2)

        # Mantém apenas os últimos 4 quadros
        if len(last_frames) > LEN_GIF:
            last_frames.pop(0)

        counter += 1

        # Quando temos pelo menos 4 quadros, fazemos a previsão
        if len(last_frames) == LEN_GIF:
            # Normalizando e preparando o array para o modelo
            imagens_por_linha = np.array(last_frames, dtype=np.float32) / 255.0
            imagens_por_linha = np.expand_dims(imagens_por_linha, axis=0)  # Adiciona dimensão do batch

            # predict the input
            #pred = model.predict(imagens_por_linha, verbose=False)
            #input_pred = np.round(pred)

        if counter_loop >= control_freq:
            counter_loop = 0

            set_point = 20
            if(future_track[0] == 1 and abs(position) < 50):
                set_point = 25 #- abs(position/2)          

            # set the signal to the frequence signal 
            trans = control_trans.pi_control(set_point - car.true_speed)
            if(trans > 0):
                signal_freq_trans.set_signal(trans)
            else:
                signal_freq_trans.set_signal(0)

        if position >= 35:
            input[0] = 1
        elif position <= -35:
            input[0] = -1
        else: 
            input[0] = 0

        # salva as fotos sem zoom
        if counter >= 30:
            add_image_and_input_to_array(screen, input)
            
        counter += 1

        input[1] = signal_freq_trans.calculate_signal() 

        # run the simulation
        screen, close = run_simluation(car, input)

        counter_loop += 1
        if close or quit:
            save_data()
            break