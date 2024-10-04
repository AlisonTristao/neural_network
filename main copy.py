from controle import *
from data_logger import *
#from neural import *
from simulation import *   
import tensorflow as tf

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

# salva as 6 ultimas imagens
last_images = []

counter = 0
contador_amostras = [0, 0, 0]
QTD_AMOSTRAS = 333

parada = False

# create the folder to save the data
create_folder("teste_030030")

while not parada:
    # init display with default values
    car.reset()
    screen, close = run_simluation(car, input)

    while True:
        quit, input = register_input(input)

        position, future_track = sensor_theta.calculate_position(screen)

        if counter_loop >= control_freq:
            counter_loop = 0

            # set the signal to the frequence signal to omega
            #signal_freq_omega.set_signal(control_omega.pd_control(position))

            set_point = 40
            if(future_track[0] == 1 and abs(position) < 50):
                set_point = 55 - abs(position/2)          

            # set the signal to the frequence signal 
            trans = control_trans.pi_control(set_point - car.true_speed)
            if(trans > 0):
                signal_freq_trans.set_signal(trans)
            else:
                signal_freq_trans.set_signal(0)

        # apply pwm signal to the car
        input[1] = signal_freq_trans.calculate_signal() 
        #input[0] = signal_freq_omega.calculate_signal()

        if position >= 35:
            input[0] = 1
        elif position <= -35:
            input[0] = -1
        else: 
            input[0] = 0

        if counter >= 30:
            input_invevrtido = input.copy()
            input_invevrtido[0] = input_invevrtido[0] * -1

            if input[0] == -1 and contador_amostras[0] < QTD_AMOSTRAS:
                add_image_and_input_to_array(cv2.flip(screen, 1), input_invevrtido)
                add_image_and_input_to_array(screen, input)
                contador_amostras[0] += 1
                print("curva -1")
            elif input[0] == 0 and contador_amostras[1] < QTD_AMOSTRAS:
                add_image_and_input_to_array(cv2.flip(screen, 1), input_invevrtido)
                add_image_and_input_to_array(screen, input)
                contador_amostras[1] += 1
                print("reta 0")
            elif input[0] == 1 and contador_amostras[2] < QTD_AMOSTRAS:
                add_image_and_input_to_array(cv2.flip(screen, 1), input_invevrtido)
                add_image_and_input_to_array(screen, input)
                contador_amostras[2] += 1
                print("curva 1")
            elif contador_amostras[0] == QTD_AMOSTRAS and contador_amostras[1] == QTD_AMOSTRAS and contador_amostras[2] == QTD_AMOSTRAS:
                parada = True
        
        counter += 1

        # run the simulation
        screen, close = run_simluation(car, input)

        counter_loop += 1
        if close or quit:
            save_data()
            break