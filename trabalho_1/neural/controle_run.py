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

# create the folder to save the data
create_folder("teste_seed_1")

#model = load_model('model_trained.keras')

for _ in range(9):
    # init display with default values
    car.reset(seed=1)
    screen, close = run_simluation(car, input)

    # save frames count
    counter = 0

    last_frames = []

    while True:
        quit, input = register_input(input)

        position, future_track = sensor_theta.calculate_position(screen)

        if counter_loop >= control_freq:
            counter_loop = 0

            set_point = 20
            if(future_track[0] == 1 and abs(position) < 50):
                set_point = 40 #- abs(position/2)          

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
            add_image_and_input_to_array(screen, input, car.true_speed)
            
        counter += 1

        input[1] = signal_freq_trans.calculate_signal() 

        # run the simulation
        screen, close = run_simluation(car, input)

        counter_loop += 1
        if close or quit:
            save_data()
            break