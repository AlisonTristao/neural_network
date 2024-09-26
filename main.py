from controle import *
from data_logger import *
#from neural import *
from simulation import *   

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
create_folder("teste_1")

# init display with default values
car.reset()
screen, close = run_simluation(car, input)

while True:
    quit, input = register_input(input)

    position, future_track = sensor_theta.calculate_position(screen)

    # calculate the control signal
    if counter_loop >= control_freq:
        counter_loop = 0

        # set the signal to the frequence signal to omega
        signal_freq_omega.set_signal(control_omega.pd_control(position))

        set_point = 50
        if(future_track[0] == 1 and abs(position) < 50):
            set_point = 60 - abs(position)          

        # set the signal to the frequence signal 
        signal_freq_trans.set_signal(control_trans.pi_control(set_point - car.true_speed))

    # apply pwm signal to the car
    input[0] = signal_freq_omega.calculate_signal()
    input[1] = signal_freq_trans.calculate_signal()

    # run the simulation
    screen, close = run_simluation(car, input)

    # save the data
    add_image_and_input_to_array(screen, input)

    counter_loop += 1
    if close or quit:
        save_data()
        break