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

# create the folder to save the data
create_folder("teste_8")
for _ in range(1):
    # init display with default values
    car.reset()
    screen, close = run_simluation(car, input)

    counter = 0

    # create a neural network
    new_model = tf.keras.models.load_model('model.keras')

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
        #input[0] = signal_freq_omega.calculate_signal()# 

        if position >= 35:
            input[0] = 1
        elif position <= -35:
            input[0] = -1
        else: 
            input[0] = 0

        #if input[0] != 0 or input[1] != 0:
        #    add_image_and_input_to_array(screen, input)


        # save the last 6 images
        image = pre_processes.crop_image_to_86x86(screen.copy())
        image = cv2.resize(image, (84, 84))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #last_images.append(image)


        testee = np.array(image, dtype=np.float32).reshape(-1, 84, 84, 1)/255.0

        #testee = testee[0:6]  # Seleciona as 6 primeiras imagens, se existirem
        #imagens_reformuladas = testee.reshape(1, 6, 84, 84, 1)

        saida = np.argmax(new_model.predict(testee[0:1]))

        print(saida)

        if saida == 8:
            input = [1, 1, 0]
        elif saida == 7:
            input = [0, 0, 0]
        elif saida == 6:
            input = [1, -1, 0]
        elif saida == 5:
            input = [0, 1, 0]
        elif saida == 4:
            input = [0, 0, 0]
        elif saida == 3:
            input = [0, -1, 0]
        elif saida == 2:
            input = [-1, 1, 0]
        elif saida == 1:
            input = [-1, 0, 0]
        elif saida == 0:
            input = [-1, -1, 0]

        input[1] = signal_freq_trans.calculate_signal() 

        # run the simulation
        screen, close = run_simluation(car, input)

        counter_loop += 1
        if close or quit:
            save_data()
            break