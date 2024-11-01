from controle import *
from data_logger import *
#from neural import *
from simulation import *   
import tensorflow as tf

from keras.models import load_model  # type: ignore

input = np.array([0.0, 0.0, 0.0])

car = CarRacing(render_mode="human")
retaCount1 = 0
curvaEsquerdaCount1 = 0
curvaDireitaCount1 = 0
retaCount2 = 0
curvaEsquerdaCount2 = 0
curvaDireitaCount2 = 0
maxCount = 200
# create the folder to save the data
create_folder("humano_valid")

#model = load_model('model_trained.keras')

for _ in range(10):
    # init display with default values
    car.reset()
    screen, close = run_simluation(car, input)

    # save frames count
    counter = 0

    last_frames = []

    while True:
        quit, input = register_input(input)

        counter += 1

        # salva as fotos sem zoom
        if counter >= 30:
            if(retaCount1 < maxCount and input[0] == 0 and car.true_speed < 30 and car.true_speed > 1.0):
                add_image_and_input_to_array(screen, input, car.true_speed)
                print("retaCount1", retaCount1)
                retaCount1 += 1
            elif(curvaEsquerdaCount1 < maxCount and input[0] == -1 and car.true_speed < 30 and car.true_speed > 1.0):
                add_image_and_input_to_array(screen, input, car.true_speed)
                print("curvaEsquerdaCount1", curvaEsquerdaCount1)
                curvaEsquerdaCount1 += 1
            elif(curvaDireitaCount1 < maxCount and input[0] == 1 and car.true_speed < 30 and car.true_speed > 1.0):
                add_image_and_input_to_array(screen, input, car.true_speed)
                curvaDireitaCount1 += 1
                print("curvaDireitaCount1", curvaDireitaCount1)
            elif(retaCount2 < maxCount and input[0] == 0 and car.true_speed > 30):
                add_image_and_input_to_array(screen, input, car.true_speed)
                print("retaCount2", retaCount2)
                retaCount2 += 1
            elif(curvaEsquerdaCount2 < 100 and input[0] == -1 and car.true_speed > 30):
                add_image_and_input_to_array(screen, input, car.true_speed)
                print("curvaEsquerdaCount2", curvaEsquerdaCount2)
                curvaEsquerdaCount2 += 1
            elif(curvaDireitaCount2 < maxCount and input[0] == 1 and car.true_speed > 30):
                add_image_and_input_to_array(screen, input, car.true_speed)
                curvaDireitaCount2 += 1
                print("curvaDireitaCount2", curvaDireitaCount2)

        # run the simulation
        screen, close = run_simluation(car, input)

        if close or quit:
            save_data()
            break