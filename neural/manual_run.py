from controle import *
from data_logger import *
#from neural import *
from simulation import *   
import tensorflow as tf

from keras.models import load_model  # type: ignore

input = np.array([0.0, 0.0, 0.0])

car = CarRacing(render_mode="human")

# create the folder to save the data
create_folder("humano_1")

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

        counter += 1

        # salva as fotos sem zoom
        if counter >= 30:
            add_image_and_input_to_array(screen, input, car.true_speed)

        # run the simulation
        screen, close = run_simluation(car, input)

        if close or quit:
            save_data()
            break