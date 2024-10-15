import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def direction(action):
    match action:
        case 'up':
            return '↑'
        case 'down':
            return '↓'
        case 'right':
            return '→'
        case 'left':
            return '←'

def world_limits(state):
    # eixo x nao pode ser negativo e meor q WORLD_HEIGHT
    state[X] = max(0, state[X])
    state[X] = min(WORLD_WIDTH - 1, state[X])
    # eixo y nao pode ser negativo e menor q WORLD_WIDTH
    state[Y] = max(0, state[Y])
    state[Y] = min(WORLD_HEIGHT - 1, state[Y])
    return state

def step(init_state, action):
    # calcula o proximo estado de acordo com a acao
    i, j = init_state
    match action:
        case 0:         # UP
            j += 1
        case 2:         # LEFT
            i -= 1
        case 3:         # RIGHT
            i += 1
        case 1:         # DOWN
            j -= 1
        case _:
            print('Invalid action')
            return init_state, -1

    # garante que nao saia do mundo
    next_state = world_limits([i, j])

    # calcula a recompensa
    reward = -1
    
    # se o estado for de penalidade, volta para o inicio e perde 100 pontos
    if next_state in PENALITY_STATES:
        reward = -100
        next_state = START
        #print("penalidade aplicada")

    return next_state, reward

def plot_world(world_height, world_width, start, finish, penality_states, q_values, states_arr=[], epoch=0, steps=0, rewards=0):
    # gera o mundo com casas normais
    world = np.zeros((world_height, world_width))

    fig, ax = plt.subplots(figsize=(world_width, world_height))

    # casas especiais
    world[start[1], start[0]] = 1               # inicio = 1
    ax.text(start[0] + 0.5, start[1] + 0.1, 'start', va='center', ha='center', fontsize=8)

    world[finish[1], finish[0]] = 2             # fim = 2
    ax.text(finish[0] + 0.5, finish[1] + 0.1, 'finish', va='center', ha='center', fontsize=8)

    for penality in penality_states:
        world[penality[1], penality[0]] = -1    # penality = -1
        ax.text(penality[0] + 0.5, penality[1] + 0.1, 'cliff', va='center', ha='center', fontsize=8)

    # cores para cada estado
    cmap = plt.cm.colors.ListedColormap(['red', 'lightgray', 'yellow', 'green'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(world, cmap=cmap, norm=norm, origin="lower", extent=[0, world_width, 0, world_height])
    ax.set_xticks(np.arange(0, world_width+1, 1))
    ax.set_yticks(np.arange(0, world_height+1, 1))
    ax.grid(color='black', linestyle='-', linewidth=1)

    # q-values e política ótima
    max_action = get_optimal_policy(q_values)
    for y in range(1, world_height):
        for x in range(world_width):
            direction_text = direction(max_action[x][y])
            if direction_text: 
                ax.text(x + 0.5, y + 0.5, direction_text, va='center', ha='center', fontsize=20)
                ax.text(x + 0.5, y + 0.1, "{:.1f}".format(q_values[x][y][np.argmax(q_values[x, y, :])]), 
                    va='center', ha='center', fontsize=8)

    # informações sobre a época, passos e recompensas
    if len(states_arr) > 0:
        info_ax = fig.add_axes([-0.01, 0.85, 0.1, 0.1])  # [left, bottom, width, height]
        info_ax.axis('off') 
        info_text = f'Epoca: {epoch}\nPassos: {steps}\nRewards: {rewards}'
        info_ax.text(0.5, 0.5, info_text, va='center', ha='left', fontsize=10, color='black',
                    bbox=dict(facecolor='white', alpha=0.8))

    def init():
        ball.set_data([], [])
        return ball,

    def update(frame):
        x, y = states_arr[frame]
        ball.set_data([x + 0.5], [y + 0.5])
        return ball,

    # anima a bola azul no caminho
    if len(states_arr) > 0:
        ball, = ax.plot([], [], 'bo', ms=20) 
        anim = FuncAnimation(fig, update, frames=len(states_arr), init_func=init, repeat=False, blit=True, interval=50)

    plt.savefig('plots/ex_qvalues.png')
    plt.show()
    plt.close()

def plot_rewards(rewards):
    total_epochs = len(rewards)
    split_index = int(total_epochs // (total_epochs * 0.1))

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(rewards[:split_index], '.-', color='green')
    plt.title('Primeiros 10% de Iterações')
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(rewards[split_index:], '.-', color='green')
    plt.title('Últimas Iterações')
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    plt.ylim(min(rewards[split_index:])-30, max(rewards[split_index:])+30)
    plt.grid()

    plt.tight_layout()
    plt.savefig('plots/ex_rewards.png')
    plt.show()

def maximum_reward_or_explore(state, q_value):
    # flip
    if np.random.rand() < EPSILON:
        # explora
        return ACTIONS[np.random.randint(0, len(ACTIONS))]
    else:
        # maximum reward
        values_ = q_value[state[X], state[Y], :] # Pega os valores de Q para o estado atual
        max_value = np.max(values_)
        
        # caso tenham acoes com a mesma recompensa
        best_actions = []
        for action, value in enumerate(values_):
            if value == max_value:
                best_actions.append(action)
        
        # desempate manual
        if len(best_actions) == 1:
            return best_actions[0]
        else:
            return best_actions[np.random.randint(0, len(best_actions))]

def get_optimal_policy(q_values):
    optimal = np.zeros((WORLD_WIDTH, WORLD_HEIGHT), dtype=object)
    for y in range(WORLD_HEIGHT):
        for x in range(WORLD_WIDTH):
            best = np.argmax(q_values[x, y, :])
            if best == 0:
                optimal[x][y] = 'up'
            elif best == 1:
                optimal[x][y] = 'down'
            elif best == 2:
                optimal[x][y] = 'left'
            elif best == 3:
                optimal[x][y] = 'right'
    return optimal

def q_learning(q_values, alpha):
    # inicia no estado 0, 0
    state = START
    rewards = 0.0
    states_arr = [state]

    while state != FINISH:
        # escolhe a acao baseado na politica epsilon-greedy
        action = maximum_reward_or_explore(state, q_values)

        # executa a acao e recebe a recompensa
        next_state, reward = step(state, action)
        rewards += reward

        # atualiza Q-Value
        q_values[state[X], state[Y], action] += alpha * (
            reward + GAMMA * np.max(q_values[next_state[X], next_state[Y], :]) -
            q_values[state[X], state[Y], action]
        )
        
        state = next_state
        states_arr.append(state)
    return rewards, states_arr

def main(run, epochs, plot_worlds_flag):
    print(f"Iniciando RUN {run} com {epochs} EPOCHS.")
    
    rewards = np.zeros(epochs)
    q_learning_values = np.zeros((WORLD_WIDTH, WORLD_HEIGHT, 4)) 
    if(plot_worlds_flag):
        plot_world(WORLD_HEIGHT, WORLD_WIDTH, START, FINISH, PENALITY_STATES, q_learning_values)

    for _ in range(run):
        print(f'Run: {_}')
        for i in range(epochs):
            reward, states_arr = q_learning(q_learning_values, ALPHA) 
            rewards[i] += reward

            if plot_worlds_flag and i % 10 == 0:
                print(f'%: {i/epochs*100:.2f}')
                plot_world(WORLD_HEIGHT, WORLD_WIDTH, START, FINISH, PENALITY_STATES, q_learning_values, states_arr, i, len(states_arr), reward)

    # plota recompensas médias e mostra os valores de Q aprendidos
    plot_rewards(rewards/run)
    plot_world(WORLD_HEIGHT, WORLD_WIDTH, START, FINISH, PENALITY_STATES, q_learning_values)

if __name__ == '__main__':
    X = 0
    Y = 1

    # constantes
    EPSILON = 0.1
    ALPHA = 0.5
    GAMMA = 1

    # acoes possiveis
    UP =    0       # '↑'
    DOWN =  1       # '↓'
    LEFT =  2       # '←'
    RIGHT = 3       # '→'
    ACTIONS = [UP, DOWN, LEFT, RIGHT]

    # tamango do mundo
    WORLD_HEIGHT = 4
    WORLD_WIDTH = 12

    # estado inicial
    START = [0, 0]
    FINISH = [11, 0]

    # estados de penalidade
    PENALITY_STATES = [
        [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], 
        [6, 0], [7, 0], [8, 0], [9, 0], [10, 0]
    ]

    parser = argparse.ArgumentParser(description='Configurações do treinamento.')
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--plot_worlds', action='store_true')
    args = parser.parse_args()

    # Chama a função principal com os argumentos fornecidos
    main(args.run, args.epochs, args.plot_worlds)
