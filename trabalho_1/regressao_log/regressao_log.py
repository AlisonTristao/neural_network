print("Alison\nOtacilio\nJoão Vitor")

import numpy as np
from matplotlib.pyplot import subplot, plot, show, clf, vlines, grid, ylabel, xlabel
import matplotlib.pyplot as plt

# conjunto de dados {(x,y)}
mean0, std0 = -0.4, 0.5
mean1, std1 = 0.9, 0.3
m = 200

x0s = np.random.randn(m//2) * std1 + mean1
x1s = np.random.randn(m//2) * std0 + mean0
xs = np.hstack((x1s, x0s))

ys = np.hstack(( np.ones(m//2), np.zeros(m//2) ))

#plot(xs[m//2:], ys[m//2:], '.')
#plot(xs[:m//2], ys[:m//2], '.')
#show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# hipotese
# sigmoid(theta[0] + theta[1] * x)

def h(x, theta):
    return sigmoid(theta[0] + theta[1] * x)

# funcao de custo para um exemplo; entropia cruzada
def cost(h, y):
    return -(y * np.log(h) + (1 - y) * np.log(1 - h))

# funcao de custo total
def J(theta, xs, ys):
    m = len(xs)
    sum = 0.0

    for k in range(m):
        sum += ys[k] * np.log(h(xs[k], theta)) + (1 - ys[k]) * np.log(1 - h(xs[k], theta))

    return -sum/m

# derivada parcial com respeito a theta[i]
def gradient(i, theta, xs, ys):
    m = len(xs) 
    grad = 0.0  
    
    for k in range(m):
        h_ = h(xs[k], theta) 
        error = h_ - ys[k]         
        grad += error * (xs[k] if i == 1 else 1)
    
    return grad / m

def plot_fronteira(theta):
    # plota a fronteira de decisao
    clf()
    #subplot(311)
    plot(xs[m//2:], ys[m//2:], '.r')
    plot(xs[:m//2], ys[:m//2], '.g')
    x = np.linspace(-2, 2, 100)
    y = sigmoid(theta[0] + theta[1] * x)
    plot(x, y, 'orange')
    vlines(-(theta[0]/theta[1]), 0, 1, color='black')
    grid()
    show()

# plota em subplots: -- os dados, com a fronteira de decisao
# e os dados classificados
def plot_modelo(theta, xs, ys):
    pass
    
def accuracy(ys, predictions):
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0

    num = sum(ys == predictions)
    return num/len(ys)

def plot_data(data0, data1, data2):
    clf()
    x = [i for i in range(len(data0))]

    subplot(311)
    plot(x, data0, '-r')
    ylabel('Custo')
    grid()

    subplot(312)
    plot(x, data1, '-g')
    ylabel('Acuracia')
    grid()

    subplot(313)
    plot(x, data2, '-y')
    ylabel('Fronteira')
    grid()
    show()

def plot_J_theta(theta_0, theta_1, J):
    clf()

    subplot(211)
    plot(theta_0, J, '-r')
    ylabel('J')
    xlabel('Theta 0')
    grid()

    subplot(212)
    plot(theta_1, J, '-g')
    ylabel('J')
    xlabel('Theta 1')
    grid()

    show()

alpha = 0.01
epochs = 2000
theta = [0, -1]

# arrays for plotting
custo = []
acuracia = []
fronteira = []
theta_0 = []
theta_1 = []

for k in range(epochs): # 10000
    # changes values at the same time
    t_0 = theta[0] - alpha * gradient(0, theta, xs, ys)
    t_1 = theta[1] - alpha * gradient(1, theta, xs, ys)

    theta[0] = t_0
    theta[1] = t_1

    custo.append(J(theta, xs, ys))
    acuracia.append(accuracy(ys, h(xs, theta)))
    fronteira.append(-theta[0]/theta[1])

    theta_0.append(theta[0])
    theta_1.append(theta[1])

    #if k % 100 == 0:
        #print(f'J(theta) = {J(theta, xs, ys)}')
        #plot_fronteira(theta)

# c
plot_data(custo, acuracia, fronteira)

# e 
plot_J_theta(theta_0, theta_1, custo)

# g (opcional)
plot_fronteira(theta)