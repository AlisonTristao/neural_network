# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates() # os estados do MDP são as possiveis posições no mundo
        values = util.Counter()
        for state in states:
            values[(state, 0)] = 0 # inicializa os valores de todos os estados da primeira iteração com 0
        for k in range(1, self.iterations + 1):
            for state in states:
                acoes = self.mdp.getPossibleActions(state) #north','west','south','east
                valorMaximo = float('-infinity')
                for acao in acoes:
                    value = 0 #variável pra armazenar o somatório dos possíveis estados futuros
                    proximoEstadoInfos = self.mdp.getTransitionStatesAndProbs(state, acao) #retorna tuple (estado, probabilidade)
                    for proximoEstadoInfo in proximoEstadoInfos:
                        proximoEstado = proximoEstadoInfo[0]
                        probabilidade = proximoEstadoInfo[1]
                        reward = self.mdp.getReward(state, acao, proximoEstado)
                        v = values[(proximoEstado, k - 1)]
                        value += probabilidade * (reward + self.discount * v)
                    valorMaximo = max(valorMaximo, value) #Entre todas as ações, pega a de maior valor
                if valorMaximo  > float('-infinity'):
                    values[(state, k)] = valorMaximo #armazena o valor esperado do estado
                else:
                    values[(state, k)] = 0 # o menor valor esperado possível é 0
        for state in states:
            self.values[state] = values[(state, self.iterations)] #armazena o valor esperado do estado na última iteração        







    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        proximoEstadoInfos = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0
        for proximoEstadoInfo in proximoEstadoInfos:
            proximoEstado = proximoEstadoInfo[0]
            probabilidade = proximoEstadoInfo[1]
            reward = self.mdp.getReward(state, action, proximoEstado)
            v = self.values[proximoEstado]
            value += probabilidade * (reward + self.discount * v)
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        acoes = self.mdp.getPossibleActions(state)
        bestValue = float('-infinity')
        for acao in acoes:
            qValue = self.computeQValueFromValues(state, acao)
            if qValue > bestValue:
                bestValue = qValue
                bestAction = acao
        return bestAction
            

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
