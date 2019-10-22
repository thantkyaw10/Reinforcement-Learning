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

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(iterations):
            nextValues = util.Counter() # Empty dictionary to be populate
            for state in mdp.getStates():
                if mdp.isTerminal(state): # If terminal state the reward is its own reward
                    nextValues[(state, 'pass')] = mdp.getReward(state, 'pass', state)
                    continue
                for action in mdp.getPossibleActions(state): #Calculate utility of all possible next state
                    nextStateProb = 0
                    for tup in mdp.getTransitionStatesAndProbs(state, action):
                        if tup[0] == state:
                            nextStateProb = tup
                    reward =  mdp.getReward(state, action, nextStateProb[0]) #Get reward of next state R(s)
                    #Problem: I'm adding the reward of next state with the utility of next state, where I'm supposed 
                    # to be adding the reward of current state with the utility of the next
                    nextVal = reward + discount * nextStateProb[1] * self.values[(nextStateProb[0], action)] #Bellman equation
                    nextValues[(nextStateProb[0], action)] = nextVal # Assigns the (s, a) tuple as a key, with its q-value as the value
            self.values = nextValues # Assigns current value dict to prev




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
        return self.values[(state, action)]

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        a = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            a[action] = self.values[(state, action)]
        return a.argMax
            
        
        


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
