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

        nextValues = util.Counter() # Empty dictionary to be populate

        for i in range(iterations):
            nextValues = self.values.copy()
            #nextValues = util.Counter() # Empty dictionary to be populate
            for state in mdp.getStates():
                vals = []
                if mdp.isTerminal(state): # If terminal state the reward is its own reward
                    nextValues[state] = self.mdp.getReward(state, 'exit', state) # I got 'pass' from mdp.py but its not right
                else:
                    for action in mdp.getPossibleActions(state): #Calculate utility of all next states
                        nextVal = 0
                        for tup in mdp.getTransitionStatesAndProbs(state, action): # Adds rewards of all resulting states times the prob to get to that state
                            nextVal += (tup[1] * self.mdp.getReward(state, action, tup[0])) + (tup[1] * self.discount * self.values[tup[0]])
                        vals.append(nextVal)
                    nextValues[state] = max(vals)
            self.values = nextValues.copy()
                   
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
        q=0 #weighted average value
        for tup in self.mdp.getTransitionStatesAndProbs(state, action): #tup is (state, action)
            q += (tup[1] * self.mdp.getReward(state, action, tup[0])) + (tup[1] * self.discount * self.values[tup[0]])
        return q

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
        else:
            compare = -999
            act = 0
            for action in self.mdp.getPossibleActions(state):
                nextVal = 0
                for tup in self.mdp.getTransitionStatesAndProbs(state, action):
                    nextVal += (tup[1] * self.mdp.getReward(state, action, tup[0])) + (tup[1] * self.discount * self.values[tup[0]])
                if nextVal > compare:
                    act = action
                    compare = nextVal
            return act

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
