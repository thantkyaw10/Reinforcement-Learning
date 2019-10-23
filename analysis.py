# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    #answerNoise = 0.2
    answerNoise = 0 #make the noise zero
    return answerDiscount, answerNoise

def question3a():
    #Prefer the close exit (+1), risking the cliff (-10)
    #risk cliff = Living reward negative (but not worse than going to cliff)
    #little to no noise
    answerDiscount = 0.9
    answerNoise = .05
    answerLivingReward = -5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    #Prefer the close exit (+1), but avoiding the cliff (-10)
    #living reward = small positive or zero = don't want to die, but also want to end (so betwee)
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    #Prefer the distant exit (+10), risking the cliff (-10)
    answerDiscount = 0.9
    answerNoise = 0.05
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    #Prefer the distant exit (+10), avoiding the cliff (-10)
    
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    #Avoid both exits and the cliff (so an episode should never terminate)
    # living is better than any of the terminals
    # OOORRRR Don't care about anything, so all zero
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 12
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question6():
    answerEpsilon = None
    answerLearningRate = None
    #return answerEpsilon, answerLearningRate
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
