import abc
########################################################################################################################
# Algorithm
# abstract class for policy optimization algorithm
########################################################################################################################
class Algorithm(abc.ABC):
    def __init__(self):
        __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, agent, env, conformal, n_steps):
        """
        trains the agent for <n_steps> steps, this form is for online training, for offline -- change the signature
        """
        return
########################################################################################################################
########################################################################################################################
# Conformal
# abstract class for the conformal component
########################################################################################################################
class Conformal(abc.ABC):
    pass
########################################################################################################################
########################################################################################################################
# Agent
# abstract class for agent
########################################################################################################################
class Agent(abc.ABC):
    def __init__(self, algorithm: Algorithm, conformal: Conformal):
        __metaclass__ = abc.ABCMeta
        self.algorithm = algorithm
        self.conformal = conformal

    @abc.abstractmethod
    def step(self, state):
        """
        input: state -- the current state (feature vector)
        output: action
        """
        action = None
        return action

    @abc.abstractmethod
    def reset(self):
        """
        resets the episode
        """
        return

    @abc.abstractmethod
    def evaluate(self, env, n_steps):
        """
        evaluates the performance of the agent in the environment <env>, for <n_steps> steps
        """
        return
########################################################################################################################
########################################################################################################################
