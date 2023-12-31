import random
from typing import Callable

import numpy as np

class Agent:
    """An agent that uses Q-learning to learn to select
    optimal actions (in an environment with a single state).
    """

    def __init__(self,
            action_count: int,
            learning_rate: float,
            discount_rate: float,
            parameters,
            group_number: int = 0,
        ):
        """
        action_count is the number of possible actions that can
        be taken in the environment

        reward_parameters maps controls how values get mapped
        to rewards, enabling different agent types with different
        preferences that change over time. The structure is 

        [(sensitivity for action 1, cost for action 1),
            (sensitivity for action 2, cost for action 2),
            ...
            ]

        and reward(action, value) = 
            sensitivity_for_action * value - cost_for_action

        group_number records which agent group this agent belongs to
        """
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.reward_parameters = parameters
        self.group_number = group_number

        #record Q-values in an array
        self.q_values = np.array([0.0,0.0,0.0])

        #record previous action
        self.previous_action = None

    def select_action_epsilon_greedy(self, epsilon: float) -> int:
        """Return the action with greatest Q-value, with probability 1-epsilon.
        Else, return a random action, to explore."""
        if random.random() > epsilon:
            #break ties randomly
            action = np.random.choice(np.flatnonzero(self.q_values == self.q_values.max()))
        else:
            action = np.random.randint(0, self.action_count)
        self.previous_action = action
        return action

    def select_action_softmax(self, temperature: float)-> int:
        """Use a categorical distribution based on the softmax of the Q-values
        to select an action."""
        assert temperature > 0
        temperature_applied = self.q_values / temperature
        #for numerical purposes, to avoid 0 division etc.
        temperature_applied -= np.max(temperature_applied)
        softmaxed = np.exp(temperature_applied) / np.sum(np.exp(temperature_applied))
        #normalise to ensure sums to exactly 1
        softmaxed /=  softmaxed.sum()
        action = np.random.choice(np.arange(self.action_count), p=softmaxed)
        self.previous_action = action
        return action
    
    def update(self, value: float, previous_action: int):
        """Update Q-values after previous_action was taken,
        resulting in value.
        """

        #the action must be in the correct range
        assert previous_action < self.action_count
        
        reward = self.calculate_reward(previous_action, value)
        self.q_values[previous_action] = (
            (1-self.learning_rate) * self.q_values[previous_action]
            + self.learning_rate
            * (reward + self.discount_rate * np.max(self.q_values))
        )

    def calculate_reward(self, action, value):
        """Calculate the reward for an action given its value"""
        sensitivity = self.reward_parameters[action][0]
        cost = self.reward_parameters[action][1]
        return sensitivity * value - cost

class FixedAgent:
    """An agent that always picks the same strategy."""
    def __init__(self, action_to_pick):
        self.action = action_to_pick
        self.previous_action = action_to_pick

    def select_action_epsilon_greedy(self, *args):
        return self.action

    def select_action_softmax(self, *args):
        return self.action

    def calculate_reward(self, *args):
        pass

    def update(self, *args):
        pass
