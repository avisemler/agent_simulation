import random
from typing import Callable

import numpy as np

class Agent:
    """An agent that uses Q-learning to learn to select
    optimal actions (in an environment with a single state)"""

    def __init__(self,
            reward_parameters,
            action_count: int,
            learning_rate: float,
            discount_rate: float
        ):
        """
        reward_parameters maps controls how values get mapped
        to rewards, enabling different agent types with different
        preferences. The structure is 

        [(sensitivity for action 1, cost for action 1),
         (sensitivity for action 2, cost for action 2),
         ...
         ]

        and reward(action, value) = 
            sensitivity_for_action * value - cost_for_action

        action_count is the number of possible actions that can
        be taken in the environment
        """
        self.reward_parameters = reward_parameters
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        #record Q-values in an array
        self.q_values = np.zeros(action_count)

        #record previous action
        self.previous_action = None

    def select_action_epsilon_greedy(self, epsilon: float) -> int:
        """Return the action with greatest Q-value"""
        if random.random() > epsilon:
            #break ties randomly
            action = np.random.choice(np.flatnonzero(self.q_values == self.q_values.max()))
        else:
            action = np.random.randint(0, self.action_count)
        self.previous_action = action
        return action

    def select_action_softmax(self, temperature: float)-> int:
        assert temperature > 0
        temperature_applied = self.q_values / temperature
        #bound values to avoid overflow
        bounded = np.clip(temperature_applied, 0, 700)
        #add an epsilon in the division to avoid dividing by zero
        softmaxed = np.exp(bounded) / np.sum(np.exp(bounded) + 0.0001)
        action = np.argmax(np.random.multinomial(1, softmaxed))
        self.previous_action = action
        return action
    
    def update(self, value: float, previous_action: int):
        """Update Q-values after previous_action was taken,
        resulting in value
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

if __name__ == "__main__":
    values = [1,2,3,7]
    a = Agent([(3, 1), (4, 3)], 2, 0.1, 0.9)

    for i in range(1000):
        action = a.select_action_softmax(2)
        a.update(values[action], action)

    print(a.q_values)
