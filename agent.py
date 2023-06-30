#%%
import random
from typing import Callable

import numpy as np

class Agent:
    """An agent that uses Q-learning to learn to select
    optimal actions (in an environment with a single state)"""

    def __init__(self,
            reward_function: Callable[float, float],
            action_count: int,
            learning_rate: float,
            discount_rate: float
        ):
        """
        reward_function maps values to rewards, enabling different
        agent types with different preferences

        action_count is the number of possible actions that can
        be taken in the environment
        """
        self.reward_function = reward_function
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        #record Q-values in an array
        self.q_values = np.zeros(action_count)

        #record previous action
        self.previous_action = None

    def select_action_epsilon_greedy(self, epsilon: float) -> int:
        """Return the action with greatest Q-value"""
        if random.random() < epsilon:
            action = np.argmax(self.q_values)
        else:
            #return action!!
            action = np.random.choice(self.q_values)
        self.previous_action = action
        return action

    def select_action_softmax(self, temperature: float)-> int:
        assert temperature > 0
        temperature_applied = self.q_values / temperature
        #add an epsilon in the division to avoid dividing by zero
        softmaxed = np.exp(temperature_applied) / np.sum(np.exp(temperature_applied) + 0.0001)
        print(softmaxed)
        action = np.argmax(np.random.multinomial(1, softmaxed))
        self.previous_action = action
        return action
    
    def update(self, value: float, previous_action: int):
        """Update Q-values after previous_action was taken,
        resulting in value
        
        use self.
        """

        #the action must be in the correct range
        assert previous_action < self.action_count
        
        reward = self.reward_function(value)
        self.q_values[previous_action] = (
            (1-self.learning_rate) * self.q_values[previous_action]
            + self.learning_rate
            * (reward + self.discount_rate * np.max(self.q_values))
        )

if __name__ == "__main__":
    values = [1,2,3,7]
    a = Agent(lambda x: x, 4, 0.1, 0.9)

    for i in range(1000):
        action = a.select_action_softmax(2)
        a.update(values[action], action)

    print(a.q_values)