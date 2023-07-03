import math

import matplotlib.pyplot as plt
import numpy as np

from agent import Agent

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.7

def gaussian_density(x, mu, sigma):
    result = math.exp(-0.5 * ( (x-mu)/sigma )** 2)
    result /= sigma * (2 * math.pi) ** 0.5
    return result * 10000

class Action:
    def __init__(self, name):
        self.name = name

    def get_value(self) -> float:
        """Return the value of the action under current conditions"""
        pass

class ConstantAction(Action):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value

    def get_value(self, *args):
        return self.value

class CongestedConstantAction(Action):
    def __init__(self, name, capacity, below_capacity_value, above_capacity_value):
        super().__init__(name)
        self.capacity = capacity
        self.below_capacity_value = below_capacity_value
        self.above_capacity_value = above_capacity_value

    def get_value(self, *args):
        if args[0] <= self.capacity:
            return self.below_capacity_value
        else:
            return self.above_capacity_value

class GaussianCongestedAction(Action):
    def __init__(self, name, capacity, left_tail_sd, right_tail_sd):
        super().__init__(name)
        self.capacity = capacity
        self.left_tail_sd = left_tail_sd
        self.right_tail_sd = right_tail_sd

    def get_value(self, *args):
        if args[0] <= self.capacity:
            #in-capacity case
            return gaussian_density(args[0], self.capacity, self.left_tail_sd)
        else:
            return gaussian_density(args[0], self.capacity, self.right_tail_sd)

class RightGaussianCongestedAction(Action):
    """Action where the right-tail uses gaussian density but the left is constant"""
    def __init__(self, name, capacity, left_tail_constant, right_tail_sd):
        super().__init__(name)
        self.capacity = capacity
        self.left_tail_constant = left_tail_constant
        self.right_tail_sd = right_tail_sd

    def get_value(self, *args):
        if args[0] <= self.capacity:
            #in-capacity case
            return self.left_tail_constant
        else:
            return gaussian_density(args[0], self.capacity, self.right_tail_sd)

class Simulation:
    """Runs a similation of agents that can take actions
    to use congested resources"""

    def __init__(self,
        actions: list[Action],
        agents: list[int],
        agent_parameters: tuple[tuple[int]],
        timesteps: int
    ):
        #create a list store agents
        #contains a sublist for each agent type
        self.agents = []
        for i, agent_type_number in enumerate(agents):
            agents_of_current_type = []
            for j in range(agent_type_number):
                #create an agent with parameters for agent group i
                a = Agent(reward_parameters = agent_parameters[i],
                        action_count=len(actions),
                        learning_rate=LEARNING_RATE,
                        discount_rate=DISCOUNT_RATE
                    )
                agents_of_current_type.append(a)
            self.agents.append(agents_of_current_type)

        self.actions = actions
        self.n_actions = len(self.actions)

        self.timesteps = timesteps

    def run(self):        
        #record which actions are selected by how many agents at each timestep
        action_count_over_time = np.zeros((self.timesteps, self.n_actions,))

        for i in range(self.timesteps):
            print(i)
            #count how many agents select each action
            action_counts = [0] * len(self.actions)

            #make each agent select an action
            for agent_type in self.agents:
                for agent in agent_type:
                    action = agent.select_action_softmax(10)
                    action_counts[action] += 1

            #record what the action counts were, so they can be plotted later
            action_count_over_time[i] = action_counts
            
            #update the agents' based on the value of the actions they took
            for agent_type in self.agents:
                for agent in agent_type:
                    action = agent.previous_action
                    
                    value = self.actions[action].get_value(action_counts[action])
                    agent.update(value, action)

        plt.plot(np.arange(self.timesteps), action_count_over_time)
        plt.show()

if __name__ == "__main__":
    Simulation(
        actions=[RightGaussianCongestedAction("1", 200, 4, 120),
        RightGaussianCongestedAction("2", 500, 1, 10),
        RightGaussianCongestedAction("3", 400, 3, 100)],
        agents=[400, 320],
        agent_parameters=[
            [(2, 0), (3, 0), (0.4, -4)],
            [(5, -2), (2, 0), (6, 4)],
        ],
        timesteps=200
    ).run()

