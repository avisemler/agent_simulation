#%%

import math

import matplotlib.pyplot as plt
import numpy as np

from agent import Agent

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.75

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
        self.n_agents = sum(agents)
        self.n_agent_types = len(agent_parameters)
        print(self.n_agent_types)

        self.timesteps = timesteps

    def run(self):        
        #record which actions are selected by how many agents at each timestep
        action_count_over_time = np.zeros((self.timesteps, self.n_agent_types, self.n_actions))
        
        for i in range(self.timesteps):
            #count how many agents select each action

            #make each agent select an action
            for agent_type_number, agent_type in enumerate(self.agents):
                action_counts = [0] * len(self.actions)
                for agent in agent_type:
                    action = agent.select_action_softmax(1.1)
                    action_counts[action] += 1
                #record what the action counts were, so they can be plotted later
                action_count_over_time[i][agent_type_number] = action_counts
            
            #update the agents' based on the value of the actions they took
            for agent_type in self.agents:
                for agent in agent_type:
                    action = agent.previous_action      
                    value = self.actions[action].get_value(action_counts[action]/self.n_agents)
                    agent.update(value, action)
            
            #plot the action counts for each agent type and summed over the types
            dimension = int(math.sqrt(self.n_agent_types) + 1) + 1
            print(dimension)
            if i % 100 == 0 and i > 0:
                fig, axis = plt.subplots(dimension, dimension)
                plot_count = 0
                for agent_type in range(self.n_agent_types):
                    for j in range(self.n_actions):
                        #plot a line for the ith action
                        x_coord = plot_count // dimension
                        y_coord = plot_count % dimension
                        axis[x_coord, y_coord].plot(np.arange(i),
                        action_count_over_time[:i,agent_type,j],
                        label=self.actions[j].name
                        )
                        axis[x_coord, y_coord].set_title("Group " + str(agent_type + 1))
                    plot_count += 1

                #plot the sum over all agent types too
                summed_actions = np.sum(action_count_over_time, 1)
                for j in range(self.n_actions):
                    x_coord = plot_count // dimension
                    y_coord = plot_count % dimension
                    axis[x_coord, y_coord].plot(np.arange(i),
                        summed_actions[:i,j],
                        label=self.actions[j].name
                    )
                    axis[x_coord, y_coord].set_title("Full population")
                fig.legend(loc='upper right')
                plt.show()


if __name__ == "__main__":
    Simulation(
        actions=[RightGaussianCongestedAction("Car", 0, 0, 0.4),
        RightGaussianCongestedAction("Bus", 0.4, 1.14, 0.35),
        ConstantAction("Walk", 1)],
        agents=[1000, 1000, 1000],
        agent_parameters=[
            [(0, 1.3), (1, 0), (1, 0)],
            [(1, 0.1), (1.35, 0.11), (1.4, 0)],
            [(1.4, -0.2), (0.7, 0), (0.8, 0)]
        ],
        timesteps=100000
    ).run()

#%%