import math

import matplotlib.pyplot as plt
import numpy as np

from agent import Agent, FixedAgent
from actions import *

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.75
COLOURS = ["blue", "red", "orange", "green", "purple"]
RUN_COUNT = 500 #how often to display a plot of actions so far

class Simulation:
    """Runs a similation of agents that can take actions
    to use congested resources"""

    def __init__(self,
        actions: list[Action],
        agents: list[Agent],
    ):
        #contains a sublist for each agent type
        self.agents = agents

        self.actions = actions
        self.n_actions = len(self.actions)
        self.n_agents = sum([len(t) for t in agents])
        self.n_agent_types = len(agents)

        self.timesteps_so_far = 0

        #An array to record what actions where chosen at each timestep,
        #for use in plotting. The array grows each timestep
        self.action_count_over_time = np.zeros((0, self.n_agent_types, self.n_actions))

    def timestep(self):
        """Perform one timestep of the simulation."""

        #create a new entry for self.action_count_over_time
        new_action_count_entry = np.zeros((self.n_agent_types, self.n_actions))

        #make each agent select an action
        for agent_type_number, agent_type in enumerate(self.agents):
            action_counts = [0] * len(self.actions)
            for agent in agent_type:
                action = agent.select_action_epsilon_greedy(0.05)
                action_counts[action] += 1
            #record what the action counts were, so they can be plotted later
            new_action_count_entry[agent_type_number] = action_counts

        #add the new entry to self.action_counts_over_time, by replacing it with a new array
        temp = np.zeros((self.timesteps_so_far+1, self.n_agent_types, self.n_actions))
        temp[:-1,...] = self.action_count_over_time
        temp[-1,...] = new_action_count_entry
        self.action_count_over_time = temp
        
        #update the agents' based on the value of the actions they took
        for agent_type_number, agent_type in enumerate(self.agents):
            for agent in agent_type:
                action = agent.previous_action
                #sum over all agent types to calculate how many chose this action
                number_choosing_this_action = np.sum(self.action_count_over_time[-1], 0)[action]  
                value = self.actions[action].get_value(number_choosing_this_action/self.n_agents)
                agent.update(value, action)
    
        self.timesteps_so_far += 1

    def plot_actions_over_time(self):
        """Display a matplotlib plot of the action counts over time."""

        #plot the action counts for each agent type and summed over the types
        dimension = math.ceil(math.sqrt(self.n_agent_types + 1))
        fig, axis = plt.subplots(dimension, dimension)
        plot_count = 0
        for agent_type in range(self.n_agent_types):
            for j in range(self.n_actions):
                #plot a line for the ith action
                x_coord = plot_count // dimension
                y_coord = plot_count % dimension
                axis[x_coord, y_coord].plot(np.arange(self.timesteps_so_far),
                    self.action_count_over_time[:,agent_type,j],
                    label=self.actions[j].name,
                    color=COLOURS[j]
                )
                axis[x_coord, y_coord].set_title("Group " + str(agent_type + 1))
            plot_count += 1

        #plot the sum over all agent types too
        summed_actions = np.sum(self.action_count_over_time, 1)
        for j in range(self.n_actions):
            x_coord = plot_count // dimension
            y_coord = plot_count % dimension
            axis[x_coord, y_coord].plot(np.arange(self.timesteps_so_far),
                summed_actions[:,j],
                label=self.actions[j].name,
                color=COLOURS[j]
            )
            axis[x_coord, y_coord].set_title("Full population")
        fig.legend(loc='upper right')
        plt.show()

    def plot_action_profiles(self):
        """Plot how proportions are getting mapped to values for actions"""
        steps = 1000
        for number, action in enumerate(self.actions):
            lines = np.zeros((steps))
            for step in range(steps):
                value = action.get_value(step/steps)
                lines[step] = self.agents[2][0].calculate_reward(number, value)
            time = np.arange(0, 1, 1/steps)
            plt.plot(time, lines, color=COLOURS[number])
        plt.show()

class FixedAgentSimulation(Simulation):
    """A simulation where some of the population are
    fixed agents that always pick the same action."""

    def __init__(self, actions, agents, n_fixed_agents,
     fixed_agent_action):
        super().__init__(actions, agents)
        fixed_agents = []
        for i in range(n_fixed_agents):
            fixed_agents.append(FixedAgent(fixed_agent_action))

        self.agents.append(fixed_agents)
        self.n_agent_types += 1
