import math
import random
from typing import Optional
import datetime

import os

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from agent import Agent, FixedAgent
from actions import *

from constants import *

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.75
RUN_COUNT = 500 #how often to display a plot of actions so far

class Simulation:
    """Runs a similation of agents that can take actions
    to use congested resources.
    
    Takes a list of actions, agent types (themselves lists of agents), and optionally a topology for the agents.
    If there is a topology, then the congestion is calculated using only neighbours
    of that node."""

    def __init__(self,
        actions,
        agents,
        topology = None
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

        self.topology = topology
        if self.topology:
            #ensure that there is the correct number of vertices
            assert self.topology.number_of_nodes() == sum([len(t) for t in self.agents])

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
        agent_index = 0 #calculate the index of this agent in the topology graph
        for agent_type_number, agent_type in enumerate(self.agents):
            for agent in agent_type:
                action = agent.previous_action
                if self.topology is None:
                    #sum over all agent types to calculate how many chose this action
                    #this is a straightforward calculation of congestion, without a topology
                    number_choosing_this_action = np.sum(self.action_count_over_time[-1], 0)[action]  
                    value = self.actions[action].get_value(number_choosing_this_action/self.n_agents)
                elif self.topology:
                    #use the topology: calculate congestion using only agents in the neighbourhood
                    neighbours_choosing_this_action = 0
                    for neighbour_key in self.topology[agent_index]:
                        if self.topology.nodes[neighbour_key]["agent_object"].previous_action == action:
                            neighbours_choosing_this_action += 1
                        congestion = neighbours_choosing_this_action / len(self.topology[agent_index])
                        value = self.actions[action].get_value(congestion)
                agent.update(value, action)
                agent_index += 1
    
        self.timesteps_so_far += 1

    def plot_action_profiles(self):
        """Plot how proportions are getting mapped to values for actions"""
        steps = 1000
        for number, action in enumerate(self.actions):
            lines = np.zeros((steps))
            for step in range(steps):
                value = action.get_value(step/steps)
                lines[step] = self.agents[2][0].calculate_reward(number, value)
            time = np.arange(0, 1, 1/steps)
            plt.plot(time, lines, color=COLOURS[number], label=action.name)
        plt.xlabel("Congestion level")
        plt.ylabel("Payoff")
        plt.legend()
        plt.savefig("actions.png")

    def save(self, directory):
        if not os.path.exists(os.path.join("jan_runs", directory)):
            os.mkdir(os.path.join("jan_runs", directory))
        path = os.path.join("jan_runs", directory, "data" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + "_" + str(random.randint(1, 1000)) + ".npy")
        with open(path, "wb") as f:
            np.save(f, self.action_count_over_time)

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
