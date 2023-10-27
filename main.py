import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from agent import Agent, FixedAgent
from actions import RightGaussianCongestedAction, ConstantAction
from simulation import Simulation
from get_params import *

#settings to control the simulation
args = get_args(*sys.argv[1:])
AGENT_NUMBERS = [1000, 1000, 1000] #number of agents of each type
#parameters (sensitivity and cost for each action) for each agent type
INITIAL_AGENT_PARAMETERS =  [
    [[1.3, 0], [1, 0], [1, 0]],
    [[1, 0.1], [1.35, 0.11], [1.4, 0]],
    [[1.4, -0.2], [0.7, 0], [0.8, 0]],
]

#entry [i][j] controls the strength of interaction from agents of group
#i to agents of group j - WAIFW (who acquires influence from whom) matrix
INFLUENCE_MATRIX = (
    (1, 0.3, 0.3),
    (0.3, 1, 0.3),
    (0.3, 0.3, 1),
)

LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.75
COLOURS = ["blue", "red", "orange", "green", "purple"]
PLOT_FREQUENCY = 1_000  #how often to display a plot of actions so far
TOTAL_TIMESTEPS = args.total_timesteps

#the set of actions that agents can take
actions = [RightGaussianCongestedAction("Car", 0, 1, 0.4),
    RightGaussianCongestedAction("Bus", 0.4, 1.14, 0.35),
    ConstantAction("Walk", 1),
]

#create a list to store the agents
agents = []
for agent_type_number, amount in enumerate(AGENT_NUMBERS):
    agents_of_current_type = []
    for j in range(amount):
        #create an agent
        a = Agent(
                action_count=len(actions),
                learning_rate=LEARNING_RATE,
                discount_rate=DISCOUNT_RATE,
                parameters = [[a[0], a[1]] for a in INITIAL_AGENT_PARAMETERS[agent_type_number]],
                group_number=agent_type_number
            )
        agents_of_current_type.append(a)
    agents.append(agents_of_current_type)

simulation = Simulation(
    actions=actions,
    agents=agents,
    topology=None,
)

simulation.plot_action_profiles()

#create a graph to represent social connections of agents
if args.graph_gen == "ws":
    agent_graph = nx.watts_strogatz_graph(sum(AGENT_NUMBERS), int(args.graph_param1), args.graph_param2)
elif args.graph_gen == "er":
    agent_graph = nx.erdos_renyi_graph(sum(AGENT_NUMBERS), args.graph_param1)
elif args.graph_gen == "ba":
    agent_graph = nx.barabasi_albert_graph(sum(AGENT_NUMBERS), args.graph_param1)

#subax1 = plt.subplot(121)
#nx.draw(agent_graph, with_labels=False, font_weight='bold')
#plt.show()

#annotate nodes in the agent graph with agent objects
current_index = 0
for agent_type_number, amount in enumerate(AGENT_NUMBERS):
    for i in range(amount):
        agent_graph.nodes[current_index]["agent_object"] = agents[agent_type_number][i]
        current_index += 1

for i in range(TOTAL_TIMESTEPS):
    simulation.timestep()

    for agent_node in range(sum(AGENT_NUMBERS)):
        current_agent = agent_graph.nodes[agent_node]["agent_object"]

        if i == 6000 and args.intervention:
            #apply intervention
            current_agent.reward_parameters[0][1] += 1.2
            INITIAL_AGENT_PARAMETERS[current_agent.group_number][0][1] += 1.2
            if current_agent.group_number in [1,2]:
                current_agent.reward_parameters[1][0] /= 3
                INITIAL_AGENT_PARAMETERS[current_agent.group_number][1][0] /= 3

        if args.use_agent_graph:
            #propogate influence through the social graph

            #retrieve the agent object associated with this node
            current_agent = agent_graph.nodes[agent_node]["agent_object"]
            neighbours = agent_graph[agent_node]

            #calculate the amount of influence this agent has towards each action
            influence = np.zeros(len(actions))
            for key in neighbours:
                #retrieve the agent object of the neighbour
                neighbour = agent_graph.nodes[key]["agent_object"]
                influence[neighbour.previous_action] += INFLUENCE_MATRIX[neighbour.group_number][current_agent.group_number]
            #normalise by dividing by the number of neighbours - having more
            #neighbours shouldn't make you more susceptible to influence
            if len(neighbours) > 0:
                influence /= len(neighbours)

            #change sensitivities of agent to account for influences
            for i in range(len(actions)):
                original = INITIAL_AGENT_PARAMETERS[current_agent.group_number][i][0]
                current_agent.reward_parameters[i][0] = original + influence[i]

name = args.run_name
if args.use_agent_graph:
    name += "_" + str(args.graph_gen) + "_p1_" + str(args.graph_param1) + "_p2_" + str(args.graph_param2)
simulation.save(name)