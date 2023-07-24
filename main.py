import math

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from agent import Agent, FixedAgent
from actions import RightGaussianCongestedAction, ConstantAction
from simulation import Simulation

#some constants to control the simulation

AGENT_NUMBERS = [1000, 1000, 1000] #number of agents of each type
#parameters (sensitivity and cost for each action) for each agent type
INITIAL_AGENT_PARAMETERS =  [
    [[1, 0], [1, 0], [1, 0]],
    [[1, 0.1], [1.35, 0.11], [1.4, 0]],
    [[1.4, -0.2], [0.7, 0], [0.8, 0]],
]

LEARNING_RATE = 0.03
DISCOUNT_RATE = 0.75
COLOURS = ["blue", "red", "orange", "green", "purple"]
PLOT_FREQUENCY = 2000  #how often to display a plot of actions so far

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
                parameters = INITIAL_AGENT_PARAMETERS[agent_type_number],
            )
        agents_of_current_type.append(a)
    agents.append(agents_of_current_type)

simulation = Simulation(
    actions=actions,
    agents=agents,
)

#create a graph to represent social connections of agents
agent_graph = nx.erdos_renyi_graph(sum(AGENT_NUMBERS), 0.0018)
#subax1 = plt.subplot(121)
#nx.draw(agent_graph, with_labels=False, font_weight='bold')
#plt.show()

for i in range(100000):
    simulation.timestep()

    if i % PLOT_FREQUENCY == 0 and i>0:
        simulation.plot_actions_over_time()

    #lower each agent's sensitivity to an action's congestion
    #if a neighbour took that action in the previous timestep
    for agent_node in range(sum(AGENT_NUMBERS)):
        neighbours = agent_graph[agent_node]