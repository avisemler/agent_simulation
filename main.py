import math

import matplotlib.pyplot as plt
import numpy as np

from agent import Agent, FixedAgent
from actions import RightGaussianCongestedAction, ConstantAction
from simulation import Simulation

#some constants to control the simulation

AGENT_NUMBERS = [1000, 1000, 1000] #number of agents of each type
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.75
COLOURS = ["blue", "red", "orange", "green", "purple"]
PLOT_FREQUENCY = 500 #how often to display a plot of actions so far

#the set of actions that agents can take
actions = [RightGaussianCongestedAction("Car", 0, 1, 0.4),
    RightGaussianCongestedAction("Bus", 0.4, 1.14, 0.35),
    ConstantAction("Walk", 1),
]

#create a list to store the agents
agents = []
for i, agent_type_number in enumerate(AGENT_NUMBERS):
    agents_of_current_type = []
    for j in range(agent_type_number):
        #create an agent
        a = Agent(
                action_count=len(actions),
                learning_rate=LEARNING_RATE,
                discount_rate=DISCOUNT_RATE
            )
        agents_of_current_type.append(a)
    agents.append(agents_of_current_type)

simulation = Simulation(
    actions=actions,
    agents=agents,
)

#parameters (sensitivity and cost for each action) for each agent type
agent_parameters = [
    [(1.3, 0), (1, 0), (1, 0)],
    [(1, 0.1), (1.35, 0.11), (1.4, 0)],
    [(1.4, -0.2), (0.7, 0), (0.8, 0)],
]

for i in range(100000):
    simulation.timestep(agent_parameters)

    if i % PLOT_FREQUENCY == 0 and i>0:
        simulation.plot_actions_over_time()