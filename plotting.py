import math

import numpy as np
import matplotlib.pyplot as plt

from constants import *

def plot_actions_over_time(filepath, title, action_names):
    """Display a matplotlib plot of the action counts over time.
    This is loaded from an array stored in a .npy binary file,"""

    with open(filepath, "rb") as file:
        action_count_over_time = np.load(file)


    #the shape of the array contains information about parameters used
    n_agent_types = action_count_over_time.shape[1]
    n_actions = action_count_over_time.shape[2]
    timesteps = action_count_over_time.shape[0]
    print(n_actions, n_agent_types)

    #plot the action counts for each agent type and summed over the types
    dimension = math.ceil(math.sqrt(n_agent_types + 1))
    fig, axis = plt.subplots(dimension, dimension)
    plot_count = 0
    for agent_type in range(n_agent_types):
        for j in range(n_actions):
            #plot a line for the ith action
            x_coord = plot_count // dimension
            y_coord = plot_count % dimension
            axis[x_coord, y_coord].plot(np.arange(timesteps),
                action_count_over_time[:,agent_type,j],
                label=action_names[j],
                color=COLOURS[j]
            )
            axis[x_coord, y_coord].set_title("Group " + str(agent_type + 1) + " " + title)
        plot_count += 1

    #plot the sum over all agent types too
    summed_actions = np.sum(action_count_over_time, 1)
    for j in range(n_actions):
        x_coord = plot_count // dimension
        y_coord = plot_count % dimension
        axis[x_coord, y_coord].plot(np.arange(timesteps),
            summed_actions[:,j],
            label=action_names[j],
            color=COLOURS[j]
        )
        axis[x_coord, y_coord].set_title("Full population " + title)
    fig.legend(loc='upper right')
    plt.show()

plot_actions_over_time("runs/1/data.npy", "B", ["car", "bus", "wallk"])
