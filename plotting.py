import math
import os

import numpy as np
import matplotlib.pyplot as plt

from constants import *

def plot_actions_over_time(directory, title, action_names):
    """Display a matplotlib plot of the action counts over time.
    This is loaded from an array stored in a .npy binary file,"""
    filepath = os.path.join("runs", directory, "data.npy")
    with open(filepath, "rb") as file:
        action_count_over_time = np.load(file)


    #the shape of the array contains information about parameters used
    n_agent_types = action_count_over_time.shape[1]
    n_actions = action_count_over_time.shape[2]
    timesteps = action_count_over_time.shape[0]

    #plot the action counts for each agent type and summed over the types
    dimension = math.ceil(math.sqrt(n_agent_types + 1))
    fig, axis = plt.subplots(dimension, dimension)
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(right=0.83)
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

    
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.savefig(os.path.join("runs", directory, "image.png"))

plot_actions_over_time("1", "B", ["car", "bus", "wallk"])
