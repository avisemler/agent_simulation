import math
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from constants import *

def plot_actions_over_time(directory, title, action_names):
    """Display a matplotlib plot of the action counts over time.
    This is loaded from an array stored in a .npy binary file,"""

    #average all the data found in the directory
    filepath = os.path.join("runs", directory)
    paths = glob.glob(os.path.join(filepath, "*.npy"))
    summed = None
    arrays = []
    for path in paths:
        print(path)
        with open(path, "rb") as file:
            array = np.load(file)
            arrays.append(array)
            if summed is None:
                summed = np.zeros_like(array)
                summed += array
            else:
                summed += array

    arrays = np.stack(arrays, axis=0)
    print("arrays shape:", arrays.shape)
    group_summed = np.sum(arrays, axis=-2) #sum over agent groups
    print("group summed shape:", group_summed.shape)
    print(group_summed[:,:,0])
    print("Standard deviation:", np.std(group_summed[:,:,0], axis=0).mean(axis=0))
    print(summed)
    action_count_over_time = summed / len(paths)

    #the shape of the array contains information about parameters used
    n_agent_types = action_count_over_time.shape[1]
    n_actions = action_count_over_time.shape[2]
    timesteps = action_count_over_time.shape[0]

    #plot the action counts for each agent type and summed over the types
    dimension = math.ceil(math.sqrt(n_agent_types + 1))
    fig, axis = plt.subplots(dimension, dimension)
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(right=0.83, left=0.12, wspace=0.45, hspace=0.45)
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
            axis[x_coord, y_coord].set_ylim(0, 2000)
            axis[x_coord, y_coord].set_xlabel("Time")
            axis[x_coord, y_coord].set_ylabel("Number of agents")
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
        axis[x_coord, y_coord].set_ylim(0, 2000)
        axis[x_coord, y_coord].set_xlabel("Time")
        axis[x_coord, y_coord].set_ylabel("Number of agents")
    
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.savefig(os.path.join("runs", directory, "aa_image.png"), dpi=300)

for name in glob.glob("./runs/*/"):
    print(name.split("/")[-2])
    plot_actions_over_time(name.split("/")[-2], "", ["car", "bus", "wallk"])
