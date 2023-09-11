# Agent-based Simulation of Congestion Games

## Introduction

Congestion games are games where there are agents who must select resources to use. These resources are congested: the more agents that use a certain resource, the lower the payoff to any one agent will be - think of a road that is prone to traffic jams as more drivers begin to use it.

In many real life examples of congestion, we seek interventions to effect the equilibrium that the agents tend to. For example, drivers could be given monetary incentive to cycle, or charged extra for using the busiest roads.

## The model

There are $n$ Q-learning agents. In each iteration, each agent uses its estimate of the value of each action to select an action (i.e. a resource in many examples). Then, each agent receives a reward based on the actions of all other agents and updates its estimate of the reward for the action using the update rule

$$Q(a)\leftarrow (1-\alpha)Q(a)+\alpha(r(a)+\gamma \max_{b\in A}Q(b))$$

where $Q$ is the function that esimates the value of each action, $\alpha$ is the learning rate, $\gamma$ is the discount rate, $r(a)$ denotes the reward received (taking into account the actions of all other agents), and $A$ is the set of all possible actions.

## The code

`simulation.py` contains the core of the model.

`agent.py` implements a Q-learning agent and softmax/epsilon-greedy action selection.

`plotting.py` contains functions that use matplotlib to visualise the results of a simulation.

`main.py` is an example simulation and intervention.
