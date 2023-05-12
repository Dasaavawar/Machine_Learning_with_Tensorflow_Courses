# Machine Learning with TensorFlow - Code 6.2. - Reinforcement Learning


# Frozen Lake Enviornment

# The enviornment we loaded above FrozenLake-v0 is one of the simplest enviornments in Open AI Gym.
# The goal of the agent is to navigate a frozen lake and find the Goal without falling through the ice (render the enviornment above to see an example).
# There are:
# - 16 states (one for each square)
# - 4 possible actions (LEFT, RIGHT, DOWN, UP)
# - 4 different types of blocks (F: frozen, H: hole, S: start, G: goal)

# Building the Q-Table

# Imports and Setup
import gym
import numpy as np
import time

# The first thing we need to do is build an empty Q-Table that we can use to store and update our values.
env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))  # create a matrix with all 0 values 
Q

# Constants

# As we discussed we need to define some constants that will be used to update our Q-Table and tell our agent when to stop training.
EPISODES = 2000 # how many times to run the enviornment from the beginning
MAX_STEPS = 100 # max number of steps allowed for each run of enviornment

LEARNING_RATE = 0.81 # learning rate
GAMMA = 0.96

# Picking an Action

# Remember that we can pick an action using one of two methods:
# - Randomly picking a valid action
# - Using the current Q-Table to find the best action.
# Here we will define a new value ϵ that will tell us the probabillity of selecting a random action.
# This value will start off very high and slowly decrease as the agent learns more about the enviornment.
epsilon = 0.9  # start with a 90% chance of picking a random action

# Code to pick action
if np.random.uniform(0, 1) < epsilon:  # we will check if a randomly selected value is less than epsilon.
    action = env.action_space.sample()  # take random action
else:
    action = np.argmax(Q[state, :])  # use Q table to pick best action based on current values

# Updating Q Values

# The code below implements the formula discussed above.
Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state, action])
