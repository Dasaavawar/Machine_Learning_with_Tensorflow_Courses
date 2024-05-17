# Machine Learning with TensorFlow - Code 6.1. - Reinforcement Learning


# Q-Learning Example

# For this example we will use the Q-Learning algorithm to train an agent to navigate a popular enviornment from the Open AI Gym.
# The Open AI Gym was developed so programmers could practice machine learning using unique enviornments.
# Intersting fact, Elon Musk is one of the founders of OpenAI!

# Let's start by looking at what Open AI Gym is.
# All you have to do to import and use open ai gym!

# Imports and Setup
import gym

# Once you import gym you can load an enviornment using the line gym.make("enviornment").
# We are going to use the FrozenLake enviornment
env = gym.make('FrozenLake-v0')

# There are a few other commands that can be used to interact and get information about the enviornment.
print(env.observation_space.n) # get number of states

print(env.action_space.n) # get number of actions

env.reset() # reset environment to default state

action = env.action_space.sample() # get a random action

new_state, reward, done, info = env.step(action) # take action, notice it returns information about the action

env.render() # render the GUI for the enviornment
