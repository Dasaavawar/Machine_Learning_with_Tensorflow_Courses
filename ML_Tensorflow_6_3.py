#Reinforcement Learning

#Putting it Together

#Imports and Setup
import gym
import numpy as np
import time

#Now that we know how to do some basic things we can combine these together to create our Q-Learning algorithm,
env = gym.make('FrozenLake-v0')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))

EPISODES = 1500 #How many times to run the enviornment from the beginning
MAX_STEPS = 100 #max number of steps allowed for each run of enviornment

LEARNING_RATE = 0.81 #Learning rate
GAMMA = 0.96

#If you want to see training set to true
RENDER = False

epsilon = 0.9

rewards = []
for episode in range(EPISODES):

  state = env.reset()
  for _ in range(MAX_STEPS):
    
    if RENDER:
      env.render()

    if np.random.uniform(0, 1) < epsilon:
      action = env.action_space.sample()  
    else:
      action = np.argmax(Q[state, :])

    next_state, reward, done, _ = env.step(action)

    Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

    state = next_state

    if done: 
      rewards.append(reward)
      epsilon -= 0.001
      break #Reached goal

print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}:")
#And now we can see our Q values!

#We can plot the training progress and see how the agent improved
import matplotlib.pyplot as plt

def get_average(values):
  return sum(values)/len(values)

avg_rewards = []
for i in range(0, len(rewards), 100):
  avg_rewards.append(get_average(rewards[i:i+100])) 

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()