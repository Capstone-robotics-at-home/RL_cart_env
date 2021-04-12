import numpy as np
import matplotlib.pyplot as plt
from cart_env import CartEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Hyper Parameters
BATCH_SIZE = 32
LR = 0.02                   # learning rate
EPSILON = 0.8               # greedy policy
GAMMA = 0.99                # reward discount
TARGET_REPLACE_ITER = 50   # target update frequency
MEMORY_CAPACITY = 2000

env=CartEnv(step_time=0.02)
env.add_obstacle([[0.5,0.5],[0.5,1],[1.5,0.5],[1.5,1]])

N_ACTIONS = 3
N_STATES = 3
ENV_A_SHAPE = 0

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 15)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(15, 15)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(15, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()


# To act greedily, just load the pkl file saved from a training. and set gamma =1
# dqn.eval_net=torch.load('xxxxxxx')
# GAMMA =1
import time 

print('\nCollecting experience...')
for i_episode in range(300):
    state = env.reset()
    episode_reward = 0

    # x,y used to plot the trajectory
    x=[]
    y=[]
    plt.ion()
    plt.cla()
    ax=plt.gca()
    t1=time.time()

    for step in range(1000):
        action = dqn.choose_action(state)

        # take action
        next_state, reward, done, info = env.step(action)

        x.append(next_state[0])
        y.append(next_state[1])

        dqn.store_transition(state, action, reward, next_state)
        episode_reward += reward                     # It's a approximate number, gamma is not considered here

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Episode: ', i_episode,
                      '| Episode_reward: ', round(episode_reward, 2))

        if done:
            print('teminal reason: ',info)
            print('num of steps: ',step)
            break
        state = next_state
    t2=time.time()
    print('time for this epoch: ',t2-t1)

    # plot this episode
    env.update_cart_polytope()
    ax.set_xlim(-0.5,2)
    ax.set_ylim(-0.5,2)
    plt.title('epoch{0}'.format(i_episode))
    ax.plot(x,y)
    env.cart_poly.plot(ax,color='green')

    # env.enlarged_ploytope.plot
    env.goal.plot(ax,alpha=0.3,color='red')
    env.obstacles[0].plot(ax,alpha=1,color='pink')
    plt.pause(0.2)


# save the neural network
torch.save(dqn.eval_net, 'cart_DQN.pkl')