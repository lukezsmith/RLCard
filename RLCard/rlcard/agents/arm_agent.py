''' DQN agent

The code is derived from https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py

Copyright (c) 2019 Matthew Judell
Copyright (c) 2019 DATA Lab at Texas A&M University
Copyright (c) 2016 Denny Britz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from copy import deepcopy

from scipy.stats import truncnorm


from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])


class ARMAgent(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''
    def __init__(self,
                replay_memory_size=2e5,
                replay_memory_init_size=1000,
                #  update_target_estimator_every=1000,
                tau=0.01,
                 discount_factor=0.99,
                 batch_size=64,
                 num_actions=2,
                 state_shape=None,
                train_every=64,
                 mlp_layers=None,
                learning_rate=0.001,
                 device=None):

        '''
        Q-Learning algorithm for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
            replay_memory_size (int): Size of the replay memory
            replay_memory_init_size (int): Number of random experiences to sample when initializing
              the reply memory.
            update_target_estimator_every (int): Copy parameters from the Q estimator to the
              target estimator every N steps
            discount_factor (float): Gamma discount factor
            batch_size (int): Size of batches to sample from the replay memory
            evaluate_every (int): Evaluate every N steps
            num_actions (int): The number of the actions
            state_space (list): The space of the state vector
            train_every (int): Train the network every X steps.
            mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent.
            device (torch.device): whether to use the cpu or gpu
        '''
        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.tau = tau
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every


        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # Create estimators
        self.q_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)
        self.prev_q_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)

        self.v_estimator = MLP(np.prod(state_shape), mlp_layers, 1).to(self.device)
        self.prev_v_estimator = MLP(np.prod(state_shape), mlp_layers, 1).to(self.device)

        self.v_target_estimator = MLP(np.prod(state_shape), mlp_layers, 1).to(self.device)


        # optimisers
        self.v_optimiser = torch.optim.Adam(self.v_estimator.parameters(), lr=learning_rate) 
        self.q_optimiser = torch.optim.Adam(self.q_estimator.qnet.parameters(), lr=learning_rate) 

        # loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)

    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(next_state['legal_actions'].keys()), done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_every == 0:
            self.train()

    def step(self, state):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''

        legal_actions = list(state['legal_actions'].keys())
        action, probs = self.predict(state, legal_actions)

        self._q_loss, self._v_loss = self.train()

        return action
    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        q_values = self.predict(state)
        best_action = np.argmax(q_values)

        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return best_action, info


    def predict(self, state, legal_actions):
        ''' Predict the masked Q-values

        Args:
            state (numpy.array): current state

        Returns:
            q_values (numpy.array): a 1-d array where each entry represents a Q value
        '''

        probs = np.zeros(self.num_actions)
        q_values = self.q_estimator.predict_nograd(np.expand_dims(state['obs'], 0))[0]
        v_values = self.v_estimator(torch.from_numpy(np.expand_dims(state['obs'], 0)).float().to(self.device)).detach()[0]
        

        regret = np.zeros(self.num_actions)

        for action in legal_actions:
            regret[action] = F.relu(q_values[action] - v_values)
        
        cumulative_regret = np.sum(regret)
        matched_regret = np.array([0.] * self.num_actions)

        for action in legal_actions:
            if cumulative_regret > 0:
                matched_regret[action] = regret[action] / cumulative_regret
            else:
                matched_regret[action] = 1/ len(legal_actions)

        action = np.random.choice(self.num_actions, p=matched_regret)
        probs[action] = 1.0

        return action, probs

    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''

        # check buffer is big enough to handle batch sample
        if (len(self.memory.memory)) < self.batch_size:
            return None, None

        state_batch, action_batch, reward_batch, next_state_batch, legal_actions_batch, done_batch = self.memory.sample()
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.Tensor(reward_batch).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).to(self.device)
        done_batch= torch.Tensor(done_batch).to(self.device)

        # Calculate best next actions using Q-network (Double DQN)
        target_values = self.v_target_estimator(next_state_batch).detach()
        target_values_next = (1.0 - done_batch) * torch.squeeze(target_values, 1)

        target_v = reward_batch + self.discount_factor * target_values_next

        q_value = self.q_estimator.predict(state_batch)
        q_action_value = torch.gather(torch.Tensor(q_value).to(self.device), 1, torch.unsqueeze(action_batch, 1))
        v_value = self.v_estimator(torch.Tensor(state_batch).to(self.device))

        target_q = torch.clamp(torch.squeeze(q_action_value - v_value, 1), min=0.0) + reward_batch + self.discount_factor * target_values_next

        # loss computation
        v_preds = torch.squeeze(self.v_estimator(torch.Tensor(state_batch).to(self.device)), 1)
        q_preds = torch.squeeze(torch.gather(torch.Tensor(self.q_estimator.predict(state_batch)).to(self.device), 1, torch.unsqueeze(action_batch, 1)), 1)

        target_q = target_q.detach()
        target_v = target_v.detach()

        q_preds.requires_grad = True
        
        v_loss = self.mse_loss(v_preds, target_v)
        q_loss = self.mse_loss(q_preds, target_q)
        print('\rINFO - Step {}, rl-loss: q({}), v({})'.format(self.total_t, v_loss, q_loss), end='')

        self.q_optimiser.zero_grad()
        self.v_optimiser.zero_grad()

        v_loss.backward()
        q_loss.backward()
        self.q_optimiser.step()
        self.v_optimiser.step()

        # parameter updates
        self.soft_update(self.v_target_estimator, self.v_estimator, self.tau)

        return q_loss, v_loss

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        self.memory.save(state, action, reward, next_state, legal_actions, done)

    def set_device(self, device):
        self.device = device
        self.q_estimator.device = device
        self.target_estimator.device = device

class Estimator(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.Estimator that
    uses PyTorch instead of Tensorflow.  All methods input/output np.ndarray.

    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    '''

    def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None):
        ''' Initilalize an Estimator object.

        Args:
            num_actions (int): the number output actions
            state_shape (list): the shape of the state space
            mlp_layers (list): size of outputs of mlp layers
            device (torch.device): whether to use cpu or gpu
        '''
        self.num_actions = num_actions
        self.learning_rate=learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device

        # set up Q model and place it in eval mode
        qnet = EstimatorNetwork(num_actions, state_shape, mlp_layers)
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        # initialize the weights using Xavier init
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # set up loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        # set up optimizer
        self.optimizer =  torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

    def predict(self, s):
        s = torch.from_numpy(s).float().to(self.device)
        q_as = self.qnet(s).cpu().detach().numpy()
        return q_as

    def predict_nograd(self, s):
        ''' Predicts action values, but prediction is not included
            in the computation graph.  It is used to predict optimal next
            actions in the Double-DQN algorithm.

        Args:
          s (np.ndarray): (batch, state_len)

        Returns:
          np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
          action values.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_as = self.qnet(s).cpu().numpy()
        return q_as

    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)

        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target

        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        q_as = self.qnet(s)
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss


class EstimatorNetwork(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        ''' Initialize the Q network

        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(EstimatorNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # build the Q network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        fc = [nn.Flatten()]
        fc.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        ''' Predict action values

        Args:
            s  (Tensor): (batch, state_shape)
        '''
        return self.fc_layers(s)

class SonnetLinear(nn.Module):
    def __init__(self, in_size, out_size, activate_relu=True):
        super(SonnetLinear, self).__init__()
        self._activate_relu = activate_relu
        stddev = 1.0 /math.sqrt(in_size)
        mean = 0
        lower = (-2 *stddev - mean) / stddev
        upper = (2 * stddev - mean) / stddev 

        self._weight = nn.Parameter(torch.Tensor(
            truncnorm.rvs(lower, upper, loc=mean, scale=stddev, 
            size=[out_size, in_size])))
        self._bias = nn.Parameter(torch.zeros(out_size))

    def forward(self, tensor):
        # tensor = tensor.double()
        y = F.linear(tensor, self._weight, self._bias)
        return F.relu(y) if self._activate_relu else y

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activate_final=False):
        super(MLP, self).__init__()
        self._layers = []
        for size in hidden_sizes:
            self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
            input_size = size
        self._layers.append(SonnetLinear(in_size=size, out_size=output_size, activate_relu=activate_final))
        self.model = nn.ModuleList(self._layers)
    
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, legal_actions, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, legal_actions, done)
        self.memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))
