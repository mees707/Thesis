import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd

        # from central_service.variables import GAMMA, LSTM_HIDDEN
from variables import GAMMA, LSTM_HIDDEN

# hyperparameters
#hidden_size = 256
#learning_rate = 3e-4
#
## Constants

#num_steps = 300
#max_episodes = 3000
layers = 1

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, use_lstm=True):
        super(ActorCritic, self).__init__()
        self.use_lstm = use_lstm
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        
        if self.use_lstm:
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.reset_lstm_memory()

        self.critic_linear = nn.Linear(hidden_size, 1)
        self.actor_linear = nn.Linear(hidden_size, num_outputs)
        
    def forward(self, state, lstm_memory):
        x = F.relu(self.linear1(state))
        
        if self.use_lstm:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            x, lstm_memory = self.lstm(x, lstm_memory)
            x = x.squeeze(0).squeeze(0)  # Remove batch and sequence dimensions
        
        value = self.critic_linear(x)
        policy_dist = F.softmax(self.actor_linear(x), dim=-1)
        
        return value, policy_dist, lstm_memory

    def lstm_after_loss(self):
        # Ensure lstm_memory is detached after loss computation to prevent backpropagation
        if self.use_lstm and self.lstm_memory is not None:
            self.lstm_memory = (self.lstm_memory[0].detach(), self.lstm_memory[1].detach())

    def reset_lstm_memory(self):
        # Reset LSTM memory to zero tensors
        if self.use_lstm:
            self.lstm_memory = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
        else:
            self.lstm_memory = None
            
    def reset_lstm_hidden(self):
        if self.lstm_memory is not None:
            self.lstm_memory = (torch.zeros(layers, LSTM_HIDDEN), self.lstm_memory[1].detach())

    def calc_a2c_loss(self, Qval, values, rewards, log_probs, entropy_terms):
        #return self.alternate_loss(Qval, values, rewards, log_probs, entropy_terms)
        # Qval = Qval.detach().numpy()[0, 0]
        values = torch.cat(values)
        Qvals = torch.zeros_like(values)  # np.zeros_like(values.detach().numpy())
        # values = values.detach()

        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        # values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals).detach()
        log_probs = torch.stack(log_probs)

        entropy_term = np.sum(entropy_terms)
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss #+ entropy_term * 0.001 #
        #loss_history_actor.append(actor_loss.detach().numpy())
        #loss_history_critic.append(critic_loss.detach().numpy())

        return ac_loss
