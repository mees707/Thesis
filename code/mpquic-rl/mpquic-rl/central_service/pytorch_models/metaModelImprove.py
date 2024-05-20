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
    def __init__(self, num_inputs, num_outputs, hidden_size, use_lstm=False):
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


class EWCRL():
    """
    Elastic Weight Consolidation (EWC) plugin for Reinforcement Learning,
    as presented in the original paper
    "Overcoming catastrophic forgetting in neural networks".
    As opposed to the non-rl version, importances are computed by sampling from
    a  ReplayMemory a pre-defined number of times and then running those
    batches through the network.
    """

    def __init__(
            self, ewc_lambda, replay_memory,
            mode='separate', fisher_update_steps: int = 10,
            batch_size: int = 32, start_ewc_after_steps: int = 0,
            start_ewc_after_experience: int = 1,
            decay_factor=None, keep_importance_data=False):
        """
            :param ewc_lambda: hyperparameter to weigh the penalty inside the
                    total loss. The larger the lambda, the larger the
                    regularization.
            :param replay_memory: the replay memory to sample from.
            :param batch_size: size of batches sampled during importance
                    computation.
            :param mode: `separate` to keep a separate penalty for each
                    previous experience. `online` to keep a single penalty
                    summed with a decay factor over all previous tasks.
            :param fisher_update_steps: How many times batches are sampled from
                    the ReplayMemory during computation of the Fisher
                    importance. Defaults to 10.
            :param start_ewc_after_steps: Start computing importances and
                    adding penalty only after this many steps. Defaults to 0.
            :param start_ewc_after_experience: Start computing importances and
                    adding penalty only after this many experiences.
                    Defaults to 0.
            :param decay_factor: used only if mode is `online`.
                    It specifies the decay term of the importance matrix.
            :param keep_importance_data: if True, keep in memory both parameter
                    values and importances for all previous task, for all modes.
                    If False, keep only last parameter values and importances.
                    If mode is `separate`, the value of `keep_importance_data`
                    is set to be True.
        """
        super().__init__(ewc_lambda, mode=mode, decay_factor=decay_factor,
                         keep_importance_data=keep_importance_data)
        self.fisher_updates_per_step = fisher_update_steps
        self.ewc_start_timestep = start_ewc_after_steps
        self.ewc_start_exp = start_ewc_after_experience
        self.memory = replay_memory
        self.batch_size = batch_size

    def after_training_exp(self, strategy: 'RLBaseStrategy', **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        # compute fisher information on task switch
        importances = self.compute_importances(strategy.model,
                                               strategy,
                                               strategy.optimizer,
                                               )

        self.update_importances(importances, strategy.training_exp_counter)

        self.saved_params[strategy.training_exp_counter] = copy_params_dict(
            strategy.model)
        # clear previuos parameter values
        if strategy.training_exp_counter > 0 and \
                (not self.keep_importance_data):
            del self.saved_params[strategy.training_exp_counter - 1]

    def before_backward(self, strategy: 'RLBaseStrategy', **kwargs):
        # add fisher penalty only after X steps
        if strategy.timestep >= self.ewc_start_timestep and \
                strategy.training_exp_counter >= self.ewc_start_exp:
            return super().before_backward(strategy, **kwargs)

    def compute_importances(self, model, strategy: 'RLBaseStrategy', optimizer):

        print("Computing Importances")

        # compute importances sampling minibatches from a replay memory/buffer
        model.train()

        importances = zerolike_params_dict(model)
        from avalanche_rl.training.strategies.dqn import DQNStrategy
        for _ in range(self.fisher_updates_per_step):
            if isinstance(strategy, DQNStrategy):
                # in DQN loss sampling from replay memory happens inside
                strategy.update(None)
            else:
                # sample batch
                batch = self.memory.sample_batch(
                    self.batch_size, strategy.device)
                strategy.update([batch])

            optimizer.zero_grad()
            strategy.loss.backward()

            # print(model.named_parameters(), importances)
            for (k1, p), (k2, imp) in zip(model.named_parameters(),
                                          importances.items()):
                assert (k1 == k2)
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over number of batches
        for _, imp in importances.items():
            imp.data /= float(self.fisher_updates_per_step)

        return importances

    def before_rollout(self, *args):
        pass


#ParamDict = Dict[str, Tensor]
#EwcDataType = Tuple[ParamDict, ParamDict]

class MetaModel():
    def __init__(self, num_inputs, num_outputs):
        self.actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
        self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    #def predict
    #def a2c(self):
    #    all_lengths = []
    #    average_lengths = []
    #    all_rewards = []
    #    entropy_term = 0
#
    #    for episode in range(max_episodes):
    #        log_probs = []
    #        values = []
    #        rewards = []
#
    #        state = env.reset()
    #        for steps in range(num_steps):
    #            value, policy_dist = self.actor_critic.forward(state)
    #            value = value.detach().numpy()[0, 0]
    #            dist = policy_dist.detach().numpy()
#
    #            action = np.random.choice(num_outputs, p=np.squeeze(dist))
    #            log_prob = torch.log(policy_dist.squeeze(0)[action])
    #            entropy = -np.sum(np.mean(dist) * np.log(dist))
    #            new_state, reward, done, _ = env.step(action)
#
    #            rewards.append(reward)
    #            values.append(value)
    #            log_probs.append(log_prob)
    #            entropy_term += entropy
    #            state = new_state
#
    #            if done or steps == num_steps - 1:
    #                Qval, _ = actor_critic.forward(new_state)
    #                Qval = Qval.detach().numpy()[0, 0]
    #                all_rewards.append(np.sum(rewards))
    #                all_lengths.append(steps)
    #                average_lengths.append(np.mean(all_lengths[-10:]))
    #                if episode % 10 == 0:
    #                    sys.stdout.write(
    #                        "episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,
    #                                                                                                  np.sum(rewards),
    #                                                                                                  steps,
    #                                                                                                  average_lengths[
    #                                                                                                      -1]))
    #                break
#
    #        # compute Q values
    #        Qvals = np.zeros_like(values)
    #        for t in reversed(range(len(rewards))):
    #            Qval = rewards[t] + GAMMA * Qval
    #            Qvals[t] = Qval
#
    #        # update actor critic
    #        values = torch.FloatTensor(values)
    #        Qvals = torch.FloatTensor(Qvals)
    #        log_probs = torch.stack(log_probs)
#
    #        advantage = Qvals - values
    #        actor_loss = (-log_probs * advantage).mean()
    #        critic_loss = 0.5 * advantage.pow(2).mean()
    #        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
#
    #        self.ac_optimizer.zero_grad()
    #        ac_loss.backward()
    #        self.ac_optimizer.step()
    #