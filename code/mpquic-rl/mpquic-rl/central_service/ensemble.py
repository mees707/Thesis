import dataclasses
import json
import random
from datetime import datetime
import math
from dataclasses import dataclass

import gym
import numpy as np
import scipy
import torch
from avalanche_rl.models.actor_critic import ActorCriticMLP
from matplotlib import pyplot as plt
from torch import optim
import time
import pandas as pd
from tqdm import tqdm

from pytorch_models.metaModelImprove import ActorCritic  # Importing the improved version
from utils.data_transf import getTrainingVariables
from utils.logger import config_logger
from customGym.envs.NetworkEnv import NetworkEnv
from typing import NamedTuple
from pathlib import Path

from variables import *
import variables as GLOBAL_VARIABLES

class _ChangeFinderAbstract(object):
    def _add_one(self, one, ts, size):
        ts.append(one)
        if len(ts) == size+1:
            ts.pop(0)

    def _smoothing(self, ts):
        return sum(ts)/float(len(ts))

def LevinsonDurbin(r, lpcOrder):
    """
    from http://aidiary.hatenablog.com/entry/20120415/1334458954
    """
    a = np.zeros(lpcOrder + 1, dtype=np.float64)
    e = np.zeros(lpcOrder + 1, dtype=np.float64)

    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    for k in range(1, lpcOrder):
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]

        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]

class _SDAR_1Dim(object):
    def __init__(self, r, order):
        self._r = r
        self._mu = np.random.random()
        self._sigma = np.random.random()
        self._order = order
        self._c = np.random.random(self._order+1) / 100.0

    def update(self, x, term):
        assert len(term) >= self._order, "term must be order or more"
        term = np.array(term)
        self._mu = (1.0 - self._r) * self._mu + self._r * x
        for i in range(1, self._order + 1):
            self._c[i] = (1 - self._r) * self._c[i] + self._r * (x - self._mu) * (term[-i] - self._mu)
        self._c[0] = (1-self._r)*self._c[0]+self._r * (x-self._mu)*(x-self._mu)
        what, e = LevinsonDurbin(self._c, self._order)
        # print(what, term, self._mu )
        mu = self._mu.numpy() if isinstance(self._mu, torch.Tensor) else self._mu
        xhat = np.dot(-what[1:], (term[::-1] - mu))+ mu
        self._sigma = (1-self._r)*self._sigma + self._r * (x-xhat) * (x-xhat)
        return -math.log(math.exp(-0.5*(x-xhat)**2/self._sigma)/((2 * math.pi)**0.5*self._sigma**0.5)), xhat
    
class ChangeFinder(_ChangeFinderAbstract):
    def __init__(self, r=0.5, order=1, smooth=7):
        assert order > 0, "order must be 1 or more."
        assert smooth > 2, "term must be 3 or more."
        self._smooth = smooth
        self._smooth2 = int(round(self._smooth/2.0))
        self._order = order
        self._r = r
        self._ts = []
        self._first_scores = []
        self._smoothed_scores = []
        self._second_scores = []
        self._sdar_first = _SDAR_1Dim(r, self._order)
        self._sdar_second = _SDAR_1Dim(r, self._order)

    def update(self, x):
        score = 0
        predict = x
        predict2 = 0
        if len(self._ts) == self._order:  # 第一段学習
            score, predict = self._sdar_first.update(x, self._ts)
            self._add_one(score, self._first_scores, self._smooth)
        self._add_one(x, self._ts, self._order)
        second_target = None
        if len(self._first_scores) == self._smooth:  # 平滑化
            second_target = self._smoothing(self._first_scores)
        if second_target and len(self._smoothed_scores) == self._order:  # 第二段学習
            score, predict2 = self._sdar_second.update(second_target, self._smoothed_scores)
            self._add_one(score,
                          self._second_scores, self._smooth2)
        if second_target:
            self._add_one(second_target, self._smoothed_scores, self._order)
        if len(self._second_scores) == self._smooth2:
            return self._smoothing(self._second_scores), predict
        else:
            return 0.0, predict


class ChangeDetect:
    def __init__(self, logger):
        self.cf = []
        self.change_cooldown = 0
        self.cooldown_time = COOLDOWN_TIME
        self.logger = logger
        for i in range(S_INFO):
            #self.cf.append(ChangeFinder(r=0.5, order=1, smooth=3))#))
            self.cf.append(ChangeFinder(r=0.4, order=1, smooth=3))  # ))

    def add_obs(self, obs) -> bool:
        change = False
        probs = np.zeros(S_INFO)
        for i, value in enumerate(obs):
            probs[i], _ = self.cf[i].update(value)
        # prob = np.cumprod(probs)

        # print(obs)
        self.change_cooldown -= 1
        #print('network switch prob {:.1f} {:.1f}'.format(np.mean(probs), np.cumprod(probs)[-1]))  # , prob
        prob = np.cumprod(probs)[-1]
        if prob > CHANGE_PROB and self.change_cooldown <= 0:
            change = True
            self.change_cooldown = self.cooldown_time
            #print("Change detected!")
            # self.logger.info("Change detected!") #commented this part, dont think its necesary but was causing issues
        return change


def moving_average(x, w):
    if len(x) < 3:
        return x
    if w == 1:
        return x
    m = np.pad(x, int(w/2), mode='mean', stat_length=int(w/2))
    return scipy.ndimage.gaussian_filter1d(m, np.std(x) * w * 2)

def evaluate_model(model, env, num_episodes=5):
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / num_episodes

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    num_inputs = S_INFO
    num_outputs = A_DIM
    hidden_size = 128  # Define hidden_size as needed
    max_steps = 4000
    lstm_memory = None
    
    run_id = datetime.now().strftime('%Y%m%d_%H_%M_%S') + f"_{model_name}_{MODE}"
    log_dir = Path("runs/" + run_id)
    log_dir.mkdir(parents=True)

    logger = config_logger('agent', log_dir / 'agent.log')
    logger.info("Run Agent until training stops...")

    print(f"RUNNING MODEL: {model_name}")
    print(f"RUNNING MODE: {MODE}")



    # Initialize environment and models
    env: NetworkEnv = gym.make('NetworkEnv', mode=MODE)
    lstm_actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size, use_lstm=True)
    minrtt_actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size) # Assuming MinRTT is a defined class or function


    if not TRAINING and LOAD_MODEL and model_name != 'minrtt':
        checkpoint = torch.load(LSTM_TRAINED_MODEL)

        lstm_actor_critic.load_state_dict(checkpoint['model_state_dict'])
        lstm_actor_critic.load_state_dict(checkpoint['optimizer_state_dict'])
        if lstm_actor_critic.use_lstm:
            lstm_actor_critic.lstm_memory = (checkpoint['lstm_memory'][0].squeeze(0), checkpoint['lstm_memory'][1].squeeze(0))
        else:
            lstm_actor_critic.lstm_memory = None
        print("model loaded from checkpoint")


    #save variables for future reference
    with open(log_dir / "variables.json", "w") as outfile:
        env_vars = {item: getattr(GLOBAL_VARIABLES, item) for item in dir(GLOBAL_VARIABLES) if
                    not item.startswith("__") and not item.endswith("__")}
        json.dump(env_vars, outfile, indent=4, sort_keys=False)


    current_model = lstm_actor_critic
    best_model = current_model
    best_reward = float('-inf')

    rewards = []
    loss_history = []

    for epoch in range(num_epochs):
        state = env.reset()
        done = False
        while not done:
            action = current_model.select_action(state)
            next_state, reward, done, _ = env.step(action)
            loss = current_model.update(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            loss_history.append(loss)

        # Evaluate models
        if epoch % eval_interval == 0:
            lstm_reward = evaluate_model(lstm_model, env)
            minrtt_reward = evaluate_model(minrtt_model, env)

            if lstm_reward > minrtt_reward + switch_threshold:
                best_model = lstm_model
            elif minrtt_reward > lstm_reward + switch_threshold:
                best_model = minrtt_model

            current_model = best_model
            best_reward = max(lstm_reward, minrtt_reward)
            print(f'Epoch {epoch}, Best Model: {type(current_model).__name__}, Best Reward: {best_reward}')

    # Plot results
    plt.plot([x for x in rewards if x > 0])
    plt.plot(moving_average([x for x in rewards if x > 0], 20))
    plt.title('Rewards')
    plt.show()

    if len(loss_history) > 1:
        smooth_loss = moving_average(loss_history, 1)
        plt.plot(smooth_loss)
        plt.ylim(np.min(smooth_loss) - 1, np.max(smooth_loss) + 1)
        plt.title('Loss')
        plt.show()

    print("Training complete")

if __name__ == '__main__':
    main()
