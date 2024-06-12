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
import copy

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

class ReplayMemory:
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.actions = []
        self.entropy_terms = []
        #self.logger = logger
    def update_memory(self, action, value, policy_dist, reward):
        dist = policy_dist.detach().numpy()
        log_prob = torch.log(policy_dist.squeeze(0)[action])
        entropy = -np.sum(np.mean(dist) * np.log(dist))

        self.actions.append(action)

        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.entropy_terms.append(entropy)
        #self.entropy_term += entropy
    def get_memory(self, length=-1):
        if length < 0:
            length = len(self.log_probs)
        entropy_term = np.sum(self.entropy_terms[-length:])
        #Qval, values, rewards, log_probs, entropy_term
        return self.values[-1], self.values[-length:], self.rewards[-length:], self.log_probs[-length:], entropy_term

    def clear(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.actions = []
        self.entropy_terms = []

    def __len__(self):
        return len(self.values)

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
        for i in range(20):
            print(i)
            action = model.select_action(state)
            if model_name == 'minrtt':
                action == 0
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
    
    run_id = datetime.now().strftime('%Y%m%d_%H_%M_%S') + f"_MINRTT_LSTM_{MODE}"
    log_dir = Path("runs/" + run_id)
    log_dir.mkdir(parents=True)

    logger = config_logger('agent', log_dir / 'agent.log')
    logger.info("Run Agent until training stops...")

    print(f"RUNNING LSTM AND MINRTT")
    print(f"RUNNING MODE: {MODE}")

    

    # Initialize environment and models
    lstm_actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size, use_lstm=True)
    memory = ChangeDetect(logger)

    ac_optimizer = optim.Adam(lstm_actor_critic.parameters(), lr=learning_rate)


    if not TRAINING and LOAD_MODEL and model_name != 'minrtt': #last bit is depricated but leaving in for now
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

    total_steps = 0
    env: NetworkEnv = gym.make('NetworkEnv', mode=MODE)
    replay_memory = ReplayMemory()

    rewards = []
    loss_history = []

    start_time = time.time()
    print("Starting agent")
    with torch.autograd.set_detect_anomaly(True):
        for episode in range(START_WITH_TRACE, START_WITH_TRACE + EPISODES_TO_RUN):
            state = env.reset()
            state = torch.Tensor(state)
            
            lstm_actor_critic.reset_lstm_memory()  # Reset LSTM memory at the beginning of each episode

            start_time = time.time()
            reward_info = None
            rewards = []
            states = []
            loss_history = []
            print("Episode ", episode)

            last_segment_update = 0
            segment_update_count = 0

            

            current_action = 0
            for step in tqdm(range(max_steps)):


                value, policy_dist, lstm_memory = lstm_actor_critic.forward(state, lstm_actor_critic.lstm_memory)
                dist = policy_dist.detach().numpy() 

                sample = random.random()
                eps_threshold = EPS_TRAIN if TRAINING else EPS_TEST
                if sample > eps_threshold:
                    ltsm_action = np.random.choice(num_outputs, p=np.squeeze(dist))
                else:
                    ltsm_action = env.action_space.sample()
                
                minrtt_action = 0
                
                # if episode in (5, 9):
                #     current_action == ltsm_action
                #     print("currently running LSTM")
                # else: 
                #     current_action == minrtt_action
                # _, minrtt_reward, _, _ = env_clone.step(minrtt_action)

                # lstm_cum_reward += lstm_reward
                # minrtt_cum_reward += minrtt_reward



                    
                if current_action == 0:
                    new_state, reward, done, reward_info = env.step(minrtt_action)
                else: 
                    new_state, reward, done, reward_info = env.step(ltsm_action)

                

                state = torch.Tensor(new_state)
                states.append(state)    
                rewards.append(reward)
                print(state)
                segment_done = False
                if reward_info == True:
                    segment_update_count += 1

                replay_memory.update_memory(ltsm_action, value, policy_dist, reward)
                change = memory.add_obs(state)
                
                if change:
                    segment_update_count = SEGMENT_UPDATES_FOR_LOSS

                if LOSS_SEGMENT_RETURN_BASED:
                    diff = step - last_segment_update
                else:
                    diff = apply_loss_steps

                if total_steps % apply_loss_steps == 0 and total_steps > 0 and len(replay_memory) > apply_loss_steps:
                    memory_values = replay_memory.get_memory(diff)
                    ac_loss = lstm_actor_critic.calc_a2c_loss(*memory_values)
                    ac_optimizer.zero_grad()
                    ac_loss.backward()
                    ac_optimizer.step()
                    if model_name == 'LSTM':
                        lstm_actor_critic.lstm_after_loss()
                    loss_history.append(ac_loss.detach().numpy())
                    msg = "TD_loss: {}, Avg_reward: {}, Avg_entropy: {}".format(ac_loss, np.mean(memory_values[2]),
                                                                                np.mean(replay_memory.entropy_terms))
                    logger.debug(msg)
                    last_segment_update = step
                    segment_update_count = 0

                
                lstm_actor_critic.reset_lstm_memory()
                replay_memory.clear()

                total_steps += 1


                if total_steps % 50 == 0:
                    last_reward = rewards[-1]

                    if current_action == 0:
                        _, other_reward, _, _ = env.step(ltsm_action)
                        if other_reward > last_reward:
                            current_action = ltsm_action
                            print('current model is ltsm')
                    else:
                        _, other_reward, _, _ = env.step(minrtt_action)
                        if other_reward > last_reward:
                            current_action = minrtt_action
                            print('current model is minrtt')


                if done:
                    current_action = 0
                    break

                lstm_actor_critic.lstm_memory = (lstm_memory[0].detach(), lstm_memory[1].detach())

            lstm_actor_critic.reset_lstm_hidden()

            torch.save({
                'model_state_dict': lstm_actor_critic.state_dict(),
                'optimizer_state_dict': lstm_actor_critic.state_dict(),
                'lstm_memory': lstm_memory
            }, log_dir / f"{episode}_model.tar")

            np.savetxt(log_dir / f"{episode}_rewards.csv", np.array(rewards), delimiter=", ", fmt='% s')
            np.savetxt(log_dir / f"{episode}_loss.csv", np.array(loss_history), delimiter=", ", fmt='% s')
            np.savetxt(log_dir / f"{episode}_states.csv", np.array(states), delimiter=", ", fmt='% s')
            segment_rewards = pd.DataFrame(env.segment_rewards)
            segment_rewards.to_csv(log_dir / f"{episode}_segments.csv")

            segment_rewards['qoe_smooth'] = segment_rewards['qoe'].rolling(10).mean()
            segment_rewards[['qoe', 'qoe_smooth']].plot()
            plt.title(f'QOE {run_id} {episode}')
            plt.savefig(log_dir / "qoe.png")
            plt.show()

            segment_rewards['bitrate'].rolling(10).mean().plot()
            plt.title(f'bitrate {run_id} {episode}')
            plt.savefig(log_dir / "bitrate.png")
            plt.show()

            logger.debug("====")
            avg_qoe = segment_rewards[segment_rewards['segment_nr'] == segment_rewards['segment_nr'].max()]['qoe'].mean()
            logger.debug(f"Epoch: {episode}, qoe: {avg_qoe}")
            print(f"Epoch: {episode}, qoe: {avg_qoe}")
            logger.debug("====")


    end_time = time.time()
    env.close()

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