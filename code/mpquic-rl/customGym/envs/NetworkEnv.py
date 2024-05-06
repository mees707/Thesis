from typing import NamedTuple

import numpy as np

gym_compat = True

if gym_compat:
    import gym
    from gym import spaces
else:
    import gymnasium as gym
    from gymnasium import spaces

# global imports
import dataclasses
from dataclasses import dataclass
import os
import threading, queue
import multiprocessing as mp

import time

# local imports
from central_service.centraltrainer.request_handler import RequestHandler
from central_service.centraltrainer.collector import Collector
from central_service.environment.environment import Environment
from central_service.utils.logger import config_logger
from central_service.utils.queue_ops import get_request, put_response
from central_service.utils.data_transf import arrangeStateStreamsInfo, getTrainingVariables, allUnique
from central_service.variables import REMOTE_HOST, A_DIM, INTERMEDIATE_REWARD, SPARSE_REWARD

# ---------- Global Variables ----------
#PATHS = [1, 3] # correspond to path ids
#DEFAULT_PATH = 1  # default path without agent
RANDOM_SEED = 42
#RAND_RANGE = 1000000

SUMMARY_DIR = '../../central_service/logs/'
LOG_FILE = '../../central_service/logs/tf_log.log'

SSH_HOST = 'localhost'



def environment(bdw_paths: mp.Array, stop_env: mp.Event, end_of_run: mp.Event, mode):
    rhostname = REMOTE_HOST  # 'mininet' + '@' + SSH_HOST

    config = {
        'server': 'ipc:///tmp/zmq',
        'client': 'tcp://*:5555',
        'publisher': 'tcp://*:5556',
        'subscriber': 'ipc:///tmp/pubsub'
    }
    #'ipc:///tmp/pubsub'
    logger = config_logger('environment', filepath='../../central_service/logs/environment.log')
    env = Environment(bdw_paths, logger=logger, mconfig=config, remoteHostname=rhostname, mode=mode)

    # Lets measure env runs in time
    while not stop_env.is_set():

        # Only the agent can unblock this loop, after a training-batch has been completed
        while not end_of_run.is_set():
            try:
                # update environment config from session
                if env.updateEnvironment() == -1:
                    stop_env.set()
                    end_of_run.set()
                    break

                # run a single session & measure
                # -------------------
                now = time.time()
                env.run()
                end = time.time()
                # -------------------

                diff = int(end - now)
                logger.debug("Time to execute one run: {}s".format(diff))

                end_of_run.set()  # set the end of run so our agent knows
                #stop_env.set() #run only once for now
                # env.spawn_middleware() # restart middleware
            except Exception as ex:
                logger.error(ex)
                break
        time.sleep(0.1)

    env.close()


@dataclass
class NetworkState:
    normalized_bwd_path0: float
    normalized_bwd_path1: float
    normalized_srtt_path0: float
    normalized_srtt_path1: float
    normalized_loss_path0: float
    normalized_loss_path1: float

class NetworkEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, mode='test'):

        self.observation_space = spaces.Box(0, 100, shape=(6,), dtype=np.float64)

        self.action_space = spaces.Discrete(A_DIM)
        spaces.Space()

        np.random.seed(RANDOM_SEED)

        # Create results path
        if not os.path.exists(SUMMARY_DIR):
            os.makedirs(SUMMARY_DIR)

        # Spawn request handler
        self.tqueue = queue.Queue(1)
        self.rhandler = RequestHandler(1, "rhandler-thread", tqueue=self.tqueue, host=SSH_HOST, port='5555')
        self.rhandler.start()

        # Spawn collector thread

        self.cqueue: mp.Queue[dict] = queue.Queue(0)
        self.collector = Collector(2, "collector-thread", queue=self.cqueue, host=SSH_HOST, port='5556')
        self.collector.start()

        # Spawn environment # process -- not a thread
        self.bdw_paths = mp.Array('i', 2)
        self.stop_env = mp.Event()
        self.end_of_run = mp.Event()
        env = mp.Process(target=environment, args=(self.bdw_paths, self.stop_env, self.end_of_run), kwargs={'mode':mode})
        env.start()
        time.sleep(10)


        # keep record of threads and processes
        self.tp_list = [self.rhandler, self.collector, env]

        self.logger = config_logger('agent', './logs/env_gym.log')
        self.logger.info("Run Agent until training stops...")

        #THESE SHOULD BE RESET EACH EPISODE
        self.request = None
        self.previous_reward = None
        self.segment_rewards = []

    def get_net_state(self) -> NetworkState:
        path1_smoothed_RTT, path1_bandwidth, path1_packets, \
            path1_retransmissions, path1_losses, \
            path2_smoothed_RTT, path2_bandwidth, path2_packets, \
            path2_retransmissions, path2_losses, \
            = getTrainingVariables(self.request)

        #1.576217, 311, 1071, 0, 0,\
        #0.515397, 929, 2884, 2, 0

        self.logger.info(f"Path1 bdw: {path1_bandwidth}, path2: {path2_bandwidth}")
        #normalized_bwd_path0 = (self.bdw_paths[0] - 1.0) / (100.0 - 1.0)
        #normalized_bwd_path1 = (self.bdw_paths[1] - 1.0) / (100.0 - 1.0)
        #normalized_srtt_path0 = ((path1_smoothed_RTT * 1000.0) - 1.0) / (120.0)
        #normalized_srtt_path1 = ((path2_smoothed_RTT * 1000.0) - 1.0) / (120.0)
        #normalized_loss_path0 = ((path1_retransmissions + path1_losses) - 0.0) / 20.0
        #normalized_loss_path1 = ((path2_retransmissions + path2_losses) - 0.0) / 20.0

        bdw_max = 25.0 * 1024 #convert from kb to mb
        rtt_max = 500.0 #in ms, input is in seconds
        max_loss = 10.0

        #ensure not zero
        path1_packets = max(1, path1_packets)
        path2_packets = max(1, path2_packets)
        path1_smoothed_RTT = max(0.01, path1_smoothed_RTT)
        path2_smoothed_RTT = max(0.01, path2_smoothed_RTT)

        normalized_bwd_path0 = path1_bandwidth / bdw_max
        normalized_bwd_path1 = path2_bandwidth / bdw_max
        normalized_srtt_path0 = (path1_smoothed_RTT * 1000.0) / (rtt_max)
        normalized_srtt_path1 = (path2_smoothed_RTT * 1000.0) / (rtt_max)
        normalized_loss_path0 = ((path1_retransmissions + path1_losses) / path1_packets) * max_loss
        normalized_loss_path1 = ((path2_retransmissions + path2_losses) / path2_packets) * max_loss
        return NetworkState(normalized_bwd_path0,normalized_bwd_path1,normalized_srtt_path0,normalized_srtt_path1,
                            normalized_loss_path0,normalized_loss_path1)
    def env_send(self, request, path):
        action_vec = np.zeros(A_DIM)
        action_vec[path] = 1
        self.logger.debug("PATH: {}".format(path))

        # prepare response
        response = [request['StreamID'], path]
        response = [str(r).encode('utf-8') for r in response]
        ev2 = threading.Event()
        put_response((response, ev2), self.tqueue, self.logger)
        ev2.wait()  # blocks until `consumer` (i.e. rh) receives response

# %%
# Constructing Observations From Environment States
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Since we will need to compute observations both in ``reset`` and
# ``step``, it is often convenient to have a (private) method ``_get_obs``
# that translates the environment’s state into an observation. However,
# this is not mandatory and you may as well compute observations in
# ``reset`` and ``step`` separately:

    def _get_obs(self):
        if self.request is not None:
            ret_state = self.get_net_state()
            self.logger.info(f"Received obs: {dataclasses.astuple(ret_state)}")
            return np.array(list(dataclasses.astuple(ret_state)))
            #return torch.Tensor(list(dataclasses.astuple(ret_state)))
        else:
            raise Exception('Can not get observation when request is null')

# %%
# We can also implement a similar method for the auxiliary information
# that is returned by ``step`` and ``reset``. In our case, we would like
# to provide the manhattan distance between the agent and the target:

    def _get_info(self):
        #return {
        #    "distance": np.linalg.norm(
        #        self._agent_location - self._target_location, ord=1
        #    )
        #}
        return self.segment_rewards

    def reward_old(self, action, completed):
        action_vec = np.zeros(A_DIM)
        action_vec[action] = 1

        s = self.get_net_state()
        aggr_srtt = s.normalized_srtt_path0 + s.normalized_srtt_path1
        aggr_loss = s.normalized_loss_path0 + s.normalized_loss_path1

        completed_factor = 0
        #if completed:
        #    stream_info = []
        #    with self.cqueue.mutex:
        #        for elem in list(self.cqueue.queue):
        #            #stream_info.append(elem)
        #            completed_factor += elem['CompletionTime']
        #    #stream_info['CompletionTime']

        reward = (action_vec[0] * s.normalized_bwd_path0 + action_vec[1] * s.normalized_bwd_path1) - completed_factor - (0.8 * aggr_srtt) - (1.0 * aggr_loss)
        return reward

    def reward(self, action, completed):
        # {
        #    "bitrate": current_bitrate,
        #    "down_shifts": config_dash.JSON_HANDLE['playback_info']['down_shifts'],
        #    "buffering_ratio": buffering_ratio,
        #    "initial_buffering": config_dash.JSON_HANDLE['playback_info']['initial_buffering_duration'],
        # }

        res = None
        #with self.cqueue.mutex:
        try:
            res = self.cqueue.get(block=False, timeout=0.0001)
        except queue.Empty:
            pass

        #with self.cqueue.mutex:
        #    if not self.cqueue.empty():
        #        res = list(self.cqueue.queue)[-1]
        #a = action -1
        reward = 0
        if INTERMEDIATE_REWARD:
            prob1 = 1 - ((action - 1) / (A_DIM - 1))
            obs = self.get_net_state()
            if (obs.normalized_bwd_path0 + obs.normalized_bwd_path1) > 0:
                reward = (obs.normalized_bwd_path0 * prob1 + obs.normalized_bwd_path1 * (1-prob1)) / (obs.normalized_bwd_path0 + obs.normalized_bwd_path1)
            else:
                reward = 0
            reward *= 0.5


        if res is None:
            if not SPARSE_REWARD and self.previous_reward is not None:
                return self.previous_reward, False

            return reward, False
        self.segment_rewards.append(res)

        bandwidth = res["bandwidth"] / 1048576
        bitrate = res['bitrate']/ 1048576 #MB
        down_shifts = 0#res["down_shifts"]
        buffering_ratio = res["buffering_ratio"]
        initial_buffering = res["initial_buffering"]
        buffer_i = buffering_ratio / (1 - buffering_ratio)
        if INTERMEDIATE_REWARD: reward += bandwidth * 5
        else: reward = bandwidth
        #reward = bitrate*0.5 - down_shifts - buffer_i * 5 - initial_buffering
        self.previous_reward = reward
        return reward, True

# %%
# Oftentimes, info will also contain some data that is only available
# inside the ``step`` method (e.g. individual reward terms). In that case,
# we would have to update the dictionary that is returned by ``_get_info``
# in ``step``.

# %%
# Reset
# ~~~~~
#
# The ``reset`` method will be called to initiate a new episode. You may
# assume that the ``step`` method will not be called before ``reset`` has
# been called. Moreover, ``reset`` should be called whenever a done signal
# has been issued. Users may pass the ``seed`` keyword to ``reset`` to
# initialize any random number generator that is used by the environment
# to a deterministic state. It is recommended to use the random number
# generator ``self.np_random`` that is provided by the environment’s base
# class, ``gymnasium.Env``. If you only use this RNG, you do not need to
# worry much about seeding, *but you need to remember to call
# ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
# correctly seeds the RNG. Once this is done, we can randomly set the
# state of our environment. In our case, we randomly choose the agent’s
# location and the random sample target positions, until it does not
# coincide with the agent’s position.
#
# The ``reset`` method should return a tuple of the initial observation
# and some auxiliary information. We can use the methods ``_get_obs`` and
# ``_get_info`` that we implemented earlier for that:

    def reset(self, seed=None, options=None, return_info=False):
        # We need the following line to seed self.np_random
        print("reset")

        super().reset(seed=seed)
        self.segment_rewards = []
        self.previous_reward = None
        self.request = None

        info = self._get_info()

        self.end_of_run.clear()

        with self.cqueue.mutex:
            # clear the queue
            self.cqueue.queue.clear()

        if self.request is None:
            self.request, ev1 = get_request(self.tqueue, self.logger, end_of_run=self.end_of_run)
            #print(self.request)
            if self.request is not None:
                ev1.set()  # let `producer` (rh) know we received request

        observation = self._get_obs()

        if gym_compat:
            return observation
        return observation, info

# %%
# Step
# ~~~~
#
# The ``step`` method usually contains most of the logic of your
# environment. It accepts an ``action``, computes the state of the
# environment after applying that action and returns the 4-tuple
# ``(observation, reward, done, info)``. Once the new state of the
# environment has been computed, we can check whether it is a terminal
# state and we set ``done`` accordingly. Since we are using sparse binary
# rewards in ``GridWorldEnv``, computing ``reward`` is trivial once we
# know ``done``. To gather ``observation`` and ``info``, we can again make
# use of ``_get_obs`` and ``_get_info``:

    def step(self, action):
        #print("step", action)
        self.env_send(self.request, action)
        #print("response sent")

        #terminated =
        reward, info = self.reward(action, False)
        observation = self._get_obs()
        #info = self._get_info()

        request, ev1 = get_request(self.tqueue, self.logger, end_of_run=self.end_of_run)

        if request is None and self.end_of_run.is_set():
            #reward, info = self.reward(action, True)
            print('done sent')
            self.request = None
            if gym_compat:
                return observation, reward, True, info
            return observation, reward, True, False, info

        self.request = request
        ev1.set()  # let `producer` (rh) know we received request

        if gym_compat:
            return observation, reward, False, info
        return observation, reward, False, False, info

    #def _done(self):


# %%
# Rendering
# ~~~~~~~~~
#
# Here, we are using PyGame for rendering. A similar approach to rendering
# is used in many environments that are included with Gymnasium and you
# can use it as a skeleton for your own environments:

    def render(self):
        return


# %%
# Close
# ~~~~~
#
# The ``close`` method should close any open resources that were used by
# the environment. In many cases, you don’t actually have to bother to
# implement this method. However, in our example ``render_mode`` may be
# ``"human"`` and we might need to close the window that has been opened:

    def close(self):
        self.stop_env.set()
        self.end_of_run.set()
        self.rhandler.stophandler()
        self.collector.stophandler()

        # wait for threads and process to finish gracefully...
        for tp in self.tp_list:
            tp.join()

    def sample_tasks(self, _):
        return [1]
    def set_task(self, _):
        return None