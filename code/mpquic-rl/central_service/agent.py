# global imports
import dataclasses
from dataclasses import dataclass
import os
import threading, queue
import multiprocessing as mp
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from torch import optim

from central_service.pytorch_models.metaModel import MetaModel, ActorCritic
from central_service.variables import REMOTE_HOST

tf.disable_v2_behavior()
import time
import signal

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch
from torch.autograd import Variable
import pandas as pd

# local imports
from centraltrainer.request_handler import RequestHandler
from centraltrainer.collector import Collector
from environment.environment import Environment
from utils.logger import config_logger
from utils.queue_ops import get_request, put_response
from utils.data_transf import arrangeStateStreamsInfo, getTrainingVariables, allUnique
from training import a3c

# ---------- Global Variables ----------
S_INFO = 6  # bandwidth_path_i, path_i_mean_RTT, path_i_retransmitted_packets + path_i_lost_packets
S_LEN = 8  # take how many frames in the past
A_DIM = 2 # two actions -> path 1 or path 2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# TRAIN_SEQ_LEN = 100  # take as a train batch
TRAIN_SEQ_LEN = 100 # take as a train batch
MODEL_SAVE_INTERVAL = 64
PATHS = [1, 3] # correspond to path ids
DEFAULT_PATH = 1  # default path without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
GRADIENT_BATCH_SIZE = 8

SUMMARY_DIR = 'logs/'
LOG_FILE = './logs/tf_log.log'
NN_MODEL = None
EPOCH = 0

SSH_HOST = 'localhost'


def environment(bdw_paths: mp.Array, stop_env: mp.Event, end_of_run: mp.Event):
    rhostname = REMOTE_HOST#'mininet' + '@' + SSH_HOST
    
    config = {
        'server': 'ipc:///tmp/zmq',
        'client': 'tcp://*:5555',
        'publisher': 'tcp://*:5556',
        'subscriber': 'ipc:///tmp/pubsub'
    }
    logger = config_logger('environment', filepath='./logs/environment.log')
    env = Environment(bdw_paths, logger=logger, mconfig=config, remoteHostname=rhostname)

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
                #-------------------
                now = time.time() 
                env.run()
                end = time.time()
                #-------------------

                diff = int (end - now)
                logger.debug("Time to execute one run: {}s".format(diff))

                end_of_run.set() # set the end of run so our agent knows
                # env.spawn_middleware() # restart middleware 
            except Exception as ex:
                logger.error(ex)
                break
        time.sleep(0.1)

    env.close()
        

def old_agent():
    np.random.seed(RANDOM_SEED)

    # Create results path
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # Spawn request handler
    tqueue = queue.Queue(1)
    rhandler = RequestHandler(1, "rhandler-thread", tqueue=tqueue, host=SSH_HOST, port='5555')
    rhandler.start()

    # Spawn collector thread
    cqueue = queue.Queue(0)
    collector = Collector(2, "collector-thread", queue=cqueue, host=SSH_HOST, port='5556')
    collector.start()

    # Spawn environment # process -- not a thread
    bdw_paths = mp.Array('i', 2)
    stop_env = mp.Event()
    end_of_run = mp.Event()
    env = mp.Process(target=environment, args=(bdw_paths, stop_env, end_of_run))
    env.start()

    # keep record of threads and processes
    tp_list = [rhandler, collector, env]


    # Main training loop
    logger = config_logger('agent', 'runs/20230527_00_51_39_minrtt_test/agent.log')
    logger.info("Run Agent until training stops...")

    with tf.compat.v1.Session() as sess, open(LOG_FILE, 'w') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = EPOCH
        time_stamp = 0

        path = DEFAULT_PATH

        action_vec = np.zeros(A_DIM)
        action_vec[path] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        actor_gradient_batch = []
        critic_gradient_batch = []

        list_states = []
        while not end_of_run.is_set():
            # Get scheduling request from rhandler thread
            request, ev1 = get_request(tqueue, logger, end_of_run=end_of_run)

            # end of iterations -> exit loop -> save -> bb
            if stop_env.is_set():
                break

            if request is None and end_of_run.is_set():
                logger.info("END_OF_RUN => BATCH UPDATE")

                # get all stream_info from collector's queue
                stream_info = []
                with cqueue.mutex:
                    for elem in list(cqueue.queue):
                        stream_info.append(elem)
                    # clear the queue
                    cqueue.queue.clear()

                # Validate
                # Proceed to next run
                # logger.info("len(list_states) {} == len(stream_info) {}".format(len(list_states), len(stream_info)))
                if len(list_states) != len(stream_info) or len(list_states) == 0:
                    entropy_record = []
                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]
                    stream_info.clear()
                    list_states.clear()
                    end_of_run.clear()
                    time.sleep(0.01)
                    continue

                # Re-order rewards
                stream_info = arrangeStateStreamsInfo(list_states, stream_info)
                list_ids = [stream['StreamID'] for stream in stream_info]
                logger.info("all unique: {}".format(allUnique(list_ids, debug=True)))

                # for i, stream in enumerate(stream_info):
                #     logger.info(stream)
                #     logger.info(list_states[i]) # print this on index based

                # For each stream calculate a reward
                completion_times = []
                for index,stream in enumerate(stream_info):
                    path1_smoothed_RTT, path1_bandwidth, path1_packets, \
                    path1_retransmissions, path1_losses, \
                    path2_smoothed_RTT, path2_bandwidth, path2_packets, \
                    path2_retransmissions, path2_losses, \
                        = getTrainingVariables(list_states[index])

                    normalized_bwd_path0 = (bdw_paths[0] - 1.0) / (100.0 - 1.0)
                    normalized_bwd_path1 = (bdw_paths[1] - 1.0) / (100.0 - 1.0)
                    normalized_srtt_path0 = ((path1_smoothed_RTT * 1000.0) - 1.0) / (120.0)
                    normalized_srtt_path1 = ((path2_smoothed_RTT * 1000.0) - 1.0) / (120.0)
                    normalized_loss_path0 = ((path1_retransmissions + path1_losses) - 0.0) / 20.0
                    normalized_loss_path1 = ((path2_retransmissions + path2_losses) - 0.0) / 20.0

                    # aggr_bdw = normalized_bwd_path0 + normalized_bwd_path1
                    aggr_srtt = normalized_srtt_path0 + normalized_srtt_path1
                    aggr_loss = normalized_loss_path0 + normalized_loss_path1

                    #based on which path was picked, get bandwidth.
                    reward = (a_batch[index][0]* normalized_bwd_path0 + a_batch[index][1]*normalized_bwd_path1) - stream['CompletionTime'] - (0.8*aggr_srtt) - (1.0 * aggr_loss)
                    r_batch.append(reward)
                    completion_times.append(stream['CompletionTime'])

                # Check if we have a stream[0] = 0 add -> 0 to r_batch
                tmp_s_batch = np.stack(s_batch[:], axis=0)
                tmp_r_batch = np.vstack(r_batch[:])
                if tmp_s_batch.shape[0] > tmp_r_batch.shape[0]:
                    logger.debug("s_batch({}) > r_batch({})".format(tmp_s_batch.shape[0], tmp_r_batch.shape[0]))
                    logger.debug(tmp_s_batch[0])
                    r_batch.insert(0, 0)

                # Save metrics for debugging
                # log time_stamp, bit_rate, buffer_size, reward
                for index, stream in enumerate(stream_info):
                    path1_smoothed_RTT, path1_bandwidth, path1_packets, \
                    path1_retransmissions, path1_losses, \
                    path2_smoothed_RTT, path2_bandwidth, path2_packets, \
                    path2_retransmissions, path2_losses, \
                        = getTrainingVariables(list_states[index])
                    log_file.write(str(time_stamp) + '\t' +
                                str(PATHS[path]) + '\t' +
                                str(bdw_paths[0]) + '\t' +
                                str(bdw_paths[1]) + '\t' +
                                str(path1_smoothed_RTT) + '\t' +
                                str(path2_smoothed_RTT) + '\t' +
                                str(path1_retransmissions+path1_losses) + '\t' +
                                str(path2_retransmissions+path2_losses) + '\t' +
                                str(stream['CompletionTime']) + '\t' +
                                str(stream['Path']) + '\n')
                    log_file.flush()
                    time_stamp += 1

                # Single Training step
                # ----------------------------------------------------------------------------------------------------
                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(s_batch=np.stack(s_batch[1:], axis=0),  # ignore the first chuck
                                        a_batch=np.vstack(a_batch[1:]),  # since we don't have the
                                        r_batch=np.vstack(r_batch[1:]),  # control over it
                                        terminal=True, actor=actor, critic=critic)
                td_loss = np.mean(td_batch)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                logger.debug ("====")
                logger.debug ("Epoch: {}".format(epoch))
                msg = "TD_loss: {}, Avg_reward: {}, Avg_entropy: {}".format(td_loss, np.mean(r_batch[1:]), np.mean(entropy_record[1:]))
                logger.debug (msg)
                logger.debug ("====")
                # ----------------------------------------------------------------------------------------------------

                # Print summary for tensorflow
                # ----------------------------------------------------------------------------------------------------
                summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: td_loss,
                        summary_vars[1]: np.mean(r_batch),
                        summary_vars[2]: np.mean(entropy_record),
                        summary_vars[3]: np.mean(completion_times)
                    })

                writer.add_summary(summary_str, epoch)
                writer.flush()
                # ----------------------------------------------------------------------------------------------------

                # Update gradients
                if len(actor_gradient_batch) >= GRADIENT_BATCH_SIZE:
                    assert len(actor_gradient_batch) == len(critic_gradient_batch)

                    for i in range(len(actor_gradient_batch)):
                        actor.apply_gradients(actor_gradient_batch[i])
                        critic.apply_gradients(critic_gradient_batch[i])

                    epoch += 1
                    if epoch % MODEL_SAVE_INTERVAL == 0:
                        save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")

                entropy_record = []

                # Clear all before proceeding to next run
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                stream_info.clear()
                list_states.clear()
                end_of_run.clear()
            else:
                ev1.set() # let `producer` (rh) know we received request
                list_states.append(request)

                # The bandwidth metrics coming from MPQUIC are not correct
                # constant values not upgraded
                path1_smoothed_RTT, path1_bandwidth, path1_packets, \
                path1_retransmissions, path1_losses, \
                path2_smoothed_RTT, path2_bandwidth, path2_packets, \
                path2_retransmissions, path2_losses, \
                    = getTrainingVariables(request)

                time_stamp += 1  # in ms
                last_path = path

                # retrieve previous state
                if len(s_batch) == 0:
                    state = np.zeros((S_INFO, S_LEN))
                else:
                    state = np.array(s_batch[-1], copy=True)

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                state[0, -1] = (bdw_paths[0] - 1.0) / (100.0 - 1.0) # bandwidth path1
                state[1, -1] = (bdw_paths[1] - 1.0) / (100.0 - 1.0) # bandwidth path2
                state[2, -1] = ((path1_smoothed_RTT * 1000.0) - 1.0) / (120.0) # max RTT so far 120ms
                state[3, -1] = ((path2_smoothed_RTT * 1000.0) - 1.0) / (120.0)
                state[4, -1] = ((path1_retransmissions + path1_losses) - 0.0) / 20.0
                state[5, -1] = ((path2_retransmissions + path2_losses) - 0.0) / 20.0

                s_batch.append(state)

                action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                path = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

                action_vec = np.zeros(A_DIM)
                action_vec[path] = 1
                a_batch.append(action_vec)

                logger.debug("PATH: {}".format(path))

                entropy_record.append(a3c.compute_entropy(action_prob[0]))

                # prepare response
                response = [request['StreamID'], PATHS[path]]
                response = [str(r).encode('utf-8') for r in response]
                ev2 = threading.Event()
                put_response((response, ev2), tqueue, logger)
                ev2.wait() # blocks until `consumer` (i.e. rh) receives response

    # send kill signal to all
    stop_env.set()
    rhandler.stophandler()
    collector.stophandler()

    # wait for threads and process to finish gracefully...
    for tp in tp_list:
        tp.join()


#def agent():
#    np.random.seed(RANDOM_SEED)
#
#    # Create results path
#    if not os.path.exists(SUMMARY_DIR):
#        os.makedirs(SUMMARY_DIR)
#
#    # Spawn request handler
#    tqueue = queue.Queue(1)
#    rhandler = RequestHandler(1, "rhandler-thread", tqueue=tqueue, host=SSH_HOST, port='5555')
#    rhandler.start()
#
#    # Spawn collector thread
#    cqueue = queue.Queue(0)
#    collector = Collector(2, "collector-thread", queue=cqueue, host=SSH_HOST, port='5556')
#    collector.start()
#
#    # Spawn environment # process -- not a thread
#    bdw_paths = mp.Array('i', 2)
#    stop_env = mp.Event()
#    end_of_run = mp.Event()
#    env = mp.Process(target=environment, args=(bdw_paths, stop_env, end_of_run))
#    env.start()
#
#    # keep record of threads and processes
#    tp_list = [rhandler, collector, env]
#
#    # Main training loop
#    logger = config_logger('agent', './logs/agent.log')
#    logger.info("Run Agent until training stops...")
#
#    with open(LOG_FILE, 'w') as log_file:
#        epoch = EPOCH
#        time_stamp = 0
#
#        path = DEFAULT_PATH
#
#        model = MetaModel(A_DIM, S_INFO)
#
#        action_vec = np.zeros(A_DIM)
#        action_vec[path] = 1
#
#        s_batch = [np.zeros((S_INFO, S_LEN))]
#        a_batch = [action_vec]
#        r_batch = []
#        entropy_record = []
#
#        actor_gradient_batch = []
#        critic_gradient_batch = []
#
#        list_states = []
#        while not end_of_run.is_set():
#            # Get scheduling request from rhandler thread
#            request, ev1 = get_request(tqueue, logger, end_of_run=end_of_run)
#
#            # end of iterations -> exit loop -> save -> bb
#            if stop_env.is_set():
#                break
#
#            if request is None and end_of_run.is_set():
#                logger.info("END_OF_RUN => BATCH UPDATE")
#
#                # get all stream_info from collector's queue
#                stream_info = []
#                with cqueue.mutex:
#                    for elem in list(cqueue.queue):
#                        stream_info.append(elem)
#                    # clear the queue
#                    cqueue.queue.clear()
#
#                # Validate
#                # Proceed to next run
#                # logger.info("len(list_states) {} == len(stream_info) {}".format(len(list_states), len(stream_info)))
#                if len(list_states) != len(stream_info) or len(list_states) == 0:
#                    entropy_record = []
#                    del s_batch[:]
#                    del a_batch[:]
#                    del r_batch[:]
#                    stream_info.clear()
#                    list_states.clear()
#                    end_of_run.clear()
#                    time.sleep(0.01)
#                    continue
#
#                # Re-order rewards
#                stream_info = arrangeStateStreamsInfo(list_states, stream_info)
#                list_ids = [stream['StreamID'] for stream in stream_info]
#                logger.info("all unique: {}".format(allUnique(list_ids, debug=True)))
#
#                # for i, stream in enumerate(stream_info):
#                #     logger.info(stream)
#                #     logger.info(list_states[i]) # print this on index based
#
#                # For each stream calculate a reward
#                completion_times = []
#                for index, stream in enumerate(stream_info):
#                    path1_smoothed_RTT, path1_bandwidth, path1_packets, \
#                        path1_retransmissions, path1_losses, \
#                        path2_smoothed_RTT, path2_bandwidth, path2_packets, \
#                        path2_retransmissions, path2_losses, \
#                        = getTrainingVariables(list_states[index])
#
#                    normalized_bwd_path0 = (bdw_paths[0] - 1.0) / (100.0 - 1.0)
#                    normalized_bwd_path1 = (bdw_paths[1] - 1.0) / (100.0 - 1.0)
#                    normalized_srtt_path0 = ((path1_smoothed_RTT * 1000.0) - 1.0) / (120.0)
#                    normalized_srtt_path1 = ((path2_smoothed_RTT * 1000.0) - 1.0) / (120.0)
#                    normalized_loss_path0 = ((path1_retransmissions + path1_losses) - 0.0) / 20.0
#                    normalized_loss_path1 = ((path2_retransmissions + path2_losses) - 0.0) / 20.0
#
#                    # aggr_bdw = normalized_bwd_path0 + normalized_bwd_path1
#                    aggr_srtt = normalized_srtt_path0 + normalized_srtt_path1
#                    aggr_loss = normalized_loss_path0 + normalized_loss_path1
#
#                    reward = (a_batch[index][0] * normalized_bwd_path0 + a_batch[index][1] * normalized_bwd_path1) - \
#                             stream['CompletionTime'] - (0.8 * aggr_srtt) - (1.0 * aggr_loss)
#                    r_batch.append(reward)
#                    completion_times.append(stream['CompletionTime'])
#
#                # Check if we have a stream[0] = 0 add -> 0 to r_batch
#                tmp_s_batch = np.stack(s_batch[:], axis=0)
#                tmp_r_batch = np.vstack(r_batch[:])
#                if tmp_s_batch.shape[0] > tmp_r_batch.shape[0]:
#                    logger.debug("s_batch({}) > r_batch({})".format(tmp_s_batch.shape[0], tmp_r_batch.shape[0]))
#                    logger.debug(tmp_s_batch[0])
#                    r_batch.insert(0, 0)
#
#                # Save metrics for debugging
#                # log time_stamp, bit_rate, buffer_size, reward
#                for index, stream in enumerate(stream_info):
#                    path1_smoothed_RTT, path1_bandwidth, path1_packets, \
#                        path1_retransmissions, path1_losses, \
#                        path2_smoothed_RTT, path2_bandwidth, path2_packets, \
#                        path2_retransmissions, path2_losses, \
#                        = getTrainingVariables(list_states[index])
#                    log_file.write(str(time_stamp) + '\t' +
#                                   str(PATHS[path]) + '\t' +
#                                   str(bdw_paths[0]) + '\t' +
#                                   str(bdw_paths[1]) + '\t' +
#                                   str(path1_smoothed_RTT) + '\t' +
#                                   str(path2_smoothed_RTT) + '\t' +
#                                   str(path1_retransmissions + path1_losses) + '\t' +
#                                   str(path2_retransmissions + path2_losses) + '\t' +
#                                   str(stream['CompletionTime']) + '\t' +
#                                   str(stream['Path']) + '\n')
#                    log_file.flush()
#                    time_stamp += 1
#
#                # Single Training step
#                # ----------------------------------------------------------------------------------------------------
#                actor_gradient, critic_gradient, td_batch = \
#                    a3c.compute_gradients(s_batch=np.stack(s_batch[1:], axis=0),  # ignore the first chuck
#                                          a_batch=np.vstack(a_batch[1:]),  # since we don't have the
#                                          r_batch=np.vstack(r_batch[1:]),  # control over it
#                                          terminal=True, actor=actor, critic=critic)
#                td_loss = np.mean(td_batch)
#
#                actor_gradient_batch.append(actor_gradient)
#                critic_gradient_batch.append(critic_gradient)
#
#                logger.debug("====")
#                logger.debug("Epoch: {}".format(epoch))
#                msg = "TD_loss: {}, Avg_reward: {}, Avg_entropy: {}".format(td_loss, np.mean(r_batch[1:]),
#                                                                            np.mean(entropy_record[1:]))
#                logger.debug(msg)
#                logger.debug("====")
#                # ----------------------------------------------------------------------------------------------------
#
#                # Print summary for tensorflow
#                # ----------------------------------------------------------------------------------------------------
#                summary_str = sess.run(summary_ops, feed_dict={
#                    summary_vars[0]: td_loss,
#                    summary_vars[1]: np.mean(r_batch),
#                    summary_vars[2]: np.mean(entropy_record),
#                    summary_vars[3]: np.mean(completion_times)
#                })
#
#                writer.add_summary(summary_str, epoch)
#                writer.flush()
#                # ----------------------------------------------------------------------------------------------------
#
#                # Update gradients
#                if len(actor_gradient_batch) >= GRADIENT_BATCH_SIZE:
#                    assert len(actor_gradient_batch) == len(critic_gradient_batch)
#
#                    for i in range(len(actor_gradient_batch)):
#                        actor.apply_gradients(actor_gradient_batch[i])
#                        critic.apply_gradients(critic_gradient_batch[i])
#
#                    epoch += 1
#                    if epoch % MODEL_SAVE_INTERVAL == 0:
#                        save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")
#
#                entropy_record = []
#
#                # Clear all before proceeding to next run
#                del s_batch[:]
#                del a_batch[:]
#                del r_batch[:]
#                stream_info.clear()
#                list_states.clear()
#                end_of_run.clear()
#            else:
#                ev1.set()  # let `producer` (rh) know we received request
#                list_states.append(request)
#
#                # The bandwidth metrics coming from MPQUIC are not correct
#                # constant values not upgraded
#                path1_smoothed_RTT, path1_bandwidth, path1_packets, \
#                    path1_retransmissions, path1_losses, \
#                    path2_smoothed_RTT, path2_bandwidth, path2_packets, \
#                    path2_retransmissions, path2_losses, \
#                    = getTrainingVariables(request)
#
#                time_stamp += 1  # in ms
#                last_path = path
#
#                # retrieve previous state
#                if len(s_batch) == 0:
#                    state = np.zeros((S_INFO, S_LEN))
#                else:
#                    state = np.array(s_batch[-1], copy=True)
#
#                # dequeue history record
#                state = np.roll(state, -1, axis=1)
#
#                # this should be S_INFO number of terms
#                state[0, -1] = (bdw_paths[0] - 1.0) / (100.0 - 1.0)  # bandwidth path1
#                state[1, -1] = (bdw_paths[1] - 1.0) / (100.0 - 1.0)  # bandwidth path2
#                state[2, -1] = ((path1_smoothed_RTT * 1000.0) - 1.0) / (120.0)  # max RTT so far 120ms
#                state[3, -1] = ((path2_smoothed_RTT * 1000.0) - 1.0) / (120.0)
#                state[4, -1] = ((path1_retransmissions + path1_losses) - 0.0) / 20.0
#                state[5, -1] = ((path2_retransmissions + path2_losses) - 0.0) / 20.0
#
#                s_batch.append(state)
#
#                action_prob = model.predict(np.reshape(state[:, -1], (1, S_INFO)))
#                action_cumsum = np.cumsum(action_prob)
#                path = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
#
#                action_vec = np.zeros(A_DIM)
#                action_vec[path] = 1
#                a_batch.append(action_vec)
#
#                logger.debug("PATH: {}".format(path))
#
#                entropy_record.append(a3c.compute_entropy(action_prob[0]))
#
#                # prepare response
#                response = [request['StreamID'], PATHS[path]]
#                response = [str(r).encode('utf-8') for r in response]
#                ev2 = threading.Event()
#                put_response((response, ev2), tqueue, logger)
#                ev2.wait()  # blocks until `consumer` (i.e. rh) receives response
#
#    # send kill signal to all
#    stop_env.set()
#    rhandler.stophandler()
#    collector.stophandler()
#
#    # wait for threads and process to finish gracefully...
#    for tp in tp_list:
#        tp.join()
#
@dataclass
class NetworkState:
    normalized_bwd_path0: float
    normalized_bwd_path1: float
    normalized_srtt_path0: float
    normalized_srtt_path1: float
    normalized_loss_path0: float
    normalized_loss_path1: float

def torch_agent():
    np.random.seed(RANDOM_SEED)

    # Create results path
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # Spawn request handler
    tqueue = queue.Queue(1)
    rhandler = RequestHandler(1, "rhandler-thread", tqueue=tqueue, host=SSH_HOST, port='5555')
    rhandler.start()

    # Spawn collector thread
    cqueue = queue.Queue(0)
    collector = Collector(2, "collector-thread", queue=cqueue, host=SSH_HOST, port='5556')
    collector.start()

    # Spawn environment # process -- not a thread
    bdw_paths = mp.Array('i', 2)
    stop_env = mp.Event()
    end_of_run = mp.Event()
    env = mp.Process(target=environment, args=(bdw_paths, stop_env, end_of_run))
    env.start()

    # keep record of threads and processes
    tp_list = [rhandler, collector, env]

    num_inputs = S_INFO
    num_outputs = A_DIM

    # hyperparameters
    hidden_size = 256
    learning_rate = 3e-4

    # Constants
    GAMMA = 0.99

    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0
    epoch = 0

    logger = config_logger('agent', 'runs/20230527_00_51_39_minrtt_test/agent.log')
    logger.info("Run Agent until training stops...")

    print("Starting agent")

        # end of iterations -> exit loop -> save -> bb

    def get_net_state(request) -> NetworkState:
        path1_smoothed_RTT, path1_bandwidth, path1_packets, \
            path1_retransmissions, path1_losses, \
            path2_smoothed_RTT, path2_bandwidth, path2_packets, \
            path2_retransmissions, path2_losses, \
            = getTrainingVariables(request)

        normalized_bwd_path0 = (bdw_paths[0] - 1.0) / (100.0 - 1.0)
        normalized_bwd_path1 = (bdw_paths[1] - 1.0) / (100.0 - 1.0)
        normalized_srtt_path0 = ((path1_smoothed_RTT * 1000.0) - 1.0) / (120.0)
        normalized_srtt_path1 = ((path2_smoothed_RTT * 1000.0) - 1.0) / (120.0)
        normalized_loss_path0 = ((path1_retransmissions + path1_losses) - 0.0) / 20.0
        normalized_loss_path1 = ((path2_retransmissions + path2_losses) - 0.0) / 20.0
        return NetworkState(normalized_bwd_path0,normalized_bwd_path1,normalized_srtt_path0,normalized_srtt_path1,
                            normalized_loss_path0,normalized_loss_path1)
    def env_send(request, path):
        #action_cumsum = np.cumsum(action_prob)
        #path = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

        action_vec = np.zeros(A_DIM)
        action_vec[path] = 1
        logger.debug("PATH: {}".format(path))

        # prepare response
        response = [request['StreamID'], PATHS[path]]
        response = [str(r).encode('utf-8') for r in response]
        ev2 = threading.Event()
        put_response((response, ev2), tqueue, logger)
        ev2.wait()  # blocks until `consumer` (i.e. rh) receives response

    while not stop_env.is_set():
        log_probs = []
        values = []
        rewards = []
        states = []
        actions = []

        if stop_env.is_set():
            break

        state = np.zeros(S_INFO)
        step = 0
        print("Epoch ", epoch)
        while not end_of_run.is_set():
            # Get scheduling request from rhandler thread
            request, ev1 = get_request(tqueue, logger, end_of_run=end_of_run)

            if request is None and end_of_run.is_set():
                break

            ev1.set() # let `producer` (rh) know we received request

            ret_state = get_net_state(request)
            state = np.array(list(dataclasses.astuple(ret_state))) #convert state to list
            states.append(ret_state)

            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            #new_state, reward, done = env.step(action)
            env_send(request, action)

            action_vec = np.zeros(A_DIM)
            action_vec[action] = 1
            actions.append(action_vec)
            #logger.info('Reward: {} at step {}'.format(reward, step))

            #rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy

            step += 1
            #state = new_state

            #if done or steps == num_steps - 1:
            #    Qval, _ = actor_critic.forward(state)
            #    Qval = Qval.detach().numpy()[0, 0]
            #    all_rewards.append(np.sum(rewards))
            #    #all_lengths.append(steps)
            #    #average_lengths.append(np.mean(all_lengths[-10:]))
            #    #sys.stdout.write(
            #    #    "episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,
            #    #                                                                              np.sum(rewards),
            #    #                                                                              steps,
            #    #                                                                              average_lengths[
            #    #                                                                                  -1]))
            #    break

        if len(log_probs) == 0:
            continue

        # compute Q values
        Qval, _ = actor_critic.forward(state)
        Qval = Qval.detach().numpy()[0, 0]
        Qvals = np.zeros_like(values)

        stream_info = []
        with cqueue.mutex:
            for elem in list(cqueue.queue):
                stream_info.append(elem)
            # clear the queue
            cqueue.queue.clear()

        for i in range(len(states)):
            s = states[i]
            stream = stream_info[i]
            aggr_srtt = s.normalized_srtt_path0 + s.normalized_srtt_path1
            aggr_loss = s.normalized_loss_path0 + s.normalized_loss_path1

            reward = (actions[i][0] * s.normalized_bwd_path0 + actions[i][1] * s.normalized_bwd_path1) - stream[
                'CompletionTime'] - (0.8 * aggr_srtt) - (1.0 * aggr_loss)
            rewards.append(reward)


        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

        end_of_run.clear()

        logger.debug("====")
        logger.debug("Epoch: {}".format(epoch))
        msg = "TD_loss: {}, Avg_reward: {}, Avg_entropy: {}".format(ac_loss, np.mean(rewards),
                                                                    entropy_term)
        logger.debug(msg)
        logger.debug("====")
        epoch += 1


    stop_env.set()
    rhandler.stophandler()
    collector.stophandler()

    # wait for threads and process to finish gracefully...
    for tp in tp_list:
        tp.join()

def main():
    agent()


if __name__ == '__main__':
    main()
