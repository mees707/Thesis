REMOTE_PORT = '2222'
REMOTE_HOST = "mininet@localhost"
REMOTE_SERVER_RUNNER_HOSTNAME = ["mininet@localhost"]
REMOTE_SERVER_RUNNER_PORT = ["2222"]

S_INFO = 6  # bandwidth_path_i, path_i_mean_RTT, path_i_retransmitted_packets + path_i_lost_packets
A_DIM = 5#6 # 0 = minrtt, 1 = 0% path 1, 3 = 50%, 5 = 100% path 1
# so A_dim = 6 is a max output action of 5



#model_name = 'minrtt' #'FALCON'
model_name = 'LSTM'
#model_name = 'a2c'
TRAINING = False #if true, store model after done, have high exploration
MODE = 'train' if TRAINING else 'test'
LOAD_MODEL = True

SEGMENT_LIMIT = 301#90#301
EPISODES_TO_RUN = 1
START_WITH_TRACE = 9

# hyperparameters
hidden_size = 256
learning_rate = 0.001 #0.005  #3e-4
apply_loss_steps = 25
SEGMENT_UPDATES_FOR_LOSS = 3

GAMMA = 0.99
EPS_TEST = 0.05 #0.05
EPS_TRAIN = 0.15

#EPS_START = 0.05
#EPS_END = 0.05
#EPS_DECAY = 1000 #higher = slower decay
INTERMEDIATE_REWARD = False
SPARSE_REWARD = False
LOSS_SEGMENT_RETURN_BASED = False

#change detection
COOLDOWN_TIME = 60
CHANGE_PROB = 50.0

LSTM_TRAINED_MODEL = "runs/20230529_02_32_23_LSTM_train/9_model.tar"
LSTM_TRAINED_MODEL_BEFORE_9 = "runs/20230528_11_22_27_LSTM_test/8_model.tar"
LSTM_TRAINED_MODEL = LSTM_TRAINED_MODEL_BEFORE_9
A2C_TRAINED_MODEL = "runs/20230529_20_04_52_a2c_train/9_model.tar"
LSTM_HIDDEN = 6#32
