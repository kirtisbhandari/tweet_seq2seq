
# parameters for processing the dataset
DATA_PATH = '/mnt/data/kirtib/twitter/data/twitter/dataset1'
#LINE_FILE = 'yelp_review_data'
TRAIN_DATA = 'user_tweets_training_set.txt'
TEST_DATA = 'user_tweets_test_set.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = '/mnt/data/kirtib/twitter/seq_model/processed'
#PROCESSED_PATH = '/Users/kirti/Documents/Projects/tweet_seq2seq/processed'
CPT_PATH = 'checkpoints'
EVAL_PATH = 'checkpoints_eval'
THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3
URL_ID = 4
USR_ID = 5


PAD = '<pad>'
UNK = '<unk>'
START = '<sot>'
END = '<eot>'
URL = '<url>'
USR = '<usr>'
DATA_SIZE = 500000
TEST_SIZE = DATA_SIZE/100

# model parameters
""" Train encoder length distribution:
[175, 92, 11883, 8387, 10656, 13613, 13480, 12850, 11802, 10165,
8973, 7731, 7005, 6073, 5521, 5020, 4530, 4421, 3746, 3474, 3192,
2724, 2587, 2413, 2252, 2015, 1816, 1728, 1555, 1392, 1327, 1248,
1128, 1084, 1010, 884, 843, 755, 705, 660, 649, 594, 558, 517, 475,
426, 444, 388, 349, 337]
These buckets size seem to work the best
"""
# [19530, 17449, 17585, 23444, 22884, 16435, 17085, 18291, 18931]
# BUCKETS = [(6, 8), (8, 10), (10, 12), (13, 15), (16, 19), (19, 22), (23, 26), (29, 32), (39, 44)]

# [37049, 33519, 30223, 33513, 37371]
BUCKETS = [(50, 50)]

#BUCKETS = [(8, 10), (12, 14), (16, 19)]

NUM_LAYERS = 3
HIDDEN_SIZE = 1024
BATCH_SIZE = 64
EPOCHS = 15

LR = 0.005
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
DECAY_FACTOR = 0.99
VOCAB_SIZE = 50006
ENC_VOCAB = 50006
DEC_VOCAB = 50006
#ENC_VOCAB = 50000
#DEC_VOCAB = 50000
#VOCAB_SIZE = 50000
BEAM_SEARCH = False
BEAM_SIZE = 20
GREEDY = True
