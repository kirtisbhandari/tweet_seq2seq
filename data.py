from __future__ import print_function

import random
import re
import os

import numpy as np

import config
from nltk.tokenize import sent_tokenize
import codecs
from collections import OrderedDict
import itertools
from nltk.tokenize import TweetTokenizer
import json

def get_tweet_pairs(filename):
    prevSent, nextSent = [], []
    count = 0
    file_path = os.path.join(config.DATA_PATH, filename)
    with codecs.open(file_path, "r") as tweet_json:
        userTweets = json.load(tweet_json)
        for user, tweet_list in userTweets.items():
            count = count+1
            for index in range(0, len(tweet_list) -1):
                prevSent.append(tweet_list[index]['tweet'])
                nextSent.append(tweet_list[index+1]['tweet'])
    print("users ", count)
    print("tweets ", len(prevSent))
    return prevSent, nextSent


def prepare_dataset(tweet_ip, tweet_op, test=False):
    # create path to store all the train & test encoder & decoder
    make_dir(config.PROCESSED_PATH)


    train_filenames = ['train.enc', 'train.dec']
    test_filenames = ['test.enc', 'test.dec']
    files = []
    if test:
        for filename in test_filenames:
            files.append(codecs.open(os.path.join(config.PROCESSED_PATH, filename),'wb', encoding='utf8'))
    else:
        for filename in train_filenames:
            files.append(codecs.open(os.path.join(config.PROCESSED_PATH, filename),'wb', encoding='utf8'))

    for i in range(len(tweet_ip)):
        if test:
            files[0].write(tweet_ip[i] + '\n')
            files[1].write(tweet_op[i] + '\n')
        else:
            files[0].write(tweet_ip[i] + '\n')
            files[1].write(tweet_op[i] + '\n')

    for file in files:
        file.close()

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    words = []
    tokenizer = TweetTokenizer()

    tweet = line.strip().lower()
    tokens = tokenizer.tokenize(tweet)
    for token in tokens:
        add_token = None
        if not token:
            continue
        elif token.find("http://") > -1 or token.find("https://") > -1 :
            #token.replace('<URL>')
            add_token = config.URL
        elif token[0] == '@':
            add_token = config.USR
        else:
            add_token = token
        words.append(add_token)
    return words

def build_vocab(filename, normalize_digits=True):
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))

    vocab = {}
    with open(in_path, 'rb') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
#    sorted_vocab  = OrderedDict(itertools.islice(sorted_vocab.iteritems(), config.VOCAB_SIZE))
    sorted_vocab = sorted_vocab[:config.VOCAB_SIZE]
    with codecs.open(out_path, 'wb', encoding='utf8') as f:
        f.write(config.PAD + '\n')
        f.write(config.UNK + '\n')
        f.write(config.START + '\n')
        f.write(config.END + '\n')
        f.write(config.URL + '\n')
        f.write(config.USR + '\n')
        index = 6
        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                with open('config.py', 'ab') as cf:
                    if filename[-3:] == 'enc':
                        cf.write('ENC_VOCAB = ' + str(index) + '\n')
                    else:
                        cf.write('DEC_VOCAB = ' + str(index) + '\n')
                break
            f.write(word + '\n')
            index += 1

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    return [vocab.get(token, vocab[config.UNK]) for token in basic_tokenizer(line)]

def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'rb')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'wb')

    lines = in_file.read().splitlines()
    for line in lines:
        #if mode == 'dec': # we only care about '<s>' and </s> in encoder
        #    ids = [vocab[config.START]]
        #else:
        ids = []
        ids.extend(sentence2id(vocab, line))
        # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
        #if mode == 'dec':
        #    ids.append(vocab[config.END])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

def prepare_raw_data(filename, test=False):
    print('Preparing raw data into train set and test set ...')
    tweet_ip, tweet_op = get_tweet_pairs(filename)
    prepare_dataset(tweet_ip, tweet_op, test)

def process_data():
    print('Preparing data to be model-ready ...')
    build_vocab('train.enc')
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')

def load_data(enc_filename, dec_filename, max_size=100000):
    encode_file = open(os.path.join(config.PROCESSED_PATH, enc_filename), 'rb')
    decode_file = open(os.path.join(config.PROCESSED_PATH, dec_filename), 'rb')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    long_count = 0
    while encode and decode:
        if i==max_size:
            break
        if (i + 1) % 100000 == 0:
            print("Bucketing conversation number", i)
        encode_ids = [int(id_) if int(id_) < config.VOCAB_SIZE else config.UNK_ID for id_ in encode.split()]
        decode_ids = [int(id_) if int(id_) < config.VOCAB_SIZE else config.UNK_ID for id_ in decode.split()]

        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size-2 :
                encode_ids = list(reversed(_pad_input(encode_ids, encode_max_size)))
                decode_ids = [config.START_ID] + _pad_input(decode_ids, decode_max_size -2)  + [config.END_ID]
                #print(len(encode_ids), len(decode_ids))
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
            else:
                long_count = long_count +1
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    #print(data_buckets[0][0])
    print("longer datapoints: ", long_count)
    encode_file.close()
    decode_file.close()
    return data_buckets

def divide_batches(data_bucket, batch_size):
    #seems to be working correctly
    num_batches = int(len(data_bucket) / batch_size )
    if num_batches==0:
        assert False, "Not enough data. Make batch_size small."

    #print("num_batches ", num_batches)
    #print("batch_size ", batch_size)
    data  = data_bucket[:num_batches * batch_size]
    #print("before processing as array")
    #print(len(data))
    #print(len(data[0]))
    data = np.asarray(data)
    #print("after processing as array")
    # print(data)
    # print(data.shape)
    # print(data[0].shape)
    data_batches = np.split(data, num_batches) #no of batches X batch_size X 2 X datapoint size
    # print((data_batches))
    # print(data_batches[0].shape) #batch_size X 2 X datapoint size
    # print(data_batches[0][0].shape)
    # print(data_batches[0][0])
    del data
    return data_batches, num_batches


def get_masks(data_batches, batch_size):
    data_masks = []
    for batch in data_batches:
        data_masks.append(get_batch_masks(batch, batch_size))
    return data_masks


def _pad_input(input_, size):
    #pad only if input is smaller than size
    #if bigger, there is a mistake
    if(size - len(input_)) >=0:
        return input_ + [config.PAD_ID] * (size - len(input_))
    else:# (size - len(input_)) < 0:
        print("Erorr " , len(input_))
    #else:
    #    print("correct length")

def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    #print(size, batch_size)
    for length_id in xrange(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in xrange(batch_size)], dtype=np.int32))
    return batch_inputs

def get_batch(data_bucket, bucket_id, batch_size=1):
    # Return one batch to feed into the model
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for i in xrange(batch_size):
        #encoder_input, decoder_input = random.choice(data_bucket)
        encoder_input, decoder_input = data_bucket[i]
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        #decoder_inputs.append(_pad_input(decoder_input, decoder_size))
        decoder_inputs.append([config.START_ID] + _pad_input(decoder_input, encoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in xrange(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in xrange(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
        #print("each mask")
        #print(batch_mask)
    #print("batch masks")
    #print(batch_masks)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks


def get_batch_masks(batch_data, batch_size=1):
    bucket_id = 0
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    #print(len(batch_data))
    #print(batch_data[0].shape)

    for length_id in xrange(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in xrange(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = batch_data[batch_id][1][length_id + 1]
                #print("target", target)
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
            #print("each mask")
            #print(batch_mask)
        batch_masks.append(batch_mask)
    batch_masks = np.asarray(batch_masks)

    #batch_masks = np.reshape(batch_masks, (batch_size, -1))
    #print("batch masks")
    #print(batch_masks)

    return batch_masks


if __name__ == '__main__':
    prepare_raw_data(config.TRAIN_DATA)
    prepare_raw_data(config.TEST_DATA, True)
    process_data()
    """
    test_buckets = load_data("train_ids.enc", "train_ids.dec", 128)
    batches, _ = divide_batches(test_buckets[0], 64)
    #_, _, old_masks = get_batch(test_buckets[0],0, 64)
    print("from main")
    print("num batches", len(batches))
    batch_0 = batches[0]
    print(batch_0)
    new_masks = get_batch_masks(batch_0, 64)
    # encoder_inputs = batch_0[:, 0, :]
    #
    # print(len(encoder_inputs))
    # print(len(encoder_inputs[0]))
    # encoder_inputs_1 =_reshape_batch(encoder_inputs, config.BUCKETS[0][0], 64)
    # print(len(encoder_inputs_1))
    # print(len(encoder_inputs_1[0]))
    # encoder_inputs_1 =_reshape_batch(encoder_inputs_1, 64, config.BUCKETS[0][0])
    # print(np.count_nonzero(encoder_inputs - encoder_inputs_1))
    print(batch_0[:, 1, :])
    print(new_masks)
    print(len(new_masks))
    print(len(new_masks[0]))
    """
