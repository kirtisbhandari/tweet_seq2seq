from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from model import ChatBotModel
import config
import data
import gc


def _get_random_bucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    rand = random.random()
    return min([i for i in xrange(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])

def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                        " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_masks), decoder_size))


def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """ Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created
    forward_only is set to True when you just want to evaluate on the test set,
    or when you want to the bot to be in chat mode. """
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in xrange(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in xrange(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in xrange(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
    gc.collect()

def _get_buckets():
    """ Load the dataset into buckets based on their lengths.
    train_buckets_scale is the inverval that'll help us
    choose a random bucket later on.
    """
    test_buckets = data.load_data('test_ids.enc', 'test_ids.dec', config.TEST_SIZE)
    data_buckets = data.load_data('train_ids.enc', 'train_ids.dec', config.DATA_SIZE)
    train_bucket_sizes = [len(data_buckets[b]) for b in xrange(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale
"""
def _get_buckets_gen(test_buckets, data_buckets, batch_size):
    test_buckets_gen = data.get_buckets_gen(test_buckets, batch_size)
    data_buckets_gen = data.get_buckets_gen(data_buckets, batch_size)
    return test_buckets_gen, data_buckets_gen
"""

def _get_skip_step(iteration):
    """ How many steps should the model train before it saves all the weights. """
    if iteration < 100:
        return 30
    return 200

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")

def _eval_test_set(sess, model, test_batch, test_mask):
    """ Evaluate on the test set. """
    for bucket_id in xrange(len(config.BUCKETS)):
        #as of now only 1 bucket, hence passing the batch
        #later might have to fix to get a batch here

        encoder_inputs, decoder_inputs = test_batch[:, 0, :], test_batch[:, 1, :]
        encoder_inputs =data._reshape_batch(encoder_inputs, config.BUCKETS[bucket_id][0], config.BATCH_SIZE)
        decoder_inputs =data._reshape_batch(decoder_inputs, config.BUCKETS[bucket_id][0], config.BATCH_SIZE)
        #print(len(encoder_inputs), len(decoder_inputs))
        #encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(test_batch,
        #                                                                bucket_id,
        #                                                                batch_size=config.BATCH_SIZE)

        decoder_masks = test_mask
        #decoder_masks =data._reshape_batch(test_mask, config.BUCKETS[bucket_id][0], config.BATCH_SIZE)

        if len(encoder_inputs) == 0:
            print("  Test: empty bucket %d" % (bucket_id))
            continue
        start = time.time()
        #encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(test_buckets[bucket_id],
        #                                                                bucket_id,
        #                                                                batch_size=config.BATCH_SIZE)
        _, step_loss, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                   decoder_masks, bucket_id, True)
        print('Test bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))
        #output_0 = [output_logit[0] for output_logit in output_logits ]
        #print(len(output_logits[0]))
        #print(len(output_0))
        #print(len(output_0[0]))
        #response = _construct_response(output_logits[0], inv_dec_vocab)
        #response = _construct_response(output_0, inv_dec_vocab)

        #print(response)
        return step_loss

def train():
    """ Train the bot """
    test_buckets, data_buckets, train_buckets_scale = _get_buckets()
    #test_buckets_gen, data_buckets_gen  = _get_buckets_gen(test_buckets, data_buckets, config.BATCH_SIZE)
    # in train mode, we need to create the backward path, so forwrad_only is False
    #inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

    #data_gen = data.get_batch_gen(data_buckets[0], 0, config.BATCH_SIZE)
    #test_gen = data.get_batch_gen(test_buckets[0], 0, config.BATCH_SIZE)

    divided_test, num_test_batches = data.divide_batches(test_buckets[0],  config.BATCH_SIZE)
    del test_buckets
    divided_data, num_data_batches = data.divide_batches(data_buckets[0],  config.BATCH_SIZE)
    del data_buckets
    print("train test batches ", num_data_batches, num_test_batches)

    data_masks =  data.get_masks(divided_data, config.BATCH_SIZE)
    test_masks = data.get_masks(divided_test, config.BATCH_SIZE)

    model = ChatBotModel(False, config.BATCH_SIZE)
    model.build_graph()
    saver = tf.train.Saver()

    with fix_gpu_memory() as sess:

        print('Running session')

        for i in range(0,config.EPOCHS):
            sess.run(tf.global_variables_initializer())
            _check_restore_parameters(sess, saver)
            iteration = model.global_step.eval()
            total_loss = 0
            previous_losses = []
            eval_losses = []
            bucket_id = 0
            batch_data_counter = 0
            batch_test_counter = 0
            while batch_data_counter < num_data_batches:

                skip_step = _get_skip_step(iteration)

                start = time.time()
                data_batch = divided_data[batch_data_counter]
                encoder_inputs, decoder_inputs = data_batch[:, 0, :], data_batch[:, 1, :]

                encoder_inputs =data._reshape_batch(encoder_inputs, config.BUCKETS[bucket_id][0], config.BATCH_SIZE)
                decoder_inputs =data._reshape_batch(decoder_inputs, config.BUCKETS[bucket_id][1], config.BATCH_SIZE)

                decoder_masks = data_masks[batch_data_counter]


                batch_data_counter = batch_data_counter +1
                time_taken = time.time()-start
                if time_taken >0.008:
                    print("time taken to get train data: ", time_taken)

                start = time.time()
                _, step_loss, _ = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
                total_loss += step_loss
                iteration += 1

                if iteration % skip_step == 0:
                    loss = total_loss/skip_step
                    print('Epoch {} Iter {}: lr {}, loss {}, time {}'.format(i, iteration, model.learning_rate.eval(), loss, time.time() - start))
                    #decay learning rate if loss has not decreased

                    if len(previous_losses) > 4:
                        previous_losses = previous_losses[-5:]
                        if loss > max(previous_losses[-4:]):
                            sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)

                    start = time.time()
                    total_loss = 0
                    #saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'), global_step=model.global_step)
                    if iteration % (10 * skip_step) == 0:
                    # Run evals on development set and print their loss
                    #_eval_test_set(sess, model, test_buckets, inv_dec_vocab)
                    #test_batch = test_gen.next()
                        test_batch = divided_test[batch_test_counter]
                        decoder_masks = test_masks[batch_test_counter]
                        batch_test_counter = batch_test_counter +1
                        eval_loss = _eval_test_set(sess, model, test_batch, decoder_masks)
                        if len(eval_losses) > 2:
                            eval_losses = eval_losses[-2:]
                            if eval_loss < min(eval_losses[-2:]):
                                saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'), global_step=model.global_step)
                        eval_losses.append(eval_loss)
                        start = time.time()
                    sys.stdout.flush()

            gc.collect()

def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

def _find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in xrange(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])

def _construct_response(output_logits, inv_dec_vocab):
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB

    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    print(output_logits[0])
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    #if config.EOS_ID in outputs:
    #    outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])

def chat(input=None):
    """ in test mode, we don't to create the backward path
    """
    _, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

    model = ChatBotModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()

    with fix_gpu_memory() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        output_file = open(os.path.join(config.PROCESSED_PATH, config.OUTPUT_FILE), 'a+')
        # Decode from standard input.
        max_length = config.BUCKETS[-1][0]
        print('Welcome to TensorBro. Say something. Enter to exit. Max length is', max_length)
        if input is not None:
            token_ids = data.sentence2id(enc_vocab, str(input))
            bucket_id = _find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(token_ids, [])],
                                                                            bucket_id,
                                                                            batch_size=1)
            # Get output logits for the sentence.
            _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _construct_response(output_logits, inv_dec_vocab)
            print(response)
            return
        while True:
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            output_file.write('HUMAN ++++ ' + line + '\n')
            # Get token-ids for the input sentence.
            line = config.START + " " + line + config.END
            token_ids = data.sentence2id(enc_vocab, str(line))
            if (len(token_ids) > max_length):
                print('Max length I can handle is:', max_length)
                line = _get_user_input()
                continue
            # Which bucket does it belong to?
            bucket_id = _find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            #PAD the input, output, construct decoder mask
            encoder_size, decoder_size = config.BUCKETS[bucket_id][0], config.BUCKETS[bucket_id][1]
            encoder_inputs = data._pad_input(token_ids, encoder_size)
            decoder_inputs = data._pad_input([], decoder_size)

            #print(encoder_inputs, decoder_inputs)

            batch_encoder_inputs = data._reshape_batch([encoder_inputs], encoder_size, 1)
            batch_decoder_inputs = data._reshape_batch([decoder_inputs], decoder_size, 1)

            batch_data = [[]]
            batch_data[0].append(encoder_inputs)
            batch_data[0].append(decoder_inputs)
            batch_decoder_masks = data.get_batch_masks(batch_data, batch_size=1)
            #batch_decoder_masks =data._reshape_batch(decoder_masks, config.BUCKETS[bucket_id][0], 1)


            #print("new")
            #print(len(encoder_inputs), len(decoder_inputs), len(decoder_masks))
            #print(batch_encoder_inputs, batch_decoder_inputs, batch_decoder_masks)
            old_encoder_inputs, old_decoder_inputs, old_decoder_masks = data.get_batch([(token_ids, [])],
                                                                            bucket_id,
                                                                            batch_size=1)
            # Get output logits for the sentence.
            #print("old")
            #print(old_encoder_inputs, old_decoder_inputs, old_decoder_masks)
            _, _, output_logits = run_step(sess, model, batch_encoder_inputs, batch_decoder_inputs,
                                           batch_decoder_masks, bucket_id, True)
            #_, _, output_logits_old = run_step(sess, model, batch_encoder_inputs, batch_decoder_inputs,
            #                               old_decoder_masks, bucket_id, True)
            response = _construct_response(output_logits, inv_dec_vocab)
            print(response)
            #response_old = _construct_response(output_logits_old, inv_dec_vocab)
           # print(response_old)
            output_file.write('BOT ++++ ' + response + '\n')
            response = None
        output_file.write('=============================================\n')
        output_file.close()


def fix_gpu_memory():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
        #init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
        #sess = tf.Session()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
        #K.set_session(sess)
    return sess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()
    """
    if not os.path.isdir(config.PROCESSED_PATH):
        data.prepare_raw_data()
        data.process_data()
    """
    print('Data ready!')
    print("BUCKETS %s, NUM_LAYERS %d, HIDDEN_SIZE %d, BATCH_SIZE %d, LR %0.3f, MAX_GRAD_NORM %0.2f, ENC_VOCAB %d, DEC_VOCAB %d, DECAY_FACTOR %0.3f"
       %( config.BUCKETS , config.NUM_LAYERS , config.HIDDEN_SIZE , config.BATCH_SIZE , config.LR , config.MAX_GRAD_NORM , config.ENC_VOCAB , config.DEC_VOCAB, config.DECAY_FACTOR))
    # create checkpoints folder if there isn't one already
    data.make_dir(config.CPT_PATH)

    if args.mode == 'train':
        train()
    elif args.mode == 'chat':
        chat()

if __name__ == '__main__':
    main()
