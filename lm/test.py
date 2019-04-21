from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
sys.path.append('github-scraper')

import tensorflow as tf
import pyreader
import pickle
import os
import getopt

from Trainer import get_initial_state, construct_feed_dict
from utils import *

flags = tf.flags

flags.DEFINE_string("vocab_file", "/data_normalised/mapping.map", "Name of the vocab file in data_path")
flags.DEFINE_integer("seq_length", 100, "Sequence length")
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_string("attention", "identifiers" and "", "Use the attention model")
flags.DEFINE_string("attention_variant", "input", "Variation of attention model to use. Possible values are: "
                                               "input, output")
flags.DEFINE_string("model_path", "./out/model/latest" or None, "Model parameters to load. If train=True, "
                                        "will continue training from these parameters. If test=True,"
                                        "will test using these model parameters")

FLAGS = flags.FLAGS

def adjust_flags():
    if FLAGS.attention:
        FLAGS.attention = FLAGS.attention.split("+")
        if "identifiers" in FLAGS.attention:
            FLAGS.attention.extend(["identifiers"] * (len(astwalker.possible_types()) - 1))

if __name__ == "__main__":
    adjust_flags()
    opts, args = getopt.getopt(sys.argv[1:], "p:", ["path="])
    opt, arg = opts[0]
    path = arg if opt in ("-p", "--path") else None
    config = FLAGS
    word_to_id_path = os.path.join(config.vocab_file)
    with open(word_to_id_path, "rb") as f:
        word_to_id = pickle.load(f)
    vocab = {v: k for k, v in word_to_id.items()}
    data = pyreader.get_data(None, [path], config.seq_length, word_to_id)[0]
    inputs = np.pad(data.inputs, [(0, config.batch_size - len(data.inputs)), (0, 0)], "constant")
    targets = np.pad(data.targets, [(0, config.batch_size - len(data.targets)), (0, 0)], "constant")
    actual_lengths = np.pad(data.actual_lengths, (0, config.batch_size - len(data.actual_lengths)), "constant")
    masks = np.pad(data.masks, [(config.batch_size - len(data.masks), 0), (0, 0), (0, 0)], "constant")
    data = inputs, targets, masks, data.identifier_usage, actual_lengths
    with open(os.path.join(config.model_path, "config.pkl"), "rb") as config_file:
        model_config_dict = pickle.load(config_file)
        model_config_dict["batch_size"] = config.batch_size
        if "attention" not in model_config_dict:
            model_config_dict["attention"] = config.attention
        model_config = FlagWrapper(model_config_dict)
    with tf.Graph().as_default(), tf.Session() as session:
        generator_config = copy_flags(model_config)
        generator_config.seq_length = 1
        generator_config.batch_size = 1
        with tf.variable_scope("model", reuse=None):
            model = create_model(model_config, False)
        with tf.variable_scope("model", reuse=True):
            generator_model = create_model(generator_config, False)
        init = tf.initialize_all_variables()
        session.run(init)
        load_model(session, config.model_path)
        state, att_states, att_ids, att_counts = get_initial_state(model)
        feed_dict, identifier_usage = construct_feed_dict(model, data, state, att_states, att_ids, att_counts)
        predicted = session.run(tf.arg_max(tf.nn.softmax(model.logits), 1), feed_dict=feed_dict)
        replace = lambda word: "\\n" if word == "\n" else word
        truth = [vocab[id] for id in inputs[0] if id != 0]
        predicted = [vocab[id] for id in predicted if id != 0]
        [print("%s\t%s" % (replace(t), replace(p))) for t, p in zip(truth, [''] + predicted)]
