#coding=utf-8
import sys
import os
import logging
import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt
#import seaborn as sns
import jieba
#from plot_attention_matrix import plot_attention_matrix

import data_helpers
from sklearn import metrics
import uniout

#tf.flags.DEFINE_string("testdata_file", "./data/query_pair.test", "source data file")
tf.flags.DEFINE_string("testdata_file", "./data/test.es", "source data file")
tf.flags.DEFINE_integer("batch_size", 24, "Batch size")

FLAGS = tf.flags.FLAGS

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

handler = logging.FileHandler("./log/pred.txt")
handler.setLevel(level = logging.INFO)
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(level = logging.INFO)
console.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(handler)


#word_index = load_wordindex("./conf/char.tsv")
model_dir = sys.argv[1] 
word_index = {}
index = 0
for line in open(os.path.join(model_dir, "vocab.txt")):
    
query = "小米圈铁耳机2购买方式"
candidate = "小米圈铁耳机2购买方式"

sent_pad_id = word_index["<pad>"]
sent_unk_id = word_index["<unk>"]
sent_end_id = word_index["</s>"]

jieba.load_userdict(os.path.join(model_dir, "userdict.txt"))

query_seged = jieba.cut(query, cut_all=False)
candidate_seged = jieba.cut(candidate, cut_all=False)

encoder_inputs_list = list()
encoder_inputs_actual_lengths_list = list()
decoder_outputs_list = list()
decoder_outputs_actual_lengths_list = list()

query_word_list, query_id_list, query_len = data_helpers.word2id(query, word_index)
print(list(query_word_list))
candidate_word_list, candidate_id_list, candidate_len = data_helpers.word2id(candidate, word_index)
print(list(candidate_word_list))

encoder_inputs_list.append(query_id_list)
encoder_inputs_actual_lengths_list.append(query_len)
decoder_outputs_list.append(candidate_id_list)
#decoder_outputs_actual_lengths_list.append(candidate_len)
decoder_outputs_actual_lengths_list.append(20)


with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
    # Load the saved meta graph and restore variables
    output_node_names = "output/predict_prob"

    encoder_inputs = sess.graph.get_operation_by_name("encoder_inputs").outputs[0]
    encoder_inputs_actual_lengths = sess.graph.get_operation_by_name("encoder_inputs_actual_lengths").outputs[0]
    decoder_outputs = sess.graph.get_operation_by_name("decoder_outputs").outputs[0]
    decoder_outputs_actual_lengths = sess.graph.get_operation_by_name("decoder_outputs_actual_lengths").outputs[0]
    batch_size = sess.graph.get_operation_by_name("batch_size").outputs[0]
    input_y = sess.graph.get_operation_by_name("input_y").outputs[0]
    test_predictions = sess.graph.get_operation_by_name("test_predictions").outputs[0]
    train_predict_prob = sess.graph.get_operation_by_name("train_predict_prob").outputs[0]
    dropout_keep_prob = sess.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    #inference_attention_matrices = sess.graph.get_operation_by_name("inference_attention_matrices").outputs[0]

    test_feed_dict = {
        encoder_inputs: encoder_inputs_list,
        encoder_inputs_actual_lengths: encoder_inputs_actual_lengths_list,
        decoder_outputs: decoder_outputs_list,
        decoder_outputs_actual_lengths: decoder_outputs_actual_lengths_list,
        batch_size: len(encoder_inputs_list),
        dropout_keep_prob: 1.0
        #input_y: one_hot_targets,
    }
    scores_batch, predictions_batch = sess.run([train_predict_prob, test_predictions], feed_dict=test_feed_dict)
    print(scores_batch[0])
    #attention_mat = inference_attention_matrices_batch[:, 0, :].T
    #filename = os.path.join("attention_visualize", "attention_matrix.png")
    #plot_attention_matrix(src=query_word_list, tgt=candidate_word_list,
    #                        matrix=attention_mat[:query_len, :candidate_len],
    #                        name=filename) 
