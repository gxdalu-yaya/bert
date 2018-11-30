#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
#import data_helpers
#from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/corpus.hei.test", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/corpus.bai.test", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
'''
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = [
        u"尊敬 的 用户 您 目前 的 积分 可 兑换 人民币 210.20 元 请 点击 gd.ylyhh.com 按 提示 激活 领取 中国移动",
        u"everything is off."
    ]
    y_test = [1, 0]

# Map data into vocabulary
#vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_path = "./container_e50_1502769053953_0334_01_000002/vocab" 
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)),np.int32)

print(vocab_processor.vocabulary_.get("gd.ylyhh.com"))
print("\nEvaluating...\n")
'''
# Evaluation
# ==================================================
model_dir = sys.argv[1]
model_num = sys.argv[2]
#checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
#model_path = "./container_e50_1502769053953_0334_01_000002/model.ckpt-15000"
model_path = model_dir + "/model.ckpt-" + model_num
reader = tf.train.NewCheckpointReader(model_path)
params = reader.get_variable_to_shape_map()
for param in params:
    print param

graph = tf.get_default_graph()  
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    dummy = tf.Variable(0)
    init_op = tf.initialize_all_variables()
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        #clear_devices = T 
        sess.run(init_op)
        saver = tf.train.import_meta_graph("{}.meta".format(model_path), clear_devices=True)
        #saver = tf.train.Saver()
        saver.restore(sess, model_path)

        input_graph_def = graph.as_graph_def()
        #output_node_names = "output/predict_prob" 
        #output_node_names = "test_predict_prob" 
        output_node_names = "Softmax" 

        #save model to pb
        output_graph = "./"+model_dir+"/frozen_model.pb"
        output_graph_def = tf.graph_util.convert_variables_to_constants(  
            sess,   
            input_graph_def,   
            output_node_names.split(",") # We split on comma for convenience  
        ) 
        # Finally we serialize and dump the output graph to the filesystem  
        tf.train.write_graph(sess.graph, model_dir, 'virus_detect.pbtxt')
        with tf.gfile.GFile(output_graph, "wb") as f:  
            f.write(output_graph_def.SerializeToString())  
        # Get the placeholders from the graph by name
        '''
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        #dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("cnn_net/output/predictions").outputs[0]
        scores = graph.get_operation_by_name("cnn_net/output/scores").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_scores = []
        true_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            #all_scores = np.concatenate([all_scores, batch_scores])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions, y_test))
out_path = os.path.join(FLAGS.checkpoint_dir, "./prediction", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
'''
