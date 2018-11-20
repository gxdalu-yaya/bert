#coding=utf-8
import sys
import os
import logging
import numpy as np
import tensorflow as tf
import shutil
import time

import data_helpers
import readfrom_filequeue
from model import Seq2seq
from sklearn import metrics
'''
logging.basicConfig(
        level = logging.DEBUG,
        handlers = [
            logging.FileHandler("./log/log.txt"),
            logging.StreamHandler()
        ]
    )
'''


tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
#tf.flags.DEFINE_string("traindata_file", "./corpus/query_pair.train", "source data file")
#tf.flags.DEFINE_string("traindata_file", "./corpus/query_pair.all", "source data file")
tf.flags.DEFINE_string("testdata_file", "./corpus/query_pair.test", "source data file")
tf.flags.DEFINE_string("traindata_file", "./data_pipeline_phone", "source data file")
tf.flags.DEFINE_string("log_file", "./log/miweb.log", "log file")
tf.flags.DEFINE_string("model_ckpt", "miweb_ckpt", "log file")
tf.flags.DEFINE_string("maxacc_ckpt", "maxacc_ih_ckpt", "log file")
tf.flags.DEFINE_string("tensorboard_result", "tensorboard_result", "log file")
tf.flags.DEFINE_string("pretrain_embeddingfile", "./conf/word.vec", "source data file")

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("max_sequence_len", 20, "句子最大长度，问句和答案都一样")
tf.flags.DEFINE_integer("batch_size", 256, "Batch size")
tf.flags.DEFINE_integer("test_batch_size", 30000, "Batch size")
tf.flags.DEFINE_integer("hidden_size", 128, "Hidden size")
#tf.flags.DEFINE_integer("vocab_size", 6649, "Vocab size")
tf.flags.DEFINE_integer("vocab_size", 298843, "Vocab size")

tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("learning_rate", 0.05, "learning rate")

tf.flags.DEFINE_integer("num_classes", 2, "num_classes")
tf.flags.DEFINE_integer("N_GPU", 2, "N_GPU")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

handler = logging.FileHandler(FLAGS.log_file)
handler.setLevel(level = logging.INFO)
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(level = logging.INFO)
console.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(handler)

loss_save = 100.0
patience = 0

#word_index = data_helpers.load_wordindex("./conf/char.tsv")
word_index = data_helpers.load_wordindex("./conf/word.tsv")

sent_end_id = word_index["</s>"]

#train_data = data_helpers.load_data(open(FLAGS.traindata_file, "r").readlines(), word_index) 
query_list, candidate_list, labels, test_data = data_helpers.load_data(open(FLAGS.testdata_file, "r").readlines(), word_index)

embedding_mat = data_helpers.load_embedding(FLAGS.pretrain_embeddingfile, FLAGS.embedding_dim)
assert len(word_index) == len(embedding_mat)
embedding_mat = np.array(embedding_mat, dtype = np.float32)

print "embedding_mat.shape"
print embedding_mat.shape

# Training
model_ckpt_path = os.path.join(FLAGS.model_ckpt, "model")
logger.info("logger test")

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      if g is not None:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

    # Average over the 'tower' dimension.
    #print len(grads)
    if len(grads) > 0:
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
  return average_grads
 
with tf.Graph().as_default(), tf.device('/cpu:0'):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        gpu_options=gpu_options,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = Seq2seq(
            max_sequence_len=FLAGS.max_sequence_len,
            embedding_size=FLAGS.embedding_dim,
            embedding_mat=embedding_mat,
            hidden_size=FLAGS.hidden_size,
            vocab_size=FLAGS.vocab_size,
            batch_size=FLAGS.batch_size,
            sent_end_id=sent_end_id,
        )

        #encoder_inputs_batch, encoder_inputs_actual_lengths_batch, decoder_inputs_batch, decoder_inputs_actual_lengths_batch, labels_batch = readfrom_filequeue.input_pipeline2("./data_pipeline",FLAGS.batch_size, FLAGS.num_classes,num_epochs=FLAGS.num_epochs)
        batch_queue = readfrom_filequeue.input_pipeline2(FLAGS.traindata_file,FLAGS.batch_size, FLAGS.num_classes,num_epochs=FLAGS.num_epochs)
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:%d' % 2):
                test_loss_op, test_predictions_op, test_true_label_op = model.inference()

        #saver = tf.train.Saver(tf.global_variables() ,max_to_keep=FLAGS.num_checkpoints)
        saver = tf.train.Saver()
        
        train_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.tensorboard_result, "train"), sess.graph)
        test_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.tensorboard_result, "test"), sess.graph)
        dev_acc_s = tf.Variable(0.0)
        dev_loss_s = tf.Variable(0.0)

        dev_acc_p = tf.placeholder(tf.float32)
        dev_loss_p = tf.placeholder(tf.float32)
        update_acc = tf.assign(dev_acc_s,dev_acc_p)
        update_loss = tf.assign(dev_loss_s,dev_loss_p)
        dev_acc_summary = tf.summary.scalar('dev_acc_s',update_acc)
        dev_loss_summary = tf.summary.scalar('dev_loss_s',update_loss)

        dev_summary_op = tf.summary.merge([dev_acc_summary, dev_loss_summary])

        opt = tf.train.AdamOptimizer(name="AdamOptimizer")
        lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
        #opt = tf.train.AdadeltaOptimizer(
        #    learning_rate=lr, epsilon=1e-6)

        tower_grads = []
        #train_batches = data_helpers.batch_iter(train_data, FLAGS.batch_size, FLAGS.num_epochs)
        with tf.variable_scope(tf.get_variable_scope()):
            for j in xrange(FLAGS.N_GPU):
                j = j + 2
                with tf.device('/gpu:%d' % j):
                    with tf.name_scope('GPU_%d' % j) as scope:
                        #step, loss, train_summary = sess.run([model.global_step, loss_op, train_summary_op], feed_dict={model.dropout_keep_prob: 0.8, model.batch_size: 1024})
                        #losses = tf.add_n(loss_op, name='total_loss')
                        encoder_inputs_batch, encoder_inputs_actual_lengths_batch, decoder_inputs_batch, decoder_inputs_actual_lengths_batch, labels_batch = batch_queue.dequeue()
                        cross_entropy_mean, cross_entropy_mean_add_l2, accuracy, predictions_op, true_label_op = model.train_loss(encoder_inputs_batch, encoder_inputs_actual_lengths_batch, decoder_inputs_batch, decoder_inputs_actual_lengths_batch, labels_batch)
                        train_loss_summary = tf.summary.scalar("train_loss", cross_entropy_mean)
                        train_acc_summary = tf.summary.scalar("train_acc", accuracy)
                        train_summary_op = tf.summary.merge([train_loss_summary, train_acc_summary])

                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(cross_entropy_mean_add_l2)
                        tower_grads.append(grads)

        print tower_grads
        grads = average_gradients(tower_grads)
        train_op = opt.apply_gradients(grads, global_step=model.global_step)

        #capped_grads, _ = tf.clip_by_global_norm(
        #    grads, 5)
        #train_op = opt.apply_gradients(zip(capped_grads, v), global_step=model.global_step)

        max_acc = 0 
        best_acc_step = 0
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)
        sess.run(tf.assign(lr, tf.constant(FLAGS.learning_rate, dtype=tf.float32)))

        #for batch in train_batches:
        #    encoder_inputs, encoder_inputs_actual_lengths, decoder_outputs, decoder_outputs_actual_lengths, one_hot_targets = zip(*batch)
            #logger.info(np.array(decoder_outputs).shape)
        #    encoder_inputs_split, encoder_inputs_actual_lengths_split, decoder_outputs_split, decoder_outputs_actual_lengths_split, one_hot_targets_split = data_helpers.split_data(encoder_inputs, encoder_inputs_actual_lengths, decoder_outputs, decoder_outputs_actual_lengths, one_hot_targets)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        i = 0
        try:
            while not coord.should_stop():
                i = i+1
                global_step = sess.run(model.global_step) + 1
                start_time = time.time()
                _, loss_value, train_acc, train_summary = sess.run([train_op, cross_entropy_mean, accuracy, train_summary_op], feed_dict={model.dropout_keep_prob: FLAGS.dropout_keep_prob, model.batch_size: FLAGS.batch_size})
                duration = time.time() - start_time
                #print(encoder_inputs_batch.eval().tolist()) 
                #train_summary_writer.add_summary(train_summary, global_step)
                _, _, train_summary = sess.run([update_acc, update_loss ,dev_summary_op], feed_dict = {dev_acc_p:train_acc, dev_loss_p: loss_value})
                train_summary_writer.add_summary(train_summary, global_step)
                
                if i % 10000 == 0:
                    num_examples_per_step = FLAGS.batch_size * FLAGS.N_GPU
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / FLAGS.N_GPU

                    format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                    print (format_str % (i, loss_value, examples_per_sec, sec_per_batch))
 

                logger.info("step: {}, loss: {}".format(i, loss_value))
                #if (i % FLAGS.evaluate_every == 0 and i >= 10000) or i == 500:
                if i % FLAGS.evaluate_every == 0 or i == 100:
                    logger.info("\nEvalution:")
                    test_data = np.array(test_data)
                    test_batches = data_helpers.batch_iter(test_data, FLAGS.test_batch_size, 1)
                    predictions_all = np.array([])
                    input_y_all = np.array([])
                    loss_all = np.array([])

                    for test_batch in test_batches: 
                        test_encoder_inputs, test_encoder_inputs_actual_lengths, test_decoder_outputs, test_decoder_outputs_actual_lengths, one_hot_targets = zip(*test_batch)
                        #logger.info(np.array(test_encoder_inputs).shape)
                        #logger.info(np.array(test_decoder_outputs).shape)

                        test_feed_dict = {
                            model.encoder_inputs: test_encoder_inputs,
                            model.encoder_inputs_actual_lengths: test_encoder_inputs_actual_lengths,
                            model.decoder_outputs: test_decoder_outputs,
                            model.decoder_outputs_actual_lengths: test_decoder_outputs_actual_lengths,
                            model.input_y: one_hot_targets,
                            model.batch_size:len(test_encoder_inputs),
                            model.dropout_keep_prob: 1.0
                        }
                        loss_batch, predictions_batch, input_y_batch  = sess.run([test_loss_op, test_predictions_op, test_true_label_op], feed_dict=test_feed_dict)
                        #logger.info("step: {}, test_loss: {}".format(test_step, loss_batch))
                        predictions_all = np.append(predictions_all, predictions_batch)
                        input_y_all = np.append(input_y_all, input_y_batch)
                        loss_all = np.append(loss_all, loss_batch)

                    m_acc = metrics.accuracy_score(input_y_all, predictions_all)
                    m_loss = np.mean(loss_all)
                    m_confusion_matrix = metrics.confusion_matrix(input_y_all, predictions_all)
                    logger.info("confusion_matrix:")
                    logger.info(m_confusion_matrix)
                    logger.info("step {}, dev_acc {:g}".format(i, m_acc)) 
                    logger.info("precision_score:")
                    logger.info(metrics.precision_score(input_y_all, predictions_all, average=None))
                    logger.info("recall_score")
                    logger.info(metrics.recall_score(input_y_all, predictions_all, average=None))
                    logger.info("f1_score")
                    logger.info(metrics.f1_score(input_y_all, predictions_all, average=None))
                    logger.info("step: {}, test_loss: {}".format(i, m_loss))
                   
                    if m_acc > max_acc:
                        max_acc = m_acc
                        best_acc_step = i 
                        saver.save(sess, model_ckpt_path, global_step=int(i))
                        os.system("cp " +FLAGS.model_ckpt+"/model-"+str(i)+"* " + FLAGS.maxacc_ckpt+"/")
                        logger.info("best_acc_step {}, max_acc {:g}".format(best_acc_step, max_acc)) 

                    _, _, dev_summary = sess.run([update_acc, update_loss ,dev_summary_op], feed_dict = {dev_acc_p:m_acc, dev_loss_p:m_loss})
                    test_summary_writer.add_summary(dev_summary, global_step)
        
                    if m_loss < loss_save:
                        loss_save = m_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience >= 3:
                        FLAGS.learning_rate /= 2.0
                        loss_save = m_loss
                        patience = 0
                    sess.run(tf.assign(lr, tf.constant(FLAGS.learning_rate, dtype=tf.float32)))
                if i % FLAGS.checkpoint_every == 0 and i >= 10000:
                    path = saver.save(sess, model_ckpt_path, global_step=int(i))
                    saver.save(sess, model_ckpt_path, global_step=int(best_acc_step))
                    logger.info("Saved model checkpoint to {}\n".format(path))
        except tf.errors.OutOfRangeError:
            print("done training --epoch limit reached")
        finally:
            coord.request_stop()
        saver.save(sess, model_ckpt_path, global_step=best_acc_step)
        coord.join(threads)
