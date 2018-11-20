import sys
import tensorflow as tf
#import marisa_trie

model_dir = sys.argv[1]
model_num = sys.argv[2]

model_path = model_dir + "/model.ckpt-" + model_num

reader = tf.train.NewCheckpointReader(model_path)
params = reader.get_variable_to_shape_map()
for param in params:
    print param

'''
tensorname = "embedding/vocab_W"
#tensorname = "embedding/vocab_W_static"
if reader.has_tensor(tensorname):
    param_value = reader.get_tensor(tensorname)
    #print param_value

p_index = 0
for line in open("./word2vec_vocab/char.tsv","r"):
    datas = line.strip().split("\t")
    if len(datas) < 2:
        continue
    index = datas[0]
    word = datas[1]
    if index == "index":
        print "unknown" + "\t" + "\t".join(['%.5f'%value for value in param_value[p_index].tolist()])
    else:
        print word + "\t" + "\t".join(['%.5f'%value for value in param_value[p_index].tolist()])
    p_index = p_index+1
'''
