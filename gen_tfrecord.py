import sys
from run_classifier import *

source_file = sys.argv[1]
index = sys.argv[2]
index_source_file = source_file + "_" + index 
print index_source_file

output_dir = "./ics_ih_tfrecord/"

max_seq_length = 80

processors = {
  "ics" : IcsProcessor,
}

tokenizer = tokenization.FullTokenizer(
  vocab_file="./ics_pretrain_model/vocab.txt", do_lower_case=True)

processor = processors["ics"]()
label_list = processor.get_labels()
train_examples = processor.get_train_examples(index_source_file)
train_file = os.path.join(output_dir, index + ".tf_record")
convert_examples_to_features(train_examples, label_list,
                                 max_seq_length, tokenizer, train_file)
