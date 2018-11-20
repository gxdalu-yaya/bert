#export BERT_BASE_DIR=./uncased_L-12_H-768_A-12
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
gxpython3="/root/.pyenv/versions/3.5.6/bin/python"
$gxpython3 create_pretraining_data.py \
  --input_file=./ics_corpus/ics.pretrain.data \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=80 \
  --max_predictions_per_seq=8 \
  --masked_lm_prob=0.1 \
  --random_seed=12345 \
  --dupe_factor=5
