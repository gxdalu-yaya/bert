#export BERT_BASE_DIR=./chinese_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
export BERT_BASE_DIR=./ics_pretrain_model
#export XNLI_DIR=./xnli
export ICS_DIR=./ics_ih
gxpython="/root/.pyenv/versions/2.7.12/bin/python"

$gxpython run_classifier.py \
  --task_name=ICS \
  --do_train=false \
  --do_eval=true \
  --do_savemodel=true \
  --data_dir=$ICS_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt-200000 \
  --max_seq_length=80 \
  --train_batch_size=128 \
  --learning_rate=5e-5 \
  --num_train_epochs=20.0 \
  --output_dir=/tmp/ics_ih_output/
