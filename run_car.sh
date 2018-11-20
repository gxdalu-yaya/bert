export BERT_BASE_DIR=./chinese_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
#export XNLI_DIR=./xnli
export ICS_DIR=./car_sent
gxpython="/root/.pyenv/versions/2.7.12/bin/python"

$gxpython run_classifier.py \
  --task_name=car \
  --do_train=true \
  --do_eval=true \
  --do_savemodel=false \
  --data_dir=$ICS_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=64 \
  --train_batch_size=64 \
  --learning_rate=5e-5 \
  --num_train_epochs=10.0 \
  --output_dir=/tmp/car_output/
