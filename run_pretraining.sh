#export BERT_BASE_DIR=./uncased_L-12_H-768_A-12
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
#gxpython="/root/.pyenv/versions/2.7.12/bin/python"
gxpython3="/root/.pyenv/versions/3.5.6/bin/python"

$gxpython3 run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=False \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=80 \
  --max_predictions_per_seq=8 \
  --num_train_steps=200000 \
  --num_warmup_steps=10 \
  --learning_rate=1e-4
  #--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
