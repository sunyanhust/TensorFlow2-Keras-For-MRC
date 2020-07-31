# Linux
export BERT_MODEL_DIR=/mnt/d/bert-base-uncased
# windows
# export BERT_MODEL_DIR = F:/bert-base-uncased

python convert_pytorch_checkpoint_to_tf2.py \
  --model_type bert \
  --tf_dump_path $BERT_MODEL_DIR/  \
  --pytorch_checkpoint_path $BERT_MODEL_DIR/pytorch_model.bin \
  --config_file $BERT_MODEL_DIR/bert_config.json