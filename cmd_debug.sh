export PYTHONPATH=./
python toolbench/train/thought_train_long_seq_debug.py \
--model_name_or_path=/home/SENSETIME/zengwang/Downloads/toolbench/huggyllama/llama-7b \
--data_path=data/toolllama_G123_dfs_train.json \
--eval_data_path=data/toolllama_G123_dfs_eval.json \
--conv_template=tool-llama-single-round \
--output_dir=work_dirs/debug \
--num_train_epochs=2 \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps=2 \
--prediction_loss_only \
--logging_steps=1 \
--lazy_preprocess=False \
--model_max_length=100