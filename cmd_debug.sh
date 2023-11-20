export PYTHONPATH=./
python toolbench/inference/toolbench_server_mine.py --tool_root_dir=data/toolenv/tools/ --corpus_tsv_path=data/retrieval/G1/corpus.tsv --retrieval_model_path=/path/to/your/retrival_model --retrieved_api_nums=5 --backbone_model=toolllama --model_path=huggyllama/llama-7b --max_observation_length=1024 --method=DFS_woFilter_w2 --input_query_file=data/test_instruction/G1_instruction.json --output_answer_file=toolllama_lora_dfs_open_domain_result --rapidapi_key=000

# INTERNIMAGE
srun -p pat_earth --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 python setup.py build install --user
srun -p pat_earth --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 pip install mmcv-full==1.5.0
python test.py
GPUS=4 GPUS_PER_NODE=4 sh slurm_test.sh pat_earth test configs/coco/cascade_internimage_xl_fpn_3x_coco2.py /mnt/lustre/zengwang/downloads/cascade_internimage_xl_fpn_3x_coco.pth --eval bbox segm
GPUS=1 GPUS_PER_NODE=1 sh slurm_test.sh pat_earth test configs/coco/cascade_internimage_xl_fpn_3x_coco.py cascade_internimage_xl_fpn_3x_coco.pth --eval bbox segm



srun -p pat_taurus --quotatype=auto --job-name=toolllama \
    --ntasks=1 --gres=gpu:2 --ntasks-per-node=1 --cpus-per-task=4 --kill-on-bad-exit=1 \
   \
torchrun --nproc_per_node=2 --master_port=20001 \
    toolbench/train/thought_train_long_seq_debug.py \
    --data_path  data/toolllama_G123_dfs_train_light.json \
    --eval_data_path  data/toolllama_G123_dfs_eval_wo_thought.json \
    --output_dir work_dirs/thought/split \
    \
    --model_name_or_path /mnt/lustre/zengwang/data/llama/huggyllama/llama-7b  \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --prediction_loss_only \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --source_model_max_length 2048 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none






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




srun -p pat_taurus --quotatype=spot --job-name=toolllama \
   --ntasks=1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=40 --kill-on-bad-exit=1 \
   \
torchrun --nproc_per_node=8 --master_port=20001 \
    toolbench/train/train_mem.py \
    --data_path  data/toolllama_G123_dfs_train_light.json \
    --eval_data_path  data/toolllama_G123_dfs_eval.json \
    --output_dir work_dirs/debug \
    \
    --model_name_or_path /mnt/lustre/zengwang/data/llama/huggyllama/llama-7b  \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "epoch" \
    --prediction_loss_only \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --source_model_max_length 2048 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none
