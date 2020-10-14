# warmup_steps: set to 10~20% steps of one epoch when have small batchsize 
#               to avoid that model randomly predicts even after epches of trainning

# Vanila
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-baseline \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 200 \
    --overwrite_output 

# Cleaned
export DATA_DIR=data/socialiqa_cleaned
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-cleaned \
    --per_gpu_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 200 \
    --overwrite_output

# Q2Rel
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa_q2rel \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-q2rel \
    --per_gpu_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 200 \
    --overwrite_output

# Margin Loss
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-marginloss \
    --per_gpu_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 200 \
    --overwrite_output \
    --margin_loss

# Margin Loss + Q2Rel
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa_q2rel \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-marginloss_q2rel \
    --per_gpu_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 400 \
    --overwrite_output \
    --margin_loss 
    
# Eval
export DATA_DIR=data/socialiqa_cleaned
export TRAINED_MODEL=output/socialiqa/roberta-large-cleaned
python run_multiple_choice.py \
    --task_name socialiqa \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_seq_length 80 \
    --per_device_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --model_name_or_path $TRAINED_MODEL \
    --output_dir output/socialiqa/roberta-large-eval \
    --overwrite_output  
