# transformers 3.0.2
# pytorch 
# warmup_steps: set to 10~20% steps of one epoch when have small batchsize 
#               to avoid that model randomly predicts even after epches of trainning
# do_train: includes dev and test evluation
# do_eval, do_predict: load a trained model and evaluate on dev or test, 
# eval_step, save_step: default is evaluating/saving at 1/4, 2/4, 3/4 and 4/4 of each epoch. can set to specific step (int)
export CUDA_VISIBLE_DEVICES=1

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
    --num_train_epochs 3 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-baseline-epoch3 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output 
    

# Cleaned
export DATA_DIR=data/socialiqa_cleaned
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path roberta-large \
    --do_train \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-cleaned \
    --per_gpu_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output

# Q2Rel
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa_q2rel \
    --model_name_or_path roberta-large \
    --do_train \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-q2rel2 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output

# Margin Loss
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path roberta-large \
    --do_train \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-marginloss \
    --per_gpu_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output \
    --margin_loss

# Margin Loss + Q2Rel
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa_q2rel \
    --model_name_or_path roberta-large \
    --do_train \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-marginloss_q2rel \
    --per_gpu_eval_batch_size=8 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --evaluate_during_training \
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
# GPT2 on ROCStories Train set
python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2
    --data_file data/rocstories/ROCStories_spring2016.csv \
    --num_return_sequences 2 \
    --overwrite_cache



# Pretrain on ROCStories Train set
export TEST_FILE=/path/to/dataset/wiki.test.raw
export TRAIN_FILE="data/rocstories/ROCStories_winter2017.csv"
python run_language_modeling.py \
    --output_dir=output/rocstories/roberta-large-mlm \
    --model_type=roberta \
    --model_name_or_path=roberta-large \
    --num_train_epochs 3 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --mlm
# --do_eval \
#    --eval_data_file=$TEST_FILE \

# 2. fine-tune on SocialIQa
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path output/rocstories/roberta-large-mlm \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-ROCmlm \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output 
