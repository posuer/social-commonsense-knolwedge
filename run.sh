# transformers 3.0.2
# pytorch 
# warmup_steps: set to 10~20% steps of one epoch when have small batchsize 
#               to avoid that model randomly predicts even after epches of trainning
# do_train: includes dev and test evluation
# do_eval, do_predict: load a trained model and evaluate on dev or test, 
# eval_step, save_step: default is evaluating/saving at 1/4, 2/4, 3/4 and 4/4 of each epoch. can set to specific step (int)


# Vanila
export DATA_DIR=socialiqa
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-baseline \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output 

#eval on above checkpoint
export DATA_DIR=socialiqa
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path output/socialiqa/roberta-large-baseline/checkpoint-3132 \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-baseline-test \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 16 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output 
    


# Q2Rel
export DATA_DIR=socialiqa
python run_multiple_choice.py \
    --task_name socialiqa_q2rel \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-q2rel-ep5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 400 \
    --overwrite_output

#eval on above checkpoint
export DATA_DIR=socialiqa
python run_multiple_choice.py \
    --task_name socialiqa_q2rel \
    --model_name_or_path  output/socialiqa/roberta-large-q2rel2/checkpoint-1827 \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-q2rel2-test2 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 400 \
    --overwrite_output



# Category

# category label classification
export DATA_DIR=socialiqa_category
python run_classification.py \
    --task_name socialiqa_class \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_seq_length 80 \
    --output_dir output/socialiqa_class/roberta-large-conf0.5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --overwrite_output \
    --confidence_threshold 0.5

# Train model with category label

export CUDA_VISIBLE_DEVICES=1,2
export DATA_DIR=socialiqa_category
python run_multiple_choice.py \
    --task_name socialiqa_q2rel_category \
    --model_name_or_path roberta-base \
    --do_train \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 8 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta_base_category_test \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 400 \
    --overwrite_output 



# Ablation
# category random
export CUDA_VISIBLE_DEVICES=3
export DATA_DIR=socialiqa_category
python run_multiple_choice.py \
    --task_name socialiqa_category_rand \
    --model_name_or_path roberta-large \
    --do_train \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 8 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta_large_category_rand \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 400 \
    --overwrite_output \
    --cache_dir cache

#q2rel random
export CUDA_VISIBLE_DEVICES=2
export DATA_DIR=socialiqa
python run_multiple_choice.py \
    --task_name socialiqa_q2rel_rand \
    --model_name_or_path roberta-large \
    --do_train \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 8 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta_large_q2rel_rand \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 400 \
    --overwrite_output \
    --cache_dir cache

