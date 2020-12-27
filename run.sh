# transformers 3.0.2
# pytorch 
# warmup_steps: set to 10~20% steps of one epoch when have small batchsize 
#               to avoid that model randomly predicts even after epches of trainning
# do_train: includes dev and test evluation
# do_eval, do_predict: load a trained model and evaluate on dev or test, 
# eval_step, save_step: default is evaluating/saving at 1/4, 2/4, 3/4 and 4/4 of each epoch. can set to specific step (int)
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=data/socialiqa_category
python run_classification.py \
    --task_name socialiqa_class \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_seq_length 80 \
    --output_dir output/socialiqa_class/roberta-large-baseline-epoch3 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --overwrite_output 


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
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-q2rel2 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 400 \
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
    --num_train_epochs 2 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-marginloss3 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 400 \
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
    --num_train_epochs 2 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-marginloss_q2rel2 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 400 \
    --overwrite_output \
    --margin_loss 

# Category
--task_name socialiqa_category
--task_name socialiqa_q2rel_category

    
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


# Pretrain on ROCStories test set
# 1. fine-tune on ROCStories
export DATA_DIR=data/rocstories
python run_multiple_choice.py \
    --task_name rocstories_cloze \
    --model_name_or_path roberta-large \
    --do_train \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --max_seq_length 80 \
    --output_dir output/rocstories/roberta-large-cloze-finetune \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output 

# 2. fine-tune on SocialIQa
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path output/rocstories/roberta-large-cloze-finetune \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-rocCloze \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output 

# Use GPT2 to generate QA examples for ROCStories Train set
# 0. finetune GPT2 with ROCStories Train set
export CUDA_VISIBLE_DEVICES=1
export TRAIN_FILE="data/rocstories/ROCStories_1617.csv"
python run_language_modeling.py \
    --output_dir=output/rocstories/gpt2_finetune1617 \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --num_train_epochs 3 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --overwrite_output_dir

# 1. generate QA set with ROCStories
export CUDA_VISIBLE_DEVICES=0
python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=output/rocstories/gpt2_finetune1617 \
    --num_return_sequences 2 \
    --data_file data/rocstories/ROCStories_1617.csv \
    --output_dir data/rocstories_qa_1617finetunedGPT2 \
    --length_penalty 6 \
    --length 15 \
    --overwrite_cache

# 2. fine-tune on ROCStories Train QA set
# move socialIQa_v1.4_tst.jsonl and socialIQa_v1.4_dev.jsonl to the the data directory
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=data/rocstories_qa_1617random_socialiqa
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path roberta-large \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-roc1617random_socialiqa \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 2 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output 

# 3a. fine-tune on SocialIQa with marginal loss and Q2rel
export CUDA_VISIBLE_DEVICES=1
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa_q2rel \
    --model_name_or_path output/rocstories/roberta-large-1617GPT2generated_trainQA_finetune \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-marginQ2Rel-ROCtrainQA-1617GPT2 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 400 \
    --overwrite_output \
    --margin_loss

# 3. fine-tune on SocialIQa
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa \
    --model_name_or_path output/rocstories/roberta-large-1617GPT2generated_trainQA_finetune  \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-ROCtrainQA-1617GPT2generated \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --evaluate_during_training \
    --warmup_steps 200 \
    --overwrite_output



# Pretrain RoBERTa Large with ROCStories Train set
# 1. pretrain roberta large
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


# Eval and output predict result
export CUDA_VISIBLE_DEVICES=1
export DATA_DIR=data/socialiqa
python run_multiple_choice.py \
    --task_name socialiqa_q2rel \
    --model_name_or_path output/socialiqa/roberta-large-marginQ2rel-ROCtrainQA-1617random \
    --do_eval \
    --data_dir $DATA_DIR \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --max_seq_length 80 \
    --output_dir output/socialiqa/roberta-large-marginQ2rel-ROCtrainQA-1617random_pred \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 400 \
    --overwrite_output \
    --margin_loss