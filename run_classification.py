# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""


import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.special import softmax

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    #Trainer,
    TrainingArguments,
    set_seed,
)
from utils_classification import ClassificationDataset, Split, processors
from trainer import Trainer

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def write_pred_result(preds, output_pred_file, confidence_threshold):
    with open(output_pred_file, "w") as writer:
        for pred in preds:
            writer.write("%d\n" % (np.argmax(pred)) if np.max(softmax(pred)) >= confidence_threshold else "-1\n")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class myTrainingArguments(TrainingArguments):
    margin_loss: bool = field(default=False, metadata={"help": "Whether to use margin loss instead of cross entropy"})
    confidence_threshold: Optional[float] = field(default=-1, metadata={"help": "Only output prediction when logit is higher than threshold"})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, myTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(training_args.output_dir,"logging.txt"),
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        #force_download=True,
    )
    
    additional_special_tokens = ["[oEffect]","[oReact]","[oWant]","[xAttr]","[xEffect]","[xIntent]","[xNeed]","[xReact]","[xWant]"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        additional_special_tokens=additional_special_tokens if data_args.task_name == "socialiqa_q2rel" else [],
        #force_download=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if data_args.task_name == "socialiqa_q2rel":
        model.resize_token_embeddings(len(tokenizer)) 
    # Get datasets
    train_dataset = (
        ClassificationDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        ClassificationDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval or (training_args.do_train and training_args.evaluate_during_training)
        else None
    )

    test_dataset = (
        ClassificationDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        if training_args.do_predict
        else None
    )

   
    training_args.total_train_batch_size = (
                training_args.per_device_train_batch_size 
                * training_args.n_gpu
                * training_args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if training_args.local_rank != -1 else 1)
            )
    
    if training_args.do_train and training_args.evaluate_during_training: 
        training_args.eval_steps = int(len(train_dataset) / training_args.total_train_batch_size / 4)
        training_args.save_steps = training_args.eval_steps

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)


    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average='macro')
        report = classification_report(p.label_ids, preds )
        confusion_matrix_report = confusion_matrix(p.label_ids, preds)
        return {"acc": simple_accuracy(preds, p.label_ids), "f1":f1, "report":report, "confusion_matrix": confusion_matrix_report}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        if training_args.evaluate_during_training:
            #logger.info("*** Test / Prediction ***")
            output_eval_file = os.path.join(training_args.output_dir, "best_test_results.txt")
            
            with open(output_eval_file, "w") as writer:
                logger.info(f" best step = {trainer.best_step}")
                logger.info(f" best eval acc = {trainer.best_acc:.3f}")
                writer.write(f" best step = {trainer.best_step}\n")
                writer.write(f" best eval acc = {trainer.best_acc:.3f}\n")
                
                if test_dataset:
                    best_model_dir = os.path.join(training_args.output_dir, f"checkpoint-{trainer.best_step}")
                    best_model = AutoModelForSequenceClassification.from_pretrained(
                        best_model_dir,
                        from_tf=bool(".ckpt" in model_args.model_name_or_path),
                        config=config,
                        cache_dir=model_args.cache_dir,
                    )
                    best_trainer = Trainer(
                        model=best_model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        compute_metrics=compute_metrics,
                    )
                    output = best_trainer.predict(test_dataset)
                    result = output.metrics
                    logger.info("***** Test results *****")
                    for key, value in result.items():
                        logger.info("  %s = %s", key.replace("eval","test"), value)
                        writer.write("%s = %s\n" % (key.replace("eval","test"), value))
                    
                    
                    preds = output.predictions
                    output_pred_file = os.path.join(training_args.output_dir, "best_test_preds.txt")
                    write_pred_result(preds, output_pred_file, training_args.confidence_threshold)

                    if result["eval_acc"] < 0.69:
                        trainer.delete_model(trainer.best_step)

               
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

        output = trainer.predict(eval_dataset)
        preds = output.predictions
        output_pred_file = os.path.join(training_args.output_dir, "last_dev_preds.txt")
        write_pred_result(preds, output_pred_file, training_args.confidence_threshold)



    if training_args.do_predict:
        logger.info("*** Test / Prediction ***")

        output = trainer.predict(test_dataset)
        result = output.metrics

        output_eval_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
                results.update(result)

        preds = output.predictions
        output_pred_file = os.path.join(training_args.output_dir, "last_test_preds.txt")
        write_pred_result(preds, output_pred_file, training_args.confidence_threshold)


    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
