#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging
import os
from filelock import FileLock
import csv
import pickle
import time
from tqdm import tqdm
import random
import re
import json

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    PreTrainedTokenizer,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


class ROCStoriesTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = csv.reader(f)
                    self.data_list = [' '.join(item[2:-1]) for item in data][1:] # select first 4 sentences
                with open(file_path, 'r', encoding='utf-8') as f: # have to load twich due to csv.reader behaviour
                    data = csv.reader(f)
                    self.gold_text_list = [item[-1] for item in data][1:] # select last sentences
 
                #batch_encoding = tokenizer(data_list, add_special_tokens=True, truncation=True, )
                tokenizer.pad_token = tokenizer.bos_token
                batch_encoding = tokenizer(self.data_list, add_special_tokens=False, padding=True, max_length=block_size, return_tensors="pt", )
                self.examples = batch_encoding["input_ids"]
                #import pdb; pdb.set_trace()
                
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long), self.data_list[i], self.gold_text_list[i], 

# def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
#     file_path = args.eval_data_file if evaluate else args.train_data_file

#     return ROCStoriesTextDataset(
#         tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
#     )

def generate(args, model, tokenizer, prefix=""):
    #eval_task_names = (args.task_name,)
    #eval_outputs_dirs = (args.output_dir,)

    #for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
    # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=not test, test=test)

    # if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    #     os.makedirs(eval_output_dir)

    #args.generate_batch_size = args.per_gpu_generate_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    if args.data_file:
        eval_dataset = ROCStoriesTextDataset(
                tokenizer=tokenizer, file_path=args.data_file, block_size=125, overwrite_cache=args.overwrite_cache
            )
    else:
        # TODO
        prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    #input_ids = [item.to(args.device) for item in encoded_prompt]
    

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.generate_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    output_path = os.path.join(args.output_dir, "ROCStoreis_trn.jsonl")
    qa_format_writer = open(output_path, 'w', encoding='utf-8')

    # Generate!
    logger.info("***** Running Generating {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.generate_batch_size)
    for batch, prompt_text_list, gold_text_list in tqdm(eval_dataloader, desc="Generating"):
        
        batch_onDevice= batch.to(args.device)
        output_sequences = model.generate(
            input_ids=batch_onDevice,
            max_length=args.length + batch.shape[-1],
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            pad_token_id=50256,
        )
        generated_answers = []
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()
            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]
            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            generated_text = text[len(tokenizer.decode(batch[int(generated_sequence_idx/args.num_return_sequences)], clean_up_tokenization_spaces=True)) :]
            
            #generated_sequences.append(total_sequence)
            generated_text = re.sub(r'(\n)|(\")|(^\ )', "",generated_text)
            generated_text = re.split("\.",generated_text)[0]
            generated_answers.append(generated_text+".")

            if generated_sequence_idx % args.num_return_sequences == 0 and generated_sequence_idx != 0:
                item_dict = {
                    "context": prompt_text_list[int(generated_sequence_idx/args.num_return_sequences)],
                    "question": "",
                    "answerA": "",
                    "answerB": "",
                    "answerC": "",
                    "correct": ""
                }
                options = ["A", "B", "C"]
                item_dict["correct"] = random.choice(options)
                item_dict["answer"+item_dict["correct"]] = gold_text_list[int(generated_sequence_idx/args.num_return_sequences)]
                options.remove(item_dict["correct"])
                random.shuffle(generated_answers)
                for option, answer in zip(options, generated_answers):
                    item_dict["answer"+option] = answer
                qa_format_writer.write(json.dumps(item_dict)+"\n")

                generated_answers = []
    qa_format_writer.close()
    return 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument("--data_file", type=str, default="")    
    parser.add_argument("--overwrite_cache", action='store_true')  
    parser.add_argument("--output_dir", type=str, default="")    
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--generate_batch_size", type=int, default=20)

    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--length_penalty", type=float, default=1.0)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, padding_side="right")
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    generate(args, model, tokenizer, prefix="")
   
    # # Different models need different input formatting and/or extra arguments
    # requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    # if requires_preprocessing:
    #     prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
    #     preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

    #     if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
    #         tokenizer_kwargs = {"add_space_before_punct_symbol": True}
    #     else:
    #         tokenizer_kwargs = {}

    #     encoded_prompt = tokenizer.encode(
    #         preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
    #     )
    # else:
    #     encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    # if args.data_file:
    #     input_ids = ROCStoriesTextDataset(
    #             tokenizer=tokenizer, file_path=args.data_file, block_size=args.length, overwrite_cache=args.overwrite_cache
    #         )
    # else:
    '''
    output_path = os.path.join(args.output_dir, os.path.basename(args.data_file).split(".")[0]+"_GPT2generated.jsonl")
    qa_format_writer = open(output_path, 'w', encoding='utf-8')

    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        data_list = [(' '.join(item[2:-1]),item[-1])  for item in data][1:] # select first 4 sentences
    for idx, (prompt_text, gold_text) in tqdm(enumerate(data_list), total=len(data_list)):
        # prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
        # if prompt_text == "exit":
        #     break

        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(args.device)

        # if encoded_prompt.size()[-1] == 0:
        #     input_ids = None
        # else:
        #     input_ids = encoded_prompt

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(input_ids[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            pad_token_id=50256,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        item_dict = {
            "context": prompt_text,
            "question": "",
            "answerA": "",
            "answerB": "",
            "answerC": "",
            "correct": ""
        }
        options = ["A", "B", "C"]
        item_dict["correct"] = random.choice(options)
        item_dict["answer"+item_dict["correct"]] = gold_text
        options.remove(item_dict["correct"])
        generated_answers = []
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            generated_text = text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)) :]
            
            #generated_sequences.append(total_sequence)
            generated_text = re.sub(r"(\n)|(\")|(^\ )","",generated_text)
            generated_text = re.split("\.",generated_text)[0]
            generated_answers.append(generated_text+".")
        
        random.shuffle(generated_answers)
        for option, answer in zip(options, generated_answers):
            item_dict["answer"+option] = answer
        qa_format_writer.write(json.dumps(item_dict)+"\n")
    '''
    return


if __name__ == "__main__":
    main()
