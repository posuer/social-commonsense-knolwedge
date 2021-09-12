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
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """


import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import tqdm
import random

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MultipleChoiceDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()

            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    label_list = processor.get_labels()
                    if mode == Split.dev:
                        examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    else:
                        examples = processor.get_train_examples(data_dir)
                    logger.info("Training examples: %s", len(examples))
                    self.features = convert_examples_to_features(
                        examples,
                        label_list,
                        max_seq_length,
                        tokenizer,
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


if is_tf_available():
    import tensorflow as tf

    class TFMultipleChoiceDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = 128,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()

            logger.info(f"Creating features from dataset file at {data_dir}")
            label_list = processor.get_labels()
            if mode == Split.dev:
                examples = processor.get_dev_examples(data_dir)
            elif mode == Split.test:
                examples = processor.get_test_examples(data_dir)
            else:
                examples = processor.get_train_examples(data_dir)
            logger.info("Training examples: %s", len(examples))

            self.features = convert_examples_to_features(
                examples,
                label_list,
                max_seq_length,
                tokenizer,
            )

            def gen():
                for (ex_index, ex) in tqdm.tqdm(enumerate(self.features), desc="convert examples to features"):
                    if ex_index % 10000 == 0:
                        logger.info("Writing example %d of %d" % (ex_index, len(examples)))

                    yield (
                        {
                            "example_id": 0,
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        },
                        ex.label,
                    )

            self.dataset = tf.data.Dataset.from_generator(
                gen,
                (
                    {
                        "example_id": tf.int32,
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                        "token_type_ids": tf.int32,
                    },
                    tf.int64,
                ),
                (
                    {
                        "example_id": tf.TensorShape([]),
                        "input_ids": tf.TensorShape([None, None]),
                        "attention_mask": tf.TensorShape([None, None]),
                        "token_type_ids": tf.TensorShape([None, None]),
                    },
                    tf.TensorShape([]),
                ),
            )

        def get_dataset(self):
            self.dataset = self.dataset.apply(tf.data.experimental.assert_cardinality(len(self.features)))

            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class SocialIQaProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))

        path = os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl" if data_dir.split('/')[1].startswith("socialiqa") else "ROCStories_trn.jsonl")
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        path = os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl")
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        path = os.path.join(data_dir, "socialiqa.jsonl" if data_dir.endswith("-test") else "socialIQa_v1.4_tst.jsonl" )
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            question_id = "%s-%s" % (set_type, idx)
            context = item["context"]
            question = item["question"]
            endings = [item["answerA"],item["answerB"],item["answerC"] ]
            label = item["correct"] if "correct" in item else None 

            examples.append(
                InputExample(
                    example_id=question_id,
                    question=question,
                    contexts=[context,context,context],
                    endings=[endings[0], endings[1], endings[2]],
                    label=label,
                ) 
            )
        return examples

class SocialIQaCatgProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialiqa.jsonl" if data_dir.endswith("-test") else "socialIQa_v1.4_tst.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        #writer = open(f"data/socialiqa_q2rel/socialIQa_v1.4_{set_type}.jsonl", "w", encoding="utf-8")
        examples = []
        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            question_id = "%s-%s" % (set_type, idx)
            context = item["context"]
            question = item["question"]
            endings = [item["answerA"],item["answerB"],item["answerC"] ]
            label = item["correct"] if "correct" in item else None
            category = item["category"]

            examples.append(
                InputExample(
                    example_id=question_id,
                    question=question,
                    contexts=[context+" "+category if category else context for _ in range(3)],
                    endings=[endings[0], endings[1], endings[2]],#, options[3]
                    label=label,
                )
            )

            #item["q2rel"] = rel_type
            #writer.write(json.dumps(item)+"\n")
        #writer.close()
        return examples



class SocialIQaQ2RelCatgProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialiqa.jsonl" if data_dir.endswith("-test") else "socialIQa_v1.4_tst.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        #writer = open(f"data/socialiqa_q2rel/socialIQa_v1.4_{set_type}.jsonl", "w", encoding="utf-8")
        examples = []
        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            question_id = "%s-%s" % (set_type, idx)
            context = item["context"]
            question = item["question"]
            endings = [item["answerA"],item["answerB"],item["answerC"] ]
            label = item["correct"] if "correct" in item else None
            category = item["category"]
            rel_type = self.question_type_mapping(context, question)

            examples.append(
                InputExample(
                    example_id=question_id,
                    question=question+' '+rel_type if rel_type else question,
                    contexts=[context+" "+category if category else context for _ in range(3)],
                    endings=[endings[0], endings[1], endings[2]],#, options[3]
                    label=label,
                )
            )

            #item["q2rel"] = rel_type
            #writer.write(json.dumps(item)+"\n")
        #writer.close()
        return examples

    def question_type_mapping(self, context_text, question_text):
        #"oEffect","oReact","oWant","xAttr","xEffect","xIntent","xNeed","xReact","xWant"
        import nltk
        from nltk.corpus import stopwords
        try:
            stop_words = stopwords.words('english')
        except:
            nltk.download('stopwords')
            stop_words = stopwords.words('english')

        import spacy
        spacy_nlp = spacy.load("en_core_web_sm")

        rel_type = None
        context_text = context_text[0].upper() + context_text[1:]
        question_text_lower = question_text.lower()
        if "describe" in question_text_lower or "think of" in question_text_lower:
            return "[xAttr]"
        elif "need to do before" in question_text_lower:
            return "[xNeed]"
        elif "why" in question_text_lower:
            return "[xIntent]"
        elif "feel" in question_text_lower:
            rel_type = 'React'
        elif "want" in question_text_lower or "do next" in question_text_lower:
            rel_type = 'Want'
        elif "happen" in question_text_lower or "experiencing" in question_text_lower:
            rel_type = 'Effect'
        
        if rel_type and "other" in question_text_lower:
            return "[o"+rel_type+"]"
        if rel_type and "'s" in question_text_lower and "what's" not in question_text_lower:
            #print(context_text, question_text)
            return "[o"+rel_type+"]"

        doc = spacy_nlp(context_text)
        sub_toks = [str(tok) for tok in doc if (tok.dep_.startswith("nsubj")) and str(tok)[0].isupper() and str(tok).lower() not in stop_words ]
        #person_toks = [ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"]
        #obj_toks = set([tok.lower() for tok in doc if (tok.dep_ in ["dobj", "pobj"])])
        question_toks = set(nltk.word_tokenize(question_text))
        #doc_question = spacy_nlp(question_text)
        #question_toks = [str(tok) for tok in doc_question if tok.dep_.startswith("nsubj") and str(tok).lower() not in stop_words]
        
        if rel_type and sub_toks and question_toks and question_toks & set([sub_toks[0]]):
            return "[x"+rel_type+"]"
        elif rel_type and sub_toks:
            #print("OTHER", context_text, question_text)
            return "[o"+rel_type+"]"
        #if rel_type:
        #    print(context_text, question_text)

        return "[x"+rel_type+"] " + "[o"+rel_type+"]" if rel_type else None



class SocialIQaQ2RelProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialiqa.jsonl" if data_dir.endswith("-test") else "socialIQa_v1.4_tst.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        #writer = open(f"data/socialiqa_q2rel/socialIQa_v1.4_{set_type}.jsonl", "w", encoding="utf-8")
        examples = []
        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            question_id = "%s-%s" % (set_type, idx)
            context = item["context"]
            question = item["question"]
            endings = [item["answerA"],item["answerB"],item["answerC"] ]
            label = item["correct"] if "correct" in item else None
            rel_type = self.question_type_mapping(context, question)

            examples.append(
                InputExample(
                    example_id=question_id,
                    question=question+' '+rel_type if rel_type else question,
                    contexts=[context,context,context],
                    endings=[endings[0], endings[1], endings[2]],#, options[3]
                    label=label,
                )
            )

            #item["q2rel"] = rel_type
            #writer.write(json.dumps(item)+"\n")
        #writer.close()
        return examples

    def question_type_mapping(self, context_text, question_text):
        #"oEffect","oReact","oWant","xAttr","xEffect","xIntent","xNeed","xReact","xWant"

        rel_type = None
        context_text = context_text[0].upper() + context_text[1:]
        question_text_lower = question_text.lower()
        if "describe" in question_text_lower or "think of" in question_text_lower:
            return "[xAttr]"
        elif "need to do before" in question_text_lower:
            return "[xNeed]"
        elif "why" in question_text_lower:
            return "[xIntent]"
        elif "feel" in question_text_lower:
            rel_type = 'React'
        elif "want" in question_text_lower or "do next" in question_text_lower:
            rel_type = 'Want'
        elif "happen" in question_text_lower or "experiencing" in question_text_lower:
            rel_type = 'Effect'
        
        if rel_type and "other" in question_text_lower:
            return "[o"+rel_type+"]"
        if rel_type and "'s" in question_text_lower and "what's" not in question_text_lower:
            #print(context_text, question_text)
            return "[o"+rel_type+"]"

        doc = spacy_nlp(context_text)
        sub_toks = [str(tok) for tok in doc if (tok.dep_.startswith("nsubj")) and str(tok)[0].isupper() and str(tok).lower() not in stop_words ]
        #person_toks = [ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"]
        #obj_toks = set([tok.lower() for tok in doc if (tok.dep_ in ["dobj", "pobj"])])
        question_toks = set(nltk.word_tokenize(question_text))
        #doc_question = spacy_nlp(question_text)
        #question_toks = [str(tok) for tok in doc_question if tok.dep_.startswith("nsubj") and str(tok).lower() not in stop_words]
        
        if rel_type and sub_toks and question_toks and question_toks & set([sub_toks[0]]):
            return "[x"+rel_type+"]"
        elif rel_type and sub_toks:
            #print("OTHER", context_text, question_text)
            return "[o"+rel_type+"]"
        #if rel_type:
        #    print(context_text, question_text)

        return "[x"+rel_type+"] " + "[o"+rel_type+"]" if rel_type else None


class SocialIQaCatgRandProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialiqa.jsonl" if data_dir.endswith("-test") else "socialIQa_v1.4_tst.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        #writer = open(f"data/socialiqa_q2rel/socialIQa_v1.4_{set_type}.jsonl", "w", encoding="utf-8")
        examples = []
        category_list = ["[event]", "[feeling]", "[knowledge]", "[interaction]"]
        random.seed(0)
        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            question_id = "%s-%s" % (set_type, idx)
            context = item["context"]
            question = item["question"]
            endings = [item["answerA"],item["answerB"],item["answerC"] ]
            label = item["correct"] if "correct" in item else None
            category = item["category"]

            examples.append(
                InputExample(
                    example_id=question_id,
                    question=question,
                    contexts=[context+" "+random.choice(category_list) for _ in range(3)],
                    endings=[endings[0], endings[1], endings[2]],#, options[3]
                    label=label,
                )
            )

            #item["q2rel"] = rel_type
            #writer.write(json.dumps(item)+"\n")
        #writer.close()
        return examples

class SocialIQaQ2RelRandProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        newpath = os.path.join(data_dir, "socialiqa.jsonl" if data_dir.endswith("-test") else "socialIQa_v1.4_tst.jsonl")
        with open(newpath, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        #writer = open(f"data/socialiqa_q2rel/socialIQa_v1.4_{set_type}.jsonl", "w", encoding="utf-8")
        examples = []
        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            question_id = "%s-%s" % (set_type, idx)
            context = item["context"]
            question = item["question"]
            endings = [item["answerA"],item["answerB"],item["answerC"] ]
            label = item["correct"] if "correct" in item else None
            #rel_type = self.question_type_mapping(context, question)
            rel_list = ["[oEffect]","[oReact]","[oWant]","[xAttr]","[xEffect]","[xIntent]","[xNeed]","[xReact]","[xWant]"]

            examples.append(
                InputExample(
                    example_id=question_id,
                    question=question+' '+random.choice(rel_list),
                    contexts=[context,context,context],
                    endings=[endings[0], endings[1], endings[2]],#, options[3]
                    label=label,
                )
            )

            #item["q2rel"] = rel_type
            #writer.write(json.dumps(item)+"\n")
        #writer.close()
        return examples



class ROCStoriesProcessor(DataProcessor):
    """Processor for the ROCStories data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        path = os.path.join(data_dir, "cloze_test_val__spring2016 - cloze_test_ALL_val.csv")
        with open(path, 'r', encoding='utf-8') as f:
            data = csv.reader(f)
            data_list = [item for item in data][1:]
        path = os.path.join(data_dir, "cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv")
        with open(path, 'r', encoding='utf-8') as f:
            data = csv.reader(f)
            data_list.extend([item for item in data][1:1500])
        
        return self._create_examples(data_list, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        path = os.path.join(data_dir, "cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv")
        with open(path, 'r', encoding='utf-8') as f:
            data = csv.reader(f)
            data_list = [item for item in data][1500:]
        
        return self._create_examples(data_list, "train")
        
    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, line in enumerate(lines):
            #item = json.loads(line.strip())
            question_id = "%s-%s" % (set_type, line[0])
            context = " ".join(line[1:5])
            question = ""
            endings = [line[5],line[6]]
            label = line[7]
          
            examples.append(
                InputExample(
                    example_id=question_id,
                    question=question,
                    contexts=[context,context],
                    endings=[endings[0], endings[1]],
                    label=label,
                )
            )
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = label_map[example.label] if example.label else None

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features

processors = {
    "socialiqa": SocialIQaProcessor,
    "socialiqa_q2rel": SocialIQaQ2RelProcessor,
    "rocstories_cloze": ROCStoriesProcessor,
    "socialiqa_category": SocialIQaCatgProcessor,
    "socialiqa_q2rel_category": SocialIQaQ2RelCatgProcessor,
    "socialiqa_category_rand": SocialIQaCatgRandProcessor,
    "socialiqa_q2rel_rand": SocialIQaQ2RelRandProcessor,
    }
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"socialiqa", 3, "socialiqa_q2rel", 3, "rocstories_cloze", 2, }
