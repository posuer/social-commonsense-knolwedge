
import csv
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum
from filelock import FileLock

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        pairID: (Optional) string. Unique identifier for the pair of sentences.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    pairID: Optional[str] = None

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
        pairID: (Optional) Unique identifier for the pair of sentences.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    pairID: Optional[int] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


import torch
from torch.utils.data.dataset import Dataset

class ClassificationDataset(Dataset):
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
                self.label_list = processor.get_labels()
                if mode == Split.dev:
                    examples = processor.get_dev_examples(data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)
                logger.info("Training examples: %s", len(examples))
                self.features = convert_examples_to_features(
                    examples,
                    self.label_list,
                    max_seq_length,
                    tokenizer,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
    
    def get_labels_list(self):
        return self.label_list




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


class SocialIQaClassProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))

        path = os.path.join(data_dir, "socialIQa_v1.4_trn.jsonl" if data_dir.endswith("category") else "trn.jsonl")
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        path = os.path.join(data_dir, "socialIQa_v1.4_dev.jsonl" if data_dir.endswith("category") else "dev.jsonl")
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        path = os.path.join(data_dir, "socialIQa_v1.4_tst.jsonl" if data_dir.endswith("category") else "tst.jsonl")
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["event", "interaction", "feeling", "knowledge"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        #dev_count = {key:0 for key in self.get_labels()}

        # dev_examples = []
        # train_examples = []
        examples = []

        for idx, line in enumerate(lines):
            item = json.loads(line.strip())
            question_id = "%s-%s" % (set_type, idx)
            context = item["context"]
            question = item["question"]
            endings = [item["answerA"],item["answerB"],item["answerC"] ]
            label = item["correct"]
            category = None
            if "category" in item:
                category = item["category"]
            examples.append(
                        InputExample(
                            guid=question_id,
                            text_a = context,
                            text_b = '\t'.join([question, endings[0]+".", endings[1]+".", endings[2]+"."]),
                            label = category if category else None
                        )
                    )

        return examples 



def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
):
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        return label_map[example.label]
     

    labels = [label_from_example(example) for example in examples]

    features = []

    for i, example in enumerate(examples):
        input = tokenizer(
                example.text_a, 
                example.text_b,
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            
        feature = InputFeatures(**input, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


processors = {
    "socialiqa_class": SocialIQaClassProcessor,
    }