import json
import logging
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union, Iterable, Callable, Iterator, Any

import torch
from torch import LongTensor, BoolTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


PAD_LABEL_ID = -1


class DatasetType(Enum):
    DISTANT = 'distant/train'
    TRAIN = 'gold/train'
    VALID = 'gold/valid'
    TEST = 'gold/test'


class DatasetName(Enum):
    CONLL2003 = 'conll03'


def load_tags_dict(dataset_name: DatasetName) -> Dict[str, int]:
    tags_dict_file = Path(os.path.join('dataset', 'data', dataset_name.value, 'tag_to_id.json'))
    if not tags_dict_file.exists():
        raise ValueError(f'{tags_dict_file} does not exists!')

    with open(tags_dict_file) as f:
        tags_dict = json.load(f)

    return tags_dict


def load_label_extensions(dataset_name: DatasetName) -> Dict[int, int]:
    tags_dict = load_tags_dict(dataset_name)
    label_extensions = {}
    for tag, label in tags_dict.items():
        if tag == 'O':
            label_extensions[label] = label
        else:
            tag_type, tag_group = tag.split('-')
            if tag_type == 'B':
                extension_tag = 'I-' + tag_group
            elif tag_type == 'I':
                extension_tag = tag
            else:
                raise ValueError(f'Unknown tag type {tag_type}!')

            label_extensions[label] = tags_dict[extension_tag]

    return label_extensions


Example = Tuple[LongTensor, BoolTensor, LongTensor]  # (token ids, token mask, labels)
BertExample = Tuple[LongTensor, BoolTensor, BoolTensor, LongTensor, BoolTensor]  # (tok ids, tok mask, attention mask, labels, label mask)
# DistantExample = Tuple[...]  # like BertExample but with gold_mask? - shows gold annotations
# during training set to 1.0 prob of gold labels


def bert_collate_fn(batch: Iterable[Example], pad_token, pad_label) -> BertExample:
    token_ids = []
    token_masks = []
    attention_masks = []
    labels = []
    label_masks = []

    for t_ids, t_mask, ls in batch:
        token_ids.append(t_ids)
        token_masks.append(t_mask)
        attention_masks.append(torch.ones(len(t_ids)))
        labels.append(ls)
        label_masks.append(torch.ones(len(ls)))

    return pad_sequence(token_ids, batch_first=True, padding_value=pad_token).long(), \
        pad_sequence(token_masks, batch_first=True, padding_value=0).bool(), \
        pad_sequence(attention_masks, batch_first=True, padding_value=0).bool(), \
        pad_sequence(labels, batch_first=True, padding_value=pad_label).long(), \
        pad_sequence(label_masks, batch_first=True, padding_value=0).bool()


class SubTokenDataset(Dataset):  # need to somehow implement gold labels

    def __init__(self, examples: Iterable[Example], token_pad: int, label_pad: int = PAD_LABEL_ID):
        self._examples = tuple(examples)
        self._token_pad = token_pad
        self._label_pad = label_pad

    @property
    def collate_fn(self) -> Callable:
        return partial(bert_collate_fn, pad_token=self._token_pad, pad_label=self._label_pad)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]


# TODO: add document-level context
def extract_ids_and_masks(json_dataset: Iterable[Dict[str, Any]],
                          tokenizer: PreTrainedTokenizer,
                          max_seq_length: int,
                          sep_token: str) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]:
    """Creates generator that outputs (token_ids, token_mask, labels) from json dataset"""
    for entry_idx, entry in enumerate(json_dataset):
        words = entry['str_words']
        labels = entry['tags']

        tokens = []
        token_mask = []
        for word, label in zip(words, labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                continue

            tokens.extend(word_tokens)
            token_mask.extend([True] + [False] * (len(word_tokens) - 1))

        special_token_count = 2  # RoBERTa uses double [SEP] token
        if len(tokens) > max_seq_length - special_token_count:
            logging.warning('Extra long sentence detected!')
            continue
            # tokens = tokens[: (max_seq_length - special_token_count)]
            # label_ids = label_ids[: (max_seq_length - special_token_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as  the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens += [sep_token, sep_token]  # double [SEP] token for RoBERTa
        token_mask += [False, False]

        token_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)

        yield tuple(token_ids), tuple(token_mask), tuple(labels)


def load_transformed_dataset(dataset_name: DatasetName, add_gold: float, tokenizer: PreTrainedTokenizer, tokenizer_name: str,
                             max_seq_length: int, sep_token: str = 'SEP') -> SubTokenDataset:

    cached_dataset_name = '_'.join([dataset_name.value, 'merged', str(add_gold), tokenizer_name])
    cached_dataset_file = Path(os.path.join('cache', 'datasets', cached_dataset_name))

    if cached_dataset_file.exists():
        dataset: SubTokenDataset = torch.load(cached_dataset_file)
    else:
        gold_dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, DatasetType.TRAIN.value + '.json'))
        distant_dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, DatasetType.DISTANT.value + '.json'))

        if not gold_dataset_file.exists():
            raise ValueError(f'{gold_dataset_file} does not exists!')

        if not distant_dataset_file.exists():
            raise ValueError(f'{distant_dataset_file} does not exists!')

        with open(gold_dataset_file) as f:
            gold_json_dataset = json.load(f)

        with open(distant_dataset_file) as f:
            distant_dataset_file = json.load(f)

        tags_dict = load_tags_dict(dataset_name)

        # token ids, token_mask, label ids
        examples: List[Example] = []
        for token_ids, token_mask, labels in extract_ids_and_masks(json_dataset, tokenizer, max_seq_length, sep_token):
            examples.append((LongTensor(token_ids), BoolTensor(token_mask), LongTensor(labels)))

        # convert to tensors and build dataset
        dataset = SubTokenDataset(examples, token_pad=tokenizer.pad_token_id)

        # cache built dataset
        torch.save(dataset, cached_dataset_file)

    return dataset


def load_dataset(dataset_name: DatasetName, dataset_type: DatasetType, tokenizer: PreTrainedTokenizer, tokenizer_name: str,
                 max_seq_length: int, sep_token: str = 'SEP') -> SubTokenDataset:

    cached_dataset_name = '_'.join([dataset_name.value, *dataset_type.value.split('/'), tokenizer_name])
    cached_dataset_file = Path(os.path.join('cache', 'datasets', cached_dataset_name))

    if cached_dataset_file.exists():
        dataset: SubTokenDataset = torch.load(cached_dataset_file)
    else:
        dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, dataset_type.value + '.json'))
        if not dataset_file.exists():
            raise ValueError(f'{dataset_file} does not exists!')

        with open(dataset_file) as f:
            json_dataset = json.load(f)

        # token ids, token_mask, label ids
        examples: List[Example] = []
        for token_ids, token_mask, labels in extract_ids_and_masks(json_dataset, tokenizer, max_seq_length, sep_token):
            examples.append((LongTensor(token_ids), BoolTensor(token_mask), LongTensor(labels)))

        # convert to tensors and build dataset
        dataset = SubTokenDataset(examples, token_pad=tokenizer.pad_token_id)

        # cache built dataset
        torch.save(dataset, cached_dataset_file)

    return dataset
