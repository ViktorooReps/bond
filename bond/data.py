import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch import LongTensor
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer

from bond.model import JunctionStrategy

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


def load_transformed_dataset(dataset_name: DatasetName, target_recall: float, target_precision: Union[str, float] = 'distant'):
    pass


def load_dataset(dataset_name: DatasetName, dataset_type: DatasetType, tokenizer: PreTrainedTokenizer, tokenizer_name: str,
                 max_seq_length: int, sep_token: str = 'SEP',
                 junction_strategy: JunctionStrategy = JunctionStrategy.IGNORE_WITH_MASK) -> TensorDataset:
    """Returns TensorDataset of tensors:
    [0]: token ids
    [1]: label ids
    [2]: label mask (0 - padded label, 1 - actual label)
    [3]: attention mask
    """

    cached_dataset_name = '_'.join([dataset_name.value, *dataset_type.value.split('/'), tokenizer_name, junction_strategy.value])
    cached_dataset_file = Path(os.path.join('cache', 'datasets', cached_dataset_name))
    if cached_dataset_file.exists():
        dataset: TensorDataset = torch.load(cached_dataset_file)
    else:
        dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, dataset_type.value + '.json'))
        if not dataset_file.exists():
            raise ValueError(f'{dataset_file} does not exists!')

        with open(dataset_file) as f:
            json_dataset = json.load(f)

        label_extensions = load_label_extensions(dataset_name)

        # token ids, label ids, label mask, attention mask
        examples: List[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = []
        for entry_idx, entry in enumerate(json_dataset):
            words = entry['str_words']
            labels = entry['tags']

            tokens = []
            label_ids = []
            for word, label in zip(words, labels):
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens:
                    continue

                tokens.extend(word_tokens)
                extension_label_id = PAD_LABEL_ID
                if junction_strategy == JunctionStrategy.FILL_WITH_I:
                    extension_label_id = label_extensions[label]

                label_ids.extend([label] + [extension_label_id] * (len(word_tokens) - 1))

            special_token_count = 2  # RoBERTa uses double [SEP] token
            if len(tokens) > max_seq_length - special_token_count:
                tokens = tokens[: (max_seq_length - special_token_count)]
                label_ids = label_ids[: (max_seq_length - special_token_count)]

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
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.

            tokens += [sep_token, sep_token]  # double [SEP] token for RoBERTa
            label_ids += [PAD_LABEL_ID, PAD_LABEL_ID]

            pad_token = tokenizer.pad_token
            padding_length = max_seq_length - len(tokens)
            tokens += [pad_token] * padding_length
            label_ids += [PAD_LABEL_ID] * padding_length

            label_mask = [0 if label_id == PAD_LABEL_ID else 1 for label_id in label_ids]   # used to calculate loss
            attention_mask = [0 if token == pad_token else 1 for token in tokens]           # used for attention layers

            token_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)

            examples.append((tuple(token_ids), tuple(label_ids), tuple(label_mask), tuple(attention_mask)))

        # group tuples
        grouped_token_ids: List[Tuple[int, ...]] = []
        grouped_label_ids: List[Tuple[int, ...]] = []
        grouped_label_masks: List[Tuple[int, ...]] = []
        grouped_attention_masks: List[Tuple[int, ...]] = []

        for t_ids, l_ids, l_mask, att_mask in examples:
            grouped_token_ids.append(t_ids)
            grouped_label_ids.append(l_ids)
            grouped_label_masks.append(l_mask)
            grouped_attention_masks.append(att_mask)

        # convert tp tensors and build dataset
        dataset = TensorDataset(LongTensor(grouped_token_ids), LongTensor(grouped_label_ids), LongTensor(grouped_label_masks),
                                LongTensor(grouped_attention_masks))

        # cache built dataset
        torch.save(dataset, cached_dataset_file)

    return dataset
