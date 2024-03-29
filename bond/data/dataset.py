import json
import logging
import os
import pickle
from argparse import Namespace
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from random import shuffle
from typing import Tuple, Dict, Iterable, Callable, List, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from bond.data.batching import collate_fn
from bond.data.example import Example, get_examples
from bond.utils import extract_entities, merge_entity_lists, convert_entities_to_labels, correct_label_distributions, \
    convert_hard_to_soft_labels

PAD_LABEL_ID = -1
DEFAULT_WEIGHT = 1.0
Entity = Tuple[int, Tuple[int, ...]]


class DatasetType(Enum):
    DISTANT = 'distant/train'
    WEIGHED = 'distant/weighed'
    NORMALIZED = 'distant/normalized'
    TRAIN = 'gold/train'
    VALID = 'gold/valid'
    TEST_CORRECTED = 'gold/test_corrected'
    TEST = 'gold/test'
    BASED = 'based/train'


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


class SubTokenDataset(Dataset):

    def __init__(self, examples: Iterable[Example], token_pad: int, label_pad: int = PAD_LABEL_ID):
        self._examples = np.array(tuple(examples))
        self._token_pad = token_pad
        self._label_pad = label_pad

    @property
    def collate_fn(self) -> Callable:
        return partial(collate_fn, pad_token_id=self._token_pad, pad_label_id=self._label_pad)

    @property
    def examples(self) -> np.ndarray:
        return self._examples

    def sub_dataset(self, indices: Union[List[int], List[bool], np.ndarray]) -> 'SubTokenDataset':
        return SubTokenDataset(deepcopy(self._examples[indices]), token_pad=self._token_pad, label_pad=self._label_pad)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: Union[int, List[int], List[bool], np.ndarray]) -> Example:
        return self._examples[idx]

    def __setitem__(self, idx: Union[int, List[int], List[bool], np.ndarray], values: Union[Example, List[Example], np.ndarray]):
        self._examples[idx] = values


def get_dataset_entities(
        dataset_name: DatasetName,
        dataset_type: DatasetType,
        tokenizer: PreTrainedTokenizer,
        *,
        max_seq_length: int = 512,
        fraction: float = 1.0
) -> List[List[Entity]]:

    cached_entities_file_name = '_'.join([dataset_name.value, dataset_type.value.replace('/', '_'), str(fraction)]) + '.pkl'
    cached_entities_file_path = Path(os.path.join('cache', 'entities', cached_entities_file_name))
    if cached_entities_file_path.exists():
        logging.info(f'Found cached version of entities {cached_entities_file_path}!')
        with open(cached_entities_file_path, 'rb') as f:
            return pickle.load(f)

    logging.info(f'Cached version wasn\'t found, creating {cached_entities_file_path}...')

    dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, dataset_type.value + '.json'))
    if not dataset_file.exists():
        raise ValueError(f'{dataset_file} does not exist!')

    with open(dataset_file) as f:
        json_dataset = json.load(f)

    tags_dict = load_tags_dict(dataset_name)

    all_entities: List[Tuple[int, Entity]] = []
    example_idx = None  # if there are no examples then enumerate will not initialize example_idx
    for example_idx, example in enumerate(get_examples(json_dataset, tokenizer, num_labels=len(tags_dict), max_seq_length=max_seq_length)):
        entities = list(extract_entities(example.label_ids.tolist(), tags_dict))
        all_entities.extend((example_idx, entity) for entity in entities)

    total_examples = 0 if example_idx is None else example_idx + 1

    shuffle(all_entities)
    keep_entities_count = int(len(all_entities) * fraction)
    kept_entities = all_entities[:keep_entities_count]
    all_kept_entities = [[] for _ in range(total_examples)]  # create empty list of kept entities for each sentence
    for sentenced_entity in kept_entities:
        example_idx, entity = sentenced_entity
        all_kept_entities[example_idx].append(entity)

    # cache entities
    with open(cached_entities_file_path, 'wb') as f:
        pickle.dump(all_kept_entities, f, pickle.HIGHEST_PROTOCOL)

    return all_kept_entities


def load_dataset(
        dataset_name: DatasetName,
        dataset_type: DatasetType,
        tokenizer: PreTrainedTokenizer,
        tokenizer_name: str,
        max_seq_length: int
) -> SubTokenDataset:

    cached_dataset_name = '_'.join([dataset_name.value, *dataset_type.value.split('/'), tokenizer_name, f'seq{max_seq_length}']) + '.pkl'
    cached_dataset_file = Path(os.path.join('cache', 'datasets', cached_dataset_name))

    logging.info(f'Fetching {dataset_name.value} of type {dataset_type.value}...')

    if cached_dataset_file.exists():
        logging.info(f'Found cached version {cached_dataset_file}!')

        with open(cached_dataset_file, 'rb') as f:
            dataset: SubTokenDataset = pickle.load(f)
    else:
        logging.info(f'Cached version wasn\'t found, creating {cached_dataset_file}...')

        dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, dataset_type.value + '.json'))
        if not dataset_file.exists():
            raise ValueError(f'{dataset_file} does not exist!')

        with open(dataset_file) as f:
            json_dataset = json.load(f)

        tags_dict = load_tags_dict(dataset_name)
        dataset_examples = get_examples(json_dataset, tokenizer, num_labels=len(tags_dict), max_seq_length=max_seq_length)
        dataset = SubTokenDataset(dataset_examples, token_pad=tokenizer.pad_token_id)

        # cache built dataset
        with open(cached_dataset_file, 'wb') as f:
            pickle.dump(dataset, f)

    logging.info('Dataset loaded!')

    return dataset


def get_transformed_dataset_name(
        dataset_name: DatasetName,
        add_gold: float,
        tokenizer_name: str,
        *,
        max_seq_length: int = 512,
        base_distributions_file: Optional[Path] = None,
        add_distant: bool = False
) -> str:

    extra_qualifiers = []
    if base_distributions_file is not None:
        extra_qualifiers.append('based__' + base_distributions_file.with_suffix('').name + '__')
    if add_distant:
        extra_qualifiers.append('distant')

    return '_'.join([
        dataset_name.value,
        'merged',
        str(add_gold),
        *extra_qualifiers,
        tokenizer_name,
        f'seq{max_seq_length}'
    ])


def get_based_dataset_name(args: Namespace, dataset_name: DatasetName, model_name: str):
    trained_dataset_name = get_transformed_dataset_name(
        dataset_name, args.add_gold_labels, args.model_name,
        max_seq_length=args.max_seq_length,
        base_distributions_file=args.base_distributions_file,
        add_distant=args.add_distant
    )

    return model_name + '_on_' + trained_dataset_name + f'_{args.k_folds}folds.pkl'


def load_transformed_dataset(
        dataset_name: DatasetName,
        add_gold: float,
        tokenizer: PreTrainedTokenizer,
        tokenizer_name: str,
        *,
        max_seq_length: int = 512,
        base_distributions_file: Optional[Path] = None,
        add_distant: bool = False
) -> SubTokenDataset:

    cached_dataset_name = get_transformed_dataset_name(
        dataset_name, add_gold, tokenizer_name,
        max_seq_length=max_seq_length,
        base_distributions_file=base_distributions_file,
        add_distant=add_distant
    ) + '.pkl'

    cached_dataset_file = Path(os.path.join('cache', 'datasets', cached_dataset_name))

    logging.info(f'Fetching transformed {dataset_name.value} with {add_gold} gold entities'
                 f'{" and distant entities" if add_distant else ""}'
                 f'{" and base distribution from " + str(base_distributions_file) if base_distributions_file is not None else ""}...')

    if cached_dataset_file.exists():
        logging.info(f'Found cached version {cached_dataset_file}!')

        with open(cached_dataset_file, 'rb') as f:
            dataset: SubTokenDataset = pickle.load(f)
    else:
        logging.info(f'Cached version wasn\'t found, creating {cached_dataset_file}...')

        distant_dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, DatasetType.DISTANT.value + '.json'))

        if not distant_dataset_file.exists():
            raise ValueError(f'{distant_dataset_file} does not exist!')

        with open(distant_dataset_file) as f:
            distant_json_dataset = json.load(f)

        tags_dict = load_tags_dict(dataset_name)
        extractor = partial(get_examples, tokenizer=tokenizer, max_seq_length=max_seq_length, num_labels=len(tags_dict))

        def create_entity_mask(entities: Iterable[Entity], vector_len: int) -> Tuple[bool]:
            mask = [False] * vector_len
            for entity_start, entity_labels in entities:
                for label_idx in range(len(entity_labels)):
                    mask[entity_start + label_idx] = True
            return tuple(mask)

        # list of (token ids, token_mask, label ids, gold label mask)
        examples: List[Example] = []

        all_distant_entities = []
        for example in extractor(distant_json_dataset):
            examples.append(example)

            distant_entities = list(extract_entities(example.label_ids.tolist(), tags_dict)) if add_distant else []
            all_distant_entities.append(distant_entities)

        if base_distributions_file is not None:
            if not base_distributions_file.exists():
                raise ValueError(f'{base_distributions_file} does not exist!')

            with open(base_distributions_file, 'rb') as f:
                base_distributions_dataset: SubTokenDataset = pickle.load(f)

            based_examples: List[Example] = []

            for based_example, old_example in zip(base_distributions_dataset.examples, examples):
                based_example: Example
                based_examples.append(old_example.with_changes(label_distributions=based_example.label_distributions))

            examples = based_examples

        all_gold_entities = get_dataset_entities(
            dataset_name=dataset_name,
            dataset_type=DatasetType.TRAIN,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            fraction=add_gold
        )

        new_examples: List[Example] = []
        for gold_entities, distant_entities, example in zip(all_gold_entities, all_distant_entities, examples):
            orig_len = len(example.label_ids)
            gold_entities_mask = create_entity_mask(gold_entities, vector_len=orig_len)
            torch_gold_entities_mask = torch.tensor(gold_entities_mask, dtype=torch.bool).bool()

            merged_entities = merge_entity_lists(high_priority_entities=gold_entities, low_priority_entities=distant_entities)
            label_ids = convert_entities_to_labels(merged_entities, no_entity_label=tags_dict['O'], vector_len=orig_len)
            torch_label_ids = torch.tensor(label_ids, dtype=torch.long).long()

            if base_distributions_file is not None:
                torch_label_distributions = correct_label_distributions(gold_entities, example.label_distributions, tags_dict)
            else:
                torch_label_distributions = convert_hard_to_soft_labels(torch_label_ids, len(tags_dict))

            new_examples.append(example.with_changes(
                label_ids=torch_label_ids,
                label_distributions=torch_label_distributions,
                gold_entities_mask=torch_gold_entities_mask
            ))

        dataset = SubTokenDataset(new_examples, token_pad=tokenizer.pad_token_id)

        # cache built dataset
        with open(cached_dataset_file, 'wb') as f:
            pickle.dump(dataset, f)

    logging.info('Dataset loaded!')

    return dataset
