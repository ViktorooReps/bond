import json
import logging
import os
from enum import Enum
from functools import partial
from pathlib import Path
from random import shuffle
from typing import Dict, List, Tuple, Iterable, Callable, Iterator, Any, Optional

import torch
from torch import LongTensor, BoolTensor, FloatTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from bond.utils import extract_entities, merge_entity_lists, convert_entities_to_labels

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


# (token ids, token mask, labels, gold label mask, weight (optional))
Example = Tuple[LongTensor, BoolTensor, LongTensor, BoolTensor, Optional[float]]

# (tok ids, tok mask, attention mask, labels, label mask, gold label mask, weight)
BertExample = Tuple[LongTensor, BoolTensor, BoolTensor, LongTensor, BoolTensor, BoolTensor, FloatTensor]


def bert_collate_fn(batch: Iterable[Example], pad_token: int, pad_label: int, default_weight: float) -> BertExample:
    token_ids = []
    token_masks = []
    attention_masks = []
    labels = []
    label_masks = []
    gold_label_masks = []
    sent_weights = []

    for t_ids, t_mask, ls, gl_mask, weight in batch:
        token_ids.append(t_ids)
        token_masks.append(t_mask)
        attention_masks.append(torch.ones(len(t_ids)))
        labels.append(ls)
        label_masks.append(torch.ones(len(ls)))
        gold_label_masks.append(gl_mask)
        sent_weights.append(weight if weight is not None else default_weight)

    return pad_sequence(token_ids, batch_first=True, padding_value=pad_token).long(), \
        pad_sequence(token_masks, batch_first=True, padding_value=0).bool(), \
        pad_sequence(attention_masks, batch_first=True, padding_value=0).bool(), \
        pad_sequence(labels, batch_first=True, padding_value=pad_label).long(), \
        pad_sequence(label_masks, batch_first=True, padding_value=0).bool(), \
        pad_sequence(gold_label_masks, batch_first=True, padding_value=0).bool(), \
        torch.tensor(sent_weights).float()


class SubTokenDataset(Dataset):

    def __init__(self, examples: Iterable[Example], token_pad: int, label_pad: int = PAD_LABEL_ID, default_weight: float = DEFAULT_WEIGHT):
        self._examples = tuple(examples)
        self._token_pad = token_pad
        self._label_pad = label_pad
        self._default_weight = default_weight

    @property
    def collate_fn(self) -> Callable:
        return partial(bert_collate_fn, pad_token=self._token_pad, pad_label=self._label_pad, default_weight=self._default_weight)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]


def extract_ids_and_masks(json_dataset: Iterable[List[Dict[str, Any]]],
                          tokenizer: PreTrainedTokenizer,
                          max_seq_length: int) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Optional[float]]]:
    """Creates generator that outputs (token_ids, token_mask, labels, weight) from json dataset"""

    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token

    for document in tqdm(json_dataset, leave=False):

        def fetch_context(s_idx: int, l_context_size: int, r_context_size: int,
                          *, fixed: bool = False) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
            l_context, r_context = [], []

            def tokenize_sent(sentence: Iterable[str]) -> Iterator[str]:
                for w in sentence:
                    yield from tokenizer(w)

            # check for document boundaries

            if s_idx == 0 and s_idx == len(document) - 1:
                return (), ()

            if s_idx == 0:
                if not fixed:
                    r_context_size += l_context_size
                l_context_size = 0

            if s_idx == len(document) - 1:
                if not fixed:
                    l_context_size += r_context_size
                r_context_size = 0

            # carve needed context

            if l_context_size > 1:
                prev_sent = document[sent_idx - 1]['str_words']
                tokenized_sent = list(tokenize_sent(prev_sent))
                actual_context_size = l_context_size - 1  # 1 for sep token
                if len(tokenized_sent) >= actual_context_size:
                    l_context = tokenized_sent[-actual_context_size:] + [sep_token]
                else:
                    next_context_size = actual_context_size - len(tokenized_sent)
                    next_context, _ = fetch_context(s_idx - 1, next_context_size, 0, fixed=True)
                    l_context = list(next_context) + tokenized_sent + [sep_token]

            if r_context_size > 1:
                next_sent = document[sent_idx + 1]['str_words']
                tokenized_sent = list(tokenize_sent(next_sent))
                actual_context_size = r_context_size - 1  # 1 for sep token
                if len(tokenized_sent) >= actual_context_size:
                    r_context = [sep_token] + tokenized_sent[:actual_context_size]
                else:
                    next_context_size = actual_context_size - len(tokenized_sent)
                    next_context, _ = fetch_context(s_idx + 1, 0, next_context_size, fixed=True)
                    r_context = [sep_token] + tokenized_sent + list(next_context)

            return tuple(l_context), tuple(r_context)

        desired_seq_len = max_seq_length - 2  # for cls and sep tokens

        for sent_idx, sent in enumerate(document):
            words = sent['str_words']
            labels = sent['tags']
            weight = sent.get('weight', None)

            tokens = []
            token_mask = []
            for word, label in zip(words, labels):
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens:
                    continue

                tokens.extend(word_tokens)
                token_mask.extend([True] + [False] * (len(word_tokens) - 1))

            if len(tokens) > desired_seq_len:
                logging.warning('Extra long sentence detected!')
                # skipping as cutting may lead to broken entities
                continue

            if len(tokens) < desired_seq_len:
                added_context = desired_seq_len - len(tokens)
                left_context_len = (added_context // 2) + (added_context % 2)
                right_context_len = added_context // 2

                left_context, right_context = fetch_context(sent_idx, left_context_len, right_context_len)
                tokens = list(left_context) + tokens + list(right_context)
                token_mask = [False] * len(left_context) + token_mask + [False] * len(right_context)

            tokens = [cls_token] + tokens + [sep_token]
            token_mask = [False] + token_mask + [False]

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

            token_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)

            yield tuple(token_ids), tuple(token_mask), tuple(labels), weight


def load_transformed_dataset(dataset_name: DatasetName, add_gold: float, tokenizer: PreTrainedTokenizer, tokenizer_name: str,
                             max_seq_length: int) -> SubTokenDataset:

    cached_dataset_name = '_'.join([dataset_name.value, 'merged', str(add_gold), tokenizer_name, f'seq{max_seq_length}'])
    cached_dataset_file = Path(os.path.join('cache', 'datasets', cached_dataset_name))

    logging.info(f'Fetching transformed {dataset_name.value} with {add_gold} gold entities...')

    if cached_dataset_file.exists():
        logging.info(f'Found cached version {cached_dataset_file}!')

        dataset: SubTokenDataset = torch.load(cached_dataset_file)
    else:
        logging.info(f'Cached version wasn\'t found, creating {cached_dataset_file}...')

        gold_dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, DatasetType.TRAIN.value + '.json'))
        distant_dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, DatasetType.DISTANT.value + '.json'))

        if not gold_dataset_file.exists():
            raise ValueError(f'{gold_dataset_file} does not exists!')

        if not distant_dataset_file.exists():
            raise ValueError(f'{distant_dataset_file} does not exists!')

        with open(gold_dataset_file) as f:
            gold_json_dataset = json.load(f)

        with open(distant_dataset_file) as f:
            distant_json_dataset = json.load(f)

        tags_dict = load_tags_dict(dataset_name)

        def create_entity_mask(entities: Iterable[Entity], final_len: int) -> Tuple[bool]:
            mask = [False] * final_len
            for entity_start, _ in entities:
                mask[entity_start] = True
            return tuple(mask)

        # list of (token ids, token_mask, label ids, gold label mask)
        examples: List[Example] = []
        extractor = partial(extract_ids_and_masks, tokenizer=tokenizer, max_seq_length=max_seq_length)
        iterator = zip(extractor(gold_json_dataset), extractor(distant_json_dataset))

        all_token_ids = []
        all_token_masks = []
        all_weights = []
        # tuples of sentence idx and entity
        all_gold_entities: List[Tuple[int, Entity]] = []
        all_distant_entities: List[Tuple[int, Entity]] = []
        for sent_idx, ((token_ids, token_mask, gold_labels, weight), (_, _, distant_labels, _)) in enumerate(iterator):
            all_token_ids.append(token_ids)
            all_token_masks.append(token_mask)
            all_weights.append(weight)

            gold_entities = list(extract_entities(gold_labels, tags_dict))
            distant_entities = list(extract_entities(distant_labels, tags_dict))
            all_gold_entities.extend(zip(range(len(gold_entities)), gold_entities))
            all_distant_entities.extend(zip(range(len(distant_entities)), distant_entities))

            # TODO
        for (token_ids, token_mask, gold_labels, weight), (_, _, distant_labels, _) in iterator:
            assert len(gold_labels) == len(distant_labels)
            original_len = len(gold_labels)

            gold_entities = list(extract_entities(gold_labels, tags_dict))
            distant_entities = list(extract_entities(distant_labels, tags_dict))
            shuffle(gold_entities)

            added_entities_count = int(len(gold_entities) * add_gold)
            entities_to_add = gold_entities[:added_entities_count]
            gold_entities_mask = create_entity_mask(entities_to_add, original_len)

            merged_entities = merge_entity_lists(high_priority_entities=entities_to_add, low_priority_entities=distant_entities)
            labels = convert_entities_to_labels(merged_entities, no_entity_label=tags_dict['O'], vector_len=original_len)

            examples.append((LongTensor(token_ids), BoolTensor(token_mask), LongTensor(labels), BoolTensor(gold_entities_mask), weight))

        # convert to tensors and build dataset
        dataset = SubTokenDataset(examples, token_pad=tokenizer.pad_token_id)

        # cache built dataset
        torch.save(dataset, cached_dataset_file)

    logging.info('Dataset loaded!')

    return dataset


def load_dataset(dataset_name: DatasetName, dataset_type: DatasetType, tokenizer: PreTrainedTokenizer, tokenizer_name: str,
                 max_seq_length: int) -> SubTokenDataset:

    cached_dataset_name = '_'.join([dataset_name.value, *dataset_type.value.split('/'), tokenizer_name, f'seq{max_seq_length}'])
    cached_dataset_file = Path(os.path.join('cache', 'datasets', cached_dataset_name))

    logging.info(f'Fetching {dataset_name.value} of type {dataset_type.value}...')

    if cached_dataset_file.exists():
        logging.info(f'Found cached version {cached_dataset_file}!')

        dataset: SubTokenDataset = torch.load(cached_dataset_file)
    else:
        logging.info(f'Cached version wasn\'t found, creating {cached_dataset_file}...')

        dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, dataset_type.value + '.json'))
        if not dataset_file.exists():
            raise ValueError(f'{dataset_file} does not exists!')

        with open(dataset_file) as f:
            json_dataset = json.load(f)

        # token ids, token_mask, label ids
        examples: List[Example] = []
        for token_ids, token_mask, labels, weight in extract_ids_and_masks(json_dataset, tokenizer, max_seq_length):
            gold_label_mask = (False,) * len(labels)  # this loader does not add any gold labels
            examples.append((LongTensor(token_ids), BoolTensor(token_mask), LongTensor(labels), BoolTensor(gold_label_mask), weight))

        # convert to tensors and build dataset
        dataset = SubTokenDataset(examples, token_pad=tokenizer.pad_token_id)

        # cache built dataset
        torch.save(dataset, cached_dataset_file)

    logging.info('Dataset loaded!')

    return dataset
