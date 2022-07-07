import json
import logging
import os
import pickle
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from random import shuffle
from typing import Dict, List, Tuple, Iterable, Callable, Iterator, Any, Optional

import torch
from torch import LongTensor, BoolTensor, FloatTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

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
    RELABELLED = 'distant/relabelled'


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
                    yield from tokenizer.tokenize(w)

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
                # skipping as cutting may lead to broken entities
                raise ValueError('Extra long sentence detected!')

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


def get_dataset_entities(
        dataset_name: DatasetName,
        dataset_type: DatasetType,
        tokenizer: PreTrainedTokenizer,
        max_seq_length,
        *,
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
    sent_idx = None  # if there are no sentences then enumerate will not initialize sent_idx
    for sent_idx, (_, _, labels, _) in enumerate(extract_ids_and_masks(json_dataset, tokenizer, max_seq_length=max_seq_length)):
        entities = list(extract_entities(labels, tags_dict))
        all_entities.extend((sent_idx, entity) for entity in entities)

    total_sentences = 0 if sent_idx is None else sent_idx + 1

    shuffle(all_entities)
    keep_entities_count = int(len(all_entities) * fraction)
    kept_entities = all_entities[:keep_entities_count]
    all_kept_entities = [[] for _ in range(total_sentences)]  # create empty list of kept entities for each sentence
    for sentenced_entity in kept_entities:
        sent_idx, entity = sentenced_entity
        all_kept_entities[sent_idx].append(entity)

    # cache entities
    with open(cached_entities_file_path, 'wb') as f:
        pickle.dump(all_kept_entities, f, pickle.HIGHEST_PROTOCOL)

    return all_kept_entities


def load_transformed_dataset(
        dataset_name: DatasetName,
        add_gold: float,
        tokenizer: PreTrainedTokenizer,
        tokenizer_name: str,
        max_seq_length: int,
        *,
        merge_relabelled: bool = False,
        add_distant: bool = False
) -> SubTokenDataset:

    extra_qualifiers = []
    if merge_relabelled:
        extra_qualifiers.append('relabelled')
    if add_distant:
        extra_qualifiers.append('distant')

    cached_dataset_name = '_'.join([dataset_name.value, 'merged', str(add_gold), *extra_qualifiers, tokenizer_name, f'seq{max_seq_length}']) + '.pt'

    cached_dataset_file = Path(os.path.join('cache', 'datasets', cached_dataset_name))

    logging.info(f'Fetching transformed {dataset_name.value} with {add_gold} gold entities'
                 f'{" and distant entities" if add_distant else ""}'
                 f'{" and relabelled entities" if merge_relabelled else ""}...')

    if cached_dataset_file.exists():
        logging.info(f'Found cached version {cached_dataset_file}!')

        dataset: SubTokenDataset = torch.load(cached_dataset_file)
    else:
        logging.info(f'Cached version wasn\'t found, creating {cached_dataset_file}...')

        distant_dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, DatasetType.DISTANT.value + '.json'))

        if not distant_dataset_file.exists():
            raise ValueError(f'{distant_dataset_file} does not exist!')

        with open(distant_dataset_file) as f:
            distant_json_dataset = json.load(f)

        tags_dict = load_tags_dict(dataset_name)
        extractor = partial(extract_ids_and_masks, tokenizer=tokenizer, max_seq_length=max_seq_length)

        def create_entity_mask(entities: Iterable[Entity], vector_len: int) -> Tuple[bool]:
            mask = [False] * vector_len
            for entity_start, entity_labels in entities:
                for label_idx in range(len(entity_labels)):
                    mask[entity_start + label_idx] = True
            return tuple(mask)

        # list of (token ids, token_mask, label ids, gold label mask)
        examples: List[Example] = []

        all_token_ids = []
        all_token_masks = []
        all_weights = []
        all_distant_entities = []
        all_original_lengths = []
        for token_ids, token_mask, distant_labels, weight in extractor(distant_json_dataset):
            all_token_ids.append(token_ids)
            all_token_masks.append(token_mask)
            all_weights.append(weight)
            all_original_lengths.append(len(distant_labels))

            distant_entities = list(extract_entities(distant_labels, tags_dict)) if add_distant else []
            all_distant_entities.append(distant_entities)

        all_relabelled_entities = [[] for _ in range(len(all_token_ids))]  # create empty list of relabelled entities for each sentence
        if merge_relabelled:
            relabelled_dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, DatasetType.RELABELLED.value + '.json'))
            if not relabelled_dataset_file.exists():
                raise ValueError(f'{relabelled_dataset_file} does not exist!')

            with open(relabelled_dataset_file) as f:
                relabelled_json_dataset = json.load(f)

            for sent_idx, (_, _, relabelled_labels, _) in enumerate(extractor(relabelled_json_dataset)):
                relabelled_entities = list(extract_entities(relabelled_labels, tags_dict))
                all_relabelled_entities[sent_idx].extend(relabelled_entities)

        all_added_entities = get_dataset_entities(dataset_name, DatasetType.TRAIN, tokenizer, max_seq_length, fraction=add_gold)

        iterator = zip(
            all_token_ids,
            all_token_masks,
            all_weights,
            all_added_entities,
            all_distant_entities,
            all_relabelled_entities,
            all_original_lengths
        )
        for token_ids, token_mask, weight, gold_entities, distant_entities, relabelled_entities, orig_len in iterator:
            gold_entities_mask = create_entity_mask(gold_entities, vector_len=orig_len)

            merged_entities = merge_entity_lists(high_priority_entities=relabelled_entities, low_priority_entities=distant_entities)
            merged_entities = merge_entity_lists(high_priority_entities=gold_entities, low_priority_entities=merged_entities)
            labels = convert_entities_to_labels(merged_entities, no_entity_label=tags_dict['O'], vector_len=orig_len)

            examples.append((LongTensor(token_ids), BoolTensor(token_mask), LongTensor(labels), BoolTensor(gold_entities_mask), weight))

        # convert to tensors and build dataset
        dataset = SubTokenDataset(examples, token_pad=tokenizer.pad_token_id)

        # cache built dataset
        torch.save(dataset, cached_dataset_file)

    logging.info('Dataset loaded!')

    return dataset


def load_dataset(
        dataset_name: DatasetName,
        dataset_type: DatasetType,
        tokenizer: PreTrainedTokenizer,
        tokenizer_name: str,
        max_seq_length: int
) -> SubTokenDataset:

    cached_dataset_name = '_'.join([dataset_name.value, *dataset_type.value.split('/'), tokenizer_name, f'seq{max_seq_length}']) + '.pt'
    cached_dataset_file = Path(os.path.join('cache', 'datasets', cached_dataset_name))

    logging.info(f'Fetching {dataset_name.value} of type {dataset_type.value}...')

    if cached_dataset_file.exists():
        logging.info(f'Found cached version {cached_dataset_file}!')

        dataset: SubTokenDataset = torch.load(cached_dataset_file)
    else:
        logging.info(f'Cached version wasn\'t found, creating {cached_dataset_file}...')

        dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, dataset_type.value + '.json'))
        if not dataset_file.exists():
            raise ValueError(f'{dataset_file} does not exist!')

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


def relabel_dataset(args, model: PreTrainedModel, dataset_name: DatasetName, tokenizer: PreTrainedTokenizer) -> None:
    dataset = load_dataset(dataset_name, DatasetType.DISTANT, tokenizer, args.model_name, args.max_seq_length)
    sampler = SequentialSampler(dataset)
    batch_size = args.batch_size * 2
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn)

    original_dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, DatasetType.DISTANT.value + '.json'))
    if not original_dataset_file.exists():
        raise ValueError(f'{original_dataset_file} does not exist!')

    with open(original_dataset_file) as f:
        json_dataset = json.load(f)

    relabelled_json_dataset: List[List[dict]] = deepcopy(json_dataset)
    doc_lengths = list(map(len, relabelled_json_dataset))
    doc_idx = 0
    example_idx = 0

    tags_dict = load_tags_dict(dataset_name)

    model.eval()
    for batch in tqdm(dataloader, desc=f"Relabelling dataset", leave=False, total=len(dataloader)):
        batch: BertExample = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            batch_token_ids, batch_token_mask, batch_attention_mask, batch_labels, batch_label_mask, batch_gold_label_mask, batch_weight = batch
            inputs = {"input_ids": batch_token_ids, "token_mask": batch_token_mask, "attention_mask": batch_attention_mask,
                      "labels": batch_labels, 'label_mask': batch_label_mask, 'seq_weights': batch_weight, "gold_label_mask": batch_gold_label_mask}
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)

            _, logits = outputs[:2]

        batch_predicted_labels = torch.argmax(logits, dim=-1)

        for predicted_labels, label_mask in zip(batch_predicted_labels, batch_label_mask):
            predicted_labels = predicted_labels[label_mask].tolist()

            # clean up loose I- annotations
            predicted_entities = tuple(extract_entities(predicted_labels, tags_dict))
            predicted_labels = convert_entities_to_labels(predicted_entities, tags_dict['O'], len(predicted_labels))

            relabelled_json_dataset[doc_idx][example_idx]['tags'] = predicted_labels
            example_idx += 1
            if example_idx >= doc_lengths[doc_idx]:
                example_idx = 0
                doc_idx += 1

    relabelled_dataset_file = Path(os.path.join('dataset', 'data', dataset_name.value, DatasetType.RELABELLED.value + '.json'))
    with open(relabelled_dataset_file, 'w') as f:
        json.dump(relabelled_json_dataset, f)
