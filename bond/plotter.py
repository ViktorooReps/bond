import os.path

import torch

from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from bond.data.dataset import load_dataset
from bond.utils import ner_scores


def plot_distant_dataset_stats(dataset_name: DatasetName) -> None:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # for loading datasets - doesn't really matter what tokenizer to use

    distant_dataset = load_dataset(dataset_name, DatasetType.DISTANT, tokenizer, 'roberta-base', 128)
    gold_dataset = load_dataset(dataset_name, DatasetType.TRAIN, tokenizer, 'roberta-base', 128)

    distant_labels = []
    for _, labels, mask, _ in DataLoader(distant_dataset, batch_size=1):
        distant_labels.extend(labels.masked_select(mask > 0).tolist())

    gold_labels = []
    for _, labels, mask, _ in DataLoader(gold_dataset, batch_size=1):
        gold_labels.extend(labels.masked_select(mask > 0).tolist())

    stats = ner_scores(gold_labels, distant_labels, load_tags_dict(dataset_name))
    print(stats)  # TODO: do actual stats visuallization


def score_cached_dataset(dataset_path: str) -> None:
    cached_name = os.path.basename(dataset_path)
    info = cached_name.split('_')
    tokenizer = RobertaTokenizer.from_pretrained(info[-2])
    dataset_name = DatasetName(info[0])
    max_seq_len = int(info[-1][3:])

    distant_dataset: SubTokenDataset = torch.load(dataset_path)
    gold_dataset = load_dataset(dataset_name, DatasetType.TRAIN, tokenizer, 'roberta-base', max_seq_len)

    distant_labels = []
    for _, _, _, labels, mask, _, _ in DataLoader(distant_dataset, batch_size=1, collate_fn=collate_fn):
        distant_labels.extend(labels.masked_select(mask).tolist())

    gold_labels = []
    for _, _, _, labels, mask, _, _ in DataLoader(gold_dataset, batch_size=1, collate_fn=collate_fn):
        gold_labels.extend(labels.masked_select(mask).tolist())

    assert len(gold_labels) == len(distant_labels)
    stats = ner_scores(gold_labels, distant_labels, load_tags_dict(dataset_name))
    print(stats)
