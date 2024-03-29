import argparse
import logging
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import List

import torch
from torch import softmax
from torch.utils.data import SequentialSampler, BatchSampler
from tqdm import tqdm
from transformers import PreTrainedModel

from bond.data.batching import BatchedExamples
from bond.data.dataset import DatasetName, DatasetType, load_tags_dict, load_dataset, SubTokenDataset, get_based_dataset_name
from bond.data.example import Example
from bond.trainer import train, TrainingFramework, prepare_dataset
from bond.utils import set_seed
from train import setup_logging, MODEL_CLASSES, get_model, create_parser

from sklearn.model_selection import KFold


def eval_base_distributions(args: argparse.Namespace, model: PreTrainedModel, dataset: SubTokenDataset) -> None:
    eval_sampler = SequentialSampler(dataset)
    batch_size = args.batch_size * 2
    batch_sampler = BatchSampler(eval_sampler, batch_size=batch_size, drop_last=False)

    model.eval()
    for batch_idxes in tqdm(batch_sampler, desc=f"Creating base distribution", leave=False):
        batch_idxes: List[int]
        batch: BatchedExamples = dataset.collate_fn(dataset[batch_idxes])
        batch = batch.without_labels()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(batch)[0]

        batch_predicted_distributions = softmax(logits, dim=-1)

        new_batch_examples: List[Example] = []
        for padded_distributions, padding_mask, example_idx in zip(batch_predicted_distributions, batch.label_padding_mask, batch_idxes):
            unpadded_distributions = padded_distributions[padding_mask]
            example = dataset[example_idx].with_changes(label_distributions=unpadded_distributions)
            new_batch_examples.append(example)

        dataset[batch_idxes] = new_batch_examples


def main(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    tb_writer = setup_logging(args)

    dataset_name = DatasetName(args.dataset)
    dataset_type = DatasetType(args.dataset_type)

    if args.head_learning_rate is None:
        args.head_learning_rate = args.learning_rate
    if args.bert_learning_rate is None:
        args.bert_learning_rate = args.learning_rate

    args.base_distributions_file = None
    if args.base_model is not None:
        based_dataset_name = get_based_dataset_name(args, dataset_name, args.base_model)
        args.base_distributions_file = Path(os.path.join('cache', 'datasets', based_dataset_name))

    logging.info('Arguments: ' + str(args))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = 1

    set_seed(args.seed)
    num_labels = len(load_tags_dict(dataset_name).keys())

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name)

    splitter = KFold(n_splits=args.k_folds)

    train_dataset = prepare_dataset(args, dataset_name, dataset_type, tokenizer)
    eval_dataset = load_dataset(dataset_name, DatasetType.VALID, tokenizer, args.model_name, args.max_seq_length)

    based_dataset = deepcopy(train_dataset)

    for train_index, eval_index in splitter.split(train_dataset):
        train_fold = train_dataset.sub_dataset(train_index)

        # train model
        model = get_model(args, model_class, config_class, num_labels)
        model = train(args, model, dataset_name, train_fold, eval_dataset, TrainingFramework(args.framework), tb_writer)

        eval_fold = train_dataset.sub_dataset(eval_index)

        # get distributions
        eval_base_distributions(args, model, eval_fold)
        based_dataset[eval_index] = eval_fold.examples

    model_name = args.framework
    if args.use_coregulation:
        model_name += '_coregularized'

    based_dataset_name = get_based_dataset_name(args, dataset_name, model_name)

    based_dataset_file = Path(os.path.join('cache', 'datasets', based_dataset_name))

    logging.info(f'Saving based dataset to {based_dataset_file}')

    with open(based_dataset_file, 'wb') as f:
        pickle.dump(based_dataset, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(create_parser())
