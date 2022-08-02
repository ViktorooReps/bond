from argparse import ArgumentParser

from tqdm import tqdm
from transformers import RobertaTokenizer

from bond.data.dataset import DatasetName, DatasetType, get_dataset_entities
from bond.utils import set_seed

if __name__ == '__main__':
    parser = ArgumentParser(description='Create cached entities for reproducibility.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default=DatasetName.CONLL2003.value, choices=[choice.value for choice in DatasetName])
    parser.add_argument('--entities_range', type=int, nargs='+', default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])

    args = parser.parse_args()

    set_seed(args.seed)
    dataset = DatasetName(args.dataset_name)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    for entities_fraction in tqdm(args.entities_range):
        get_dataset_entities(dataset_name=dataset, dataset_type=DatasetType.TRAIN, tokenizer=tokenizer, fraction=entities_fraction)
