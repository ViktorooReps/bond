from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from bond.data import DatasetName, DatasetType, load_dataset, load_tags_dict
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
