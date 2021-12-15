import argparse
import logging
import os
import warnings
from pathlib import Path
from time import localtime, strftime

import torch
from transformers import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig, RobertaTokenizer

from bond.data import DatasetName, DatasetType, load_tags_dict
from bond.model import JunctionStrategy, RobertaCRFForTokenClassification, RobertaForTokenClassificationOriginal
from bond.trainer import evaluate, train
from bond.utils import Scores, set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

ALL_MODELS = sum(
    (
        tuple(conf_map.keys())
        for conf_map in (ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,)
    ),
    (),
)

MODEL_CLASSES = {
    "roberta-original": (RobertaConfig, RobertaForTokenClassificationOriginal, RobertaTokenizer),
    "roberta-crf": (RobertaConfig, RobertaCRFForTokenClassification, RobertaTokenizer)
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required parameters
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="One of " + ', '.join(dataset_name.value for dataset_name in DatasetName))
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument('--experiment_name', default=None, type=str, required=True,
                        help='Name of the experiment')

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    # General training parameters
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--steps_per_epoch", type=int, default=-1,
                        help="Number of optimization steps per epoch.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=-1,
                        help='Number of batches in each step.')
    parser.add_argument("--bert_learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for RoBERTa.")
    parser.add_argument('--head_learning_rate', default=1e-3, type=float,
                        help='The initial learning rate for model\' head: LSTM-CRF or CRF')
    parser.add_argument("--lr_decrease", default=1.0, type=float,
                        help="LR decrease with layer depth")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float,
                        help="BETA1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float,
                        help="BETA2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm for gradient clipping.")
    parser.add_argument('--junction_strategy', default='none', type=str,
                        help='One of ' + ', '.join(junction.value for junction in JunctionStrategy))
    parser.add_argument('--add_lstm', action='store_true',
                        help='Add LSTM layer between BERT and CRF.')
    parser.add_argument('--lstm_hidden_size', default=128, type=int,
                        help='Size of LSTM layer.')
    parser.add_argument('--lstm_num_layers', default=2, type=int,
                        help='Number of LSTM layers')

    # NER training parameters
    parser.add_argument("--ner_fit_epochs", default=1, type=int,
                        help="number of epochs for NER fitting stage")
    parser.add_argument('--ner_fit_steps', default=-1, type=int,
                        help='Number of NER fitting steps. -1 for using epochs')
    parser.add_argument("--warmup_steps", default=200, type=int,
                        help="Linear warmup over warmup_steps.")

    # Self-training parameters
    parser.add_argument('--self_training_epochs', type=int, default=50,
                        help='number of epochs for self training stage')
    parser.add_argument('--label_keep_threshold', type=float, default=0.9,
                        help='Label keeping threshold for self training stage')
    parser.add_argument("--lr_st_decay", default=1.0, type=float,
                        help="Learning rate decay between stages for self training stage")
    parser.add_argument('--correct_frequency', action='store_true',
                        help='Do soft label frequency correction before choosing labels with threshold')
    parser.add_argument('--use_kldiv_loss', action='store_true',
                        help='Use KLDivLoss during self training')
    parser.add_argument('--period', default=-1, type=int,
                        help='Period for updating teacher model. -1 for once per epoch')

    return parser


def main(parser: argparse.ArgumentParser) -> Scores:
    args = parser.parse_args()
    run_name = strftime("%Y-%m-%d_%Hh%Mm%Ss", localtime())

    warnings.simplefilter("ignore", UserWarning)

    tb_dir = Path(os.path.join('tfboard', args.experiment_name))
    log_dir = Path(os.path.join('logs', args.experiment_name))

    log_name = run_name + '.log'

    # Create output directory if needed
    if not tb_dir.exists():
        os.makedirs(tb_dir)
    if not log_dir.exists():
        os.makedirs(log_dir)

    logging.basicConfig(format='[%(asctime)s] %(message)s', filename=os.path.join(log_dir, log_name), level=logging.INFO)
    tb_writer = SummaryWriter(os.path.join(tb_dir, run_name))

    logging.info('Arguments: ' + str(args))

    dataset = DatasetName(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = 1

    set_seed(args.seed)
    num_labels = len(load_tags_dict(dataset).keys())

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name, num_labels=num_labels)  # TODO: do caching
    tokenizer = tokenizer_class.from_pretrained(args.model_name)  # TODO: do caching

    # Training
    model = model_class.from_pretrained(args.model_name, config=config, junction_strategy=JunctionStrategy(args.junction_strategy),
                                        add_lstm=args.add_lstm, lstm_hidden=args.lstm_hidden_size, lstm_layers=args.lstm_num_layers)
    model, global_step, tr_loss = train(args, model, dataset, tokenizer, tb_writer)

    # TODO: save model

    # Evaluation
    results = evaluate(args, model, dataset, DatasetType.TEST, tokenizer)
    logging.info('Results on test: ' + str(results))

    results = evaluate(args, model, dataset, DatasetType.VALID, tokenizer)
    logging.info('Results on valid: ' + str(results))

    return results


if __name__ == '__main__':
    main(create_parser())
