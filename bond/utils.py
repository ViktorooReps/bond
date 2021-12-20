import random
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, BoolTensor
from torch.ao.sparsity import BaseScheduler
from torch.nn import Softmax
from torch.nn.functional import pad
from torch.optim import Optimizer
from transformers import AdamW, PreTrainedModel, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

Scores = Dict[str, float]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ner_scores(gold_labels: Iterable[int], predicted_labels: Iterable[int], tags_dict: Dict[str, int]) -> Scores:
    labels_dict = {label: tag for tag, label in tags_dict.items()}
    entity_start = {tag for tag in tags_dict if tag.startswith('B')}
    no_entity = {'O'}

    def extract_entities(labels: Iterable[int]) -> Iterable[Tuple[int, tuple]]:
        """Extracts entities as tuples: (start_idx, (label0, label1, ..., labelN))"""
        entity_labels = []
        entity_idx: Optional[int] = None

        for curr_idx, label in enumerate(labels):
            tag = labels_dict[label]
            if tag in entity_start:
                if entity_idx is not None:
                    yield entity_idx, tuple(entity_labels)
                entity_idx = curr_idx
                entity_labels = [label]
            elif tag in no_entity:
                if entity_idx is not None:
                    yield entity_idx, tuple(entity_labels)
                entity_idx = None
            else:
                entity_labels.append(label)

        if entity_idx is not None:
            yield entity_idx, tuple(entity_labels)

    gold_entities = set(extract_entities(gold_labels))
    predicted_entities = set(extract_entities(predicted_labels))

    total_predicted = len(predicted_entities)  # true_positive + false_positive
    total_entities = len(gold_entities)        # true_positive + false_negative

    true_positive = len(gold_entities.intersection(predicted_entities))
    precision = true_positive / total_predicted if total_predicted > 0 else 0
    recall = true_positive / total_entities if total_entities > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {'f1': f1, 'precision': precision, 'recall': recall}


def initialize_roberta(args, model: PreTrainedModel, total_steps: int,
                       warmup_steps: int) -> Tuple[PreTrainedModel, Optimizer, BaseScheduler]:
    """Only compatible with RoBERTa-base for now"""
    model.to(args.device)

    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = args.bert_learning_rate
    head_lr = args.head_learning_rate
    lr = init_lr
    decay = args.weight_decay

    # === Pooler and CRF ======================================================

    params_0 = [p for n, p in named_parameters if ("pooler" in n or "roberta" not in n)
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "roberta" not in n)
                and not any(nd in n for nd in no_decay)]

    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0, "name": "head_params_without_decay"}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": head_lr, "weight_decay": decay, "name": "head_params_with_decay"}
    opt_parameters.append(head_params)

    # === 12 Hidden layers ==========================================================
    for layer in range(11, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]

        if params_0:
            layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0, "name": f"layer{layer}_params_without_decay"}
            opt_parameters.append(layer_params)

        if params_1:
            layer_params = {"params": params_1, "lr": lr, "weight_decay": decay, "name": f"layer{layer}_params_with_decay"}
            opt_parameters.append(layer_params)

        lr *= args.lr_decrease

    # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]

    if params_0:
        embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0, "name": f"embeddings_without_decay"}
        opt_parameters.append(embed_params)

    if params_1:
        embed_params = {"params": params_1, "lr": lr, "weight_decay": decay, "name": f"embeddings_with_decay"}
        opt_parameters.append(embed_params)

    optimizer = AdamW(opt_parameters, eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    if args.use_linear_scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    model.zero_grad()
    return model, optimizer, scheduler


def soft_frequency(logits: Tensor, power: int = 2, probs: bool = False):
    """
    Unsupervised Deep Embedding for Clustering Analysis
    https://arxiv.org/abs/1511.06335
    """
    if not probs:
        softmax = Softmax(dim=1)
        y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
    else:
        y = logits
    f = torch.sum(y, dim=(0, 1))
    t = (y ** power) / f
    p = t / torch.sum(t, dim=2, keepdim=True)

    return p


def apply_mask(sub_tokens_repr: Tensor, mask: BoolTensor) -> Tuple[Tensor, BoolTensor]:
    device = sub_tokens_repr.device
    batch_size, seq_len, num_features = sub_tokens_repr.shape

    # apply mask to sub tokens
    masked_seqs = [seq[seq_mask] for seq, seq_mask in zip(sub_tokens_repr, mask)]
    assert len(masked_seqs) == batch_size

    # pad sequences to equal length
    max_len = max(len(seq) for seq in masked_seqs)
    add_lens = [max_len - len(seq) for seq in masked_seqs]

    def pad_seq(sequence: Tensor, added_len: int) -> Tensor:
        return pad(sequence, (0, 0, 0, added_len), value=0)

    padded_seqs = [pad_seq(seq, add_len) for seq, add_len in zip(masked_seqs, add_lens)]
    tokens_repr = torch.stack(padded_seqs)
    assert tokens_repr.shape == (batch_size, max_len, num_features)

    # create new mask based on padded sequence
    batch_size, seq_len, num_features = tokens_repr.shape

    def create_mask(padding: int) -> Tensor:
        new_mask = torch.ones(seq_len, device=device)
        if padding > 0:
            new_mask[-padding:] = 0

        return new_mask

    new_label_mask = torch.stack([create_mask(add_len) for add_len in add_lens])
    assert new_label_mask.shape == (batch_size, max_len)

    return tokens_repr, new_label_mask.bool()
