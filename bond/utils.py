import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import LongTensor, Tensor, BoolTensor, FloatTensor
from torch.ao.sparsity import BaseScheduler
from torch.nn import Softmax
from torch.nn.functional import pad, one_hot
from torch.optim import Optimizer
from transformers import AdamW, PreTrainedModel, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from bond.data import Entity

Scores = Dict[str, float]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_entities(labels: Iterable[int], tags_dict: Dict[str, int]) -> Iterable[Entity]:
    """Extracts entities as tuples: (start_idx, (label0, label1, ..., labelN))"""
    labels_dict = {label: tag for tag, label in tags_dict.items()}
    entity_start = {tag for tag in tags_dict if tag.startswith('B')}
    no_entity = {'O'}

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


def merge_entity_lists(high_priority_entities: Iterable[Entity], low_priority_entities: Iterable[Entity]) -> Tuple[Entity]:
    hp_entities = set(high_priority_entities)
    lp_entities = set(low_priority_entities)

    coarse_merged_entities = list(hp_entities.union(lp_entities))
    sorted_coarse_entities = sorted(coarse_merged_entities, key=lambda t: t[0])

    unwanted_entities = set()

    def check_collision(entity_idx: int) -> bool:
        def collides(ent1: Entity, ent2: Entity) -> bool:
            """[ent_start, ent_end)"""
            ent1_start_pos, ent1_labels = ent1
            ent1_end_pos = ent1_start_pos + len(ent1_labels)

            ent2_start_pos, _ = ent2
            return ent1_end_pos <= ent2_start_pos

        if entity_idx > 0 and collides(sorted_coarse_entities[entity_idx - 1], sorted_coarse_entities[entity_idx]):
            return True
        if entity_idx < len(sorted_coarse_entities) - 1 and collides(sorted_coarse_entities[entity_idx],
                                                                     sorted_coarse_entities[entity_idx + 1]):
            return True
        return False

    for ent_idx, ent in enumerate(sorted_coarse_entities):
        if ent in lp_entities and check_collision(ent_idx):
            unwanted_entities.add(ent)

    return tuple(filter(lambda e: e not in unwanted_entities, sorted_coarse_entities))


def convert_entities_to_labels(entities: Iterable[Entity], no_entity_label: int, vector_len: int) -> Tuple[int]:
    result = [no_entity_label] * vector_len

    def validate_entity(start_idx: int, end_idx: int) -> None:
        for idx in range(start_idx, end_idx):
            if result[idx] != no_entity_label:
                raise ValueError('Entity collision detected!')

    for entity in entities:
        entity_start, entity_labels = entity
        entity_end = entity_start + len(entity_labels)
        validate_entity(entity_start, entity_end)
        for label_idx, label in zip(range(entity_start, entity_end), entity_labels):
            result[label_idx] = label

    return tuple(result)


def ner_scores(gold_labels: Iterable[int], predicted_labels: Iterable[int], tags_dict: Dict[str, int]) -> Scores:
    gold_entities = set(extract_entities(gold_labels, tags_dict))
    predicted_entities = set(extract_entities(predicted_labels, tags_dict))

    total_predicted = len(predicted_entities)  # true_positive + false_positive
    total_entities = len(gold_entities)        # true_positive + false_negative

    true_positive = len(gold_entities.intersection(predicted_entities))
    precision = true_positive / total_predicted if total_predicted > 0 else 0
    recall = true_positive / total_entities if total_entities > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {'f1': f1, 'precision': precision, 'recall': recall}


def initialize_roberta(args, model: PreTrainedModel, total_steps: int,
                       warmup_steps: int, end_lr_proportion: float = 0) -> Tuple[PreTrainedModel, Optimizer, BaseScheduler]:
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

    total_steps = int(total_steps * (1 + end_lr_proportion / (1 - end_lr_proportion)))

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


def extract_subwords(seq_repr: Tensor, seq_lens: Iterable[int], token_mask: BoolTensor) -> Tensor:
    batch_size, seq_len, num_features = seq_repr.shape

    # extract subword indices
    word_start_indices: List[LongTensor] = [torch.arange(seq_len)[mask] for mask in token_mask]
    word_end_indices: List[LongTensor] = [torch.roll(ws_inds, -1).long() for ws_inds in word_start_indices]
    for we, s_len in zip(word_end_indices, seq_lens):
        we[-1] = s_len

    # pad subwords to equal length
    max_subword_count = max(max(batch_we - batch_ws - 1) for batch_ws, batch_we in zip(word_start_indices, word_end_indices))

    def get_subword_repr(seq: Tensor, word_start: int, word_end: int) -> Tensor:
        added_len = max_subword_count - (word_end - word_start - 1)
        return pad(seq[word_start + 1: word_end].view(-1, num_features), (0, 0, 0, added_len), value=0)

    subword_reprs: List[Tensor] = []
    for ws_inds, we_inds, batch_seq in zip(word_start_indices, word_end_indices, seq_repr):
        batch_subwords = [get_subword_repr(batch_seq, ws, we) for ws, we, in zip(ws_inds, we_inds)]
        subword_reprs.append(torch.stack(batch_subwords))

    # pad sequences to equal length
    max_len = max(len(seq) for seq in subword_reprs)
    add_lens = [max_len - len(seq) for seq in subword_reprs]

    def pad_seq(sequence: Tensor, added_len: int) -> Tensor:
        return pad(sequence, (0, 0, 0, 0, 0, added_len), value=0)

    padded_seqs = [pad_seq(seq, add_len) for seq, add_len in zip(subword_reprs, add_lens)]
    tokens_repr = torch.stack(padded_seqs)

    return tokens_repr


def convert_hard_to_soft_labels(labels, num_labels: int) -> FloatTensor:
    labels[labels < 0] = 0
    return one_hot(labels, num_labels)

