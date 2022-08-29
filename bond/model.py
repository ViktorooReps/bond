from enum import Enum
from time import sleep
from typing import Iterable, Optional, Tuple, Callable

import torch
from torch import LongTensor, Tensor, BoolTensor
from torch.nn.functional import softmax
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from bond.crf import MarginalCRF
from bond.data.batching import BatchedExamples
from bond.utils import apply_mask, extract_subwords, convert_hard_to_soft_labels

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class JunctionStrategy(Enum):
    NONE = 'none'
    IGNORE_WITH_MASK_BEFORE_CRF = 'mask_ignore_before_crf'
    IGNORE_WITH_MASK_BEFORE_LSTM = 'mask_ignore_before_lstm'


class JunctionError(Exception):
    pass


class PoolingStrategy(Enum):
    LAST = 'last'
    SUM = 'sum'
    CONCAT = 'concat'


def sum_last_4_layers(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Sums the last 4 hidden representations of a sequence output of BERT.
    Args:
    -----
    sequence_output: Tuple of tensors of shape (batch, seq_length, hidden_size).
        For BERT base, the Tuple has length 13.
    Returns:
    --------
    summed_layers: Tensor of shape (batch, seq_length, hidden_size)
    """
    last_layers = sequence_outputs[-4:]
    return torch.stack(last_layers, dim=0).sum(dim=0)


def get_last_layer(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Returns the last tensor of a list of tensors."""
    return sequence_outputs[-1]


def concat_last_4_layers(sequence_outputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """Concatenate the last 4 tensors of a tuple of tensors."""
    last_layers = sequence_outputs[-4:]
    return torch.cat(last_layers, dim=-1)


POOLERS = {
    PoolingStrategy.SUM: sum_last_4_layers,
    PoolingStrategy.LAST: get_last_layer,
    PoolingStrategy.CONCAT: concat_last_4_layers,
}


class BERTHead(nn.Module):

    def __init__(
            self,
            num_labels: int,
            hidden_size: int,
            dropout_prob: float,
            subword_repr_size: int = 0,
            add_lstm: bool = False,
            lstm_layers: int = 2,
            lstm_hidden: int = 128,
            lstm_dropout: float = 0.5,
            add_crf: bool = False,
            no_entity_weight: float = 1.0
    ):
        super().__init__()

        self.subword_repr_size = subword_repr_size

        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)

        if self.subword_repr_size > 0:
            self.cnn = nn.Conv2d(in_channels=1, out_channels=subword_repr_size, kernel_size=(3, hidden_size), padding=(1, 0))
        else:
            self.cnn = None

        if add_lstm:
            self.lstm = nn.LSTM(input_size=hidden_size + self.subword_repr_size, hidden_size=lstm_hidden, num_layers=lstm_layers,
                                bidirectional=True, dropout=lstm_dropout, batch_first=True)
        else:
            self.lstm = None

        total_hidden_size = 2 * lstm_hidden if add_lstm else hidden_size + self.subword_repr_size
        self.hidden2labels = nn.Linear(total_hidden_size, num_labels)

        nn.init.zeros_(self.hidden2labels.bias)
        self.hidden2labels.bias.data[0] = 6  # make model biased towards 'O' label

        if add_crf:
            self.crf = MarginalCRF(num_labels)
        else:
            self.crf = None

        self.no_entity_weight = no_entity_weight

    def forward(
            self,
            seq_repr: Tensor,
            seq_lens: Iterable[int],
            labeled_tokens_mask: BoolTensor,
            labels: Optional[LongTensor] = None,
            label_mask: Optional[BoolTensor] = None,
            self_training: bool = False,
            use_kldiv_loss: bool = False
    ):
        """Returns (loss), marginal tag distribution, label mask

        loss is returned only when labels are given
        label_mask is returned because it might change"""
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        seq_repr = seq_repr.to(device)
        labeled_tokens_mask = labeled_tokens_mask.to(device)

        seq_lens = tuple(seq_lens)

        if self.subword_repr_size > 0:
            # extract subwords for each entity
            subwords_repr = extract_subwords(seq_repr, seq_lens, labeled_tokens_mask)
            batch_size, seq_len, subword_count, num_features = subwords_repr.shape

            pooler = nn.MaxPool2d(kernel_size=(subword_count, 1))
            subwords_repr = self.cnn(subwords_repr.view(-1, 1, subword_count, num_features))
            subwords_repr = pooler(subwords_repr).view(batch_size, seq_len, self.subword_repr_size)

            # get first subword of each word
            seq_repr, new_padding_mask = apply_mask(seq_repr, labeled_tokens_mask)
            assert seq_repr.shape == (batch_size, seq_len, num_features)

            # concatenate representations
            seq_repr = torch.cat([seq_repr, subwords_repr], dim=2)
        else:
            # get first subword of each word
            seq_repr, new_padding_mask = apply_mask(seq_repr, labeled_tokens_mask)

        if self.lstm is not None:
            seq_repr = self.dropout(seq_repr)
            seq_repr, _ = self.lstm(seq_repr)

        seq_repr = self.dropout(seq_repr)
        label_scores = self.hidden2labels(seq_repr)

        if self.crf is not None:
            label_probs = self.crf.marginal_probabilities(label_scores).transpose(0, 1)
        else:
            label_probs = softmax(label_scores, dim=-1)
        log_probs = torch.log(label_probs)

        outputs = (label_probs,)

        if labels is not None:
            labels = labels.to(device)
            label_mask = label_mask.to(device)

            if labels.shape != label_probs.shape:
                # convert hard labels into one-hots
                labels = convert_hard_to_soft_labels(labels, self.num_labels)

            raveled_gold_labels = labels.contiguous().view(-1, self.num_labels)
            raveled_mask = label_mask.contiguous().view(-1)

            not_entities = (torch.argmax(raveled_gold_labels, dim=-1) == 0)
            label_weights = torch.ones_like(raveled_mask, dtype=torch.float)
            label_weights[not_entities] = self.no_entity_weight if not self_training else 1.0

            if use_kldiv_loss:
                raveled_log_probs = log_probs.contiguous().view(-1, self.num_labels)
                tok_wise_loss: Tensor = KLDivLoss(reduction='batchmean')(raveled_log_probs[raveled_mask], raveled_gold_labels[raveled_mask])
                loss = torch.mean(tok_wise_loss * label_weights[raveled_mask])
            else:
                if self.crf is not None:
                    batch_loss = self.crf.forward(label_scores, marginal_tags=labels, mask=label_mask, reduction='none')
                    loss = batch_loss.sum() / label_mask.float().sum()
                else:
                    raveled_label_scores = label_scores.contiguous().view(-1, self.num_labels)
                    tok_wise_loss: Tensor = CrossEntropyLoss(reduction='none')(raveled_label_scores[raveled_mask],
                                                                               raveled_gold_labels[raveled_mask])
                    loss = torch.mean(tok_wise_loss * label_weights[raveled_mask])

            outputs = (loss,) + outputs

        return outputs  # (loss), label probs


class RobertaWithHead(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaCRFForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(
            self,
            config: RobertaConfig,
            freeze_bert: bool = False,
            pooler: PoolingStrategy = PoolingStrategy.LAST,
            subword_repr_size: int = 0,
            add_lstm: bool = False,
            lstm_layers: int = 2,
            lstm_hidden: int = 128,
            lstm_dropout: float = 0.5,
            head_dropout: float = 0.5,
            add_crf: bool = False,
            no_entity_weight: float = 1.0
    ):

        super().__init__(config)
        self.num_labels = config.num_labels

        hidden_size = config.hidden_size if pooler != PoolingStrategy.CONCAT else 4 * config.hidden_size

        self.roberta = RobertaModel(config)
        self.frozen_bert = freeze_bert
        self.pooler = pooler
        self.head = BERTHead(
            num_labels=self.num_labels,
            hidden_size=hidden_size,
            dropout_prob=head_dropout,
            subword_repr_size=subword_repr_size,
            add_lstm=add_lstm,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            add_crf=add_crf,
            no_entity_weight=no_entity_weight
        )

        if self.frozen_bert:
            self.freeze_bert()

        self.init_weights()

    def freeze_bert(self):
        """Freeze all BERT parameters. Only the classifier weights will be
        updated."""
        for p in self.roberta.parameters():
            p.requires_grad = False

    def prepare_for_self_training(self):
        pass

    @property
    def returns_probs(self) -> bool:
        return self.head.returns_probs

    def forward(self, examples: BatchedExamples, self_training: bool = False, use_kldiv_loss: bool = False, **argv):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        attention_mask = examples.token_padding_mask.to(device)
        seq_lens = [mask.sum() for mask in attention_mask]

        input_ids = examples.token_ids.to(device)

        outputs: BaseModelOutputWithPoolingAndCrossAttentions

        if self.frozen_bert:
            with torch.no_grad():
                self.roberta.eval()
                outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # outputs: final_embedding, pooler_output, (hidden_states), (attentions)

        final_embedding = outputs.last_hidden_state
        all_layers_sequence_outputs = outputs.hidden_states
        pooled_embedding = POOLERS[self.pooler](all_layers_sequence_outputs)
        head_outputs = self.head(
            pooled_embedding,
            seq_lens=seq_lens,
            labeled_tokens_mask=examples.labeled_token_mask,
            labels=examples.label_distributions,
            label_mask=examples.label_mask,
            self_training=self_training,
            use_kldiv_loss=use_kldiv_loss
        )  # (loss), scores

        outputs = head_outputs + (final_embedding,) + outputs[2:]

        return outputs  # (loss), scores, final_embedding, (hidden_states), (attentions)


class CoregulatedModel(nn.Module):

    def __init__(self, num_labels: int, model_generator: Callable[[], nn.Module], n_models: int = 4, agreement_strength: float = 5.0):
        super().__init__()
        self._num_labels = num_labels
        self._models = nn.ModuleList([model_generator() for _ in range(n_models)])
        self._agreement_strength = agreement_strength
        self._main_model_idx = 0

    def prepare_for_self_training(self):
        self._models = nn.ModuleList([self._models[self._main_model_idx]])
        self._main_model_idx = 0
        torch.cuda.empty_cache()
        sleep(10)

    def forward(self, examples: BatchedExamples, *args, warmup: bool = False, self_training: bool = False, **kwargs):
        if examples.label_ids is None:
            return self._models[self._main_model_idx](examples)

        if self_training:
            return self._models[self._main_model_idx](*args, **kwargs, examples=examples, self_training=self_training)

        # do not compute kld loss for gold labels
        compute_kld_loss = ~examples.gold_entities_mask & examples.label_mask

        n_models = len(self._models)

        # outputs are tuples of (loss, predicted label probs)
        outputs = [model(*args, **kwargs, examples=examples)[:2] for model in self._models]
        models_loss = sum(loss for loss, _ in outputs) / n_models
        if not warmup:
            models_avg_probs = sum(probs for _, probs in outputs) / n_models

            raveled_models_avg_probs = models_avg_probs.contiguous().view(-1, self._num_labels)
            raveled_mask = compute_kld_loss.contiguous().view(-1)
            masked_avg_probs = raveled_models_avg_probs[raveled_mask]

            raveled_probs = (probs.contiguous().view(-1, self._num_labels) for _, probs in outputs)

            kld_loss = KLDivLoss(reduction='mean')
            agreement_loss = sum(kld_loss(torch.log(probs[raveled_mask]), masked_avg_probs) for probs in raveled_probs) / n_models
        else:
            agreement_loss = 0

        loss = models_loss + self._agreement_strength * agreement_loss
        _, pred_labels = outputs[self._main_model_idx]

        return loss, pred_labels
