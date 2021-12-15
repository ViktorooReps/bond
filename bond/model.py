from enum import Enum
from typing import Optional, Tuple

import torch
from torch import FloatTensor, LongTensor, Tensor
from torch.nn.functional import one_hot
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from bond.crf import MarginalCRF

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaForTokenClassificationOriginal(BertPreTrainedModel):
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
        model = RobertaForTokenClassificationOriginal.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, **__):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @property
    def returns_probs(self) -> bool:
        return False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        self_training: bool = False,
        use_kldiv_loss: bool = False
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        final_embedding = outputs[0]
        sequence_output = self.dropout(final_embedding)
        logits = self.classifier(sequence_output)

        outputs = (logits, label_mask, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:

            # Only keep active parts of the loss
            if attention_mask is not None or label_mask is not None:
                active_loss = True
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                if label_mask is not None:
                    active_loss = active_loss & (label_mask.view(-1) == 1)
                active_logits = logits.view(-1, self.num_labels)[active_loss]

            if self_training:
                if use_kldiv_loss:
                    loss_fct = KLDivLoss()
                else:
                    loss_fct = CrossEntropyLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1, self.num_labels)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), scores, label_mask, final_embedding, (hidden_states), (attentions)


def convert_hard_to_soft_labels(labels, num_labels: int) -> FloatTensor:
    labels[labels < 0] = 0
    return one_hot(labels, num_labels)


class JunctionStrategy(Enum):
    IGNORE_WITH_MASK = 'mask_ignore'
    FILL_WITH_I = 'labels_fill'
    IGNORE_WITH_MASK_BEFORE_CRF = 'mask_ignore_before_crf'
    IGNORE_WITH_MASK_BEFORE_LSTM = 'mask_ignore_before_lstm'
    TOKEN_WISE_AVERAGE_BEFORE_CRF = 'average_before_crf'
    TOKEN_WISE_AVERAGE_BEFORE_LSTM = 'average_before_lstm'


class JunctionError(Exception):
    pass


class CRFForBERT(nn.Module):
    """BERT-aware MarginalCRF implementation"""

    def __init__(self, num_labels: int, hidden_size: int, dropout_prob: float, junction_strategy: JunctionStrategy, add_lstm: bool = False,
                 lstm_layers: int = 2, lstm_hidden: int = 128):
        super().__init__()

        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        if add_lstm:
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden, num_layers=lstm_layers, bidirectional=True,
                                dropout=dropout_prob, batch_first=True)
        else:
            self.lstm = None
        self.hidden2labels = nn.Linear(hidden_size if not add_lstm else 2 * lstm_hidden, num_labels)
        self.crf = MarginalCRF(num_labels)
        self.strategy = junction_strategy

    def forward(self, seq_repr: Tensor, labels: Optional[LongTensor] = None, label_mask: Optional[Tensor] = None,
                self_training: bool = False, use_kldiv_loss: bool = False):
        """Returns (loss), marginal tag distribution, label mask

        loss is returned only when labels are given
        label_mask is returned because it might change"""

        def apply_mask(sub_tokens_repr: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
            device = sub_tokens_repr.device
            batch_size, seq_len, num_features = sub_tokens_repr.shape

            # apply mask to sub tokens
            masked_seqs = [seq[seq_mask > 0] for seq, seq_mask in zip(sub_tokens_repr, mask)]
            assert len(masked_seqs) == batch_size

            # pad sequences to equal length
            max_len = max(len(seq) for seq in masked_seqs)
            add_lens = [max_len - len(seq) for seq in masked_seqs]
            pad_tensor = torch.zeros(num_features, device=device).view(1, num_features)

            padded_seqs = [torch.cat([seq] + [pad_tensor] * add_len, dim=0) for seq, add_len in zip(masked_seqs, add_lens)]
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

            return tokens_repr, new_label_mask

        if self.strategy == JunctionStrategy.IGNORE_WITH_MASK or self.strategy == JunctionStrategy.FILL_WITH_I:
            pass
        elif self.strategy == JunctionStrategy.IGNORE_WITH_MASK_BEFORE_LSTM:
            if self.lstm is None:
                raise JunctionError(f'Cannot apply strategy {self.strategy.value} with no LSTM layer!')

            # apply mask to BERT output
            seq_repr, label_mask = apply_mask(seq_repr, label_mask)
            seq_repr = self.dropout(seq_repr)
            seq_repr, _ = self.lstm(seq_repr)
        elif self.strategy == JunctionStrategy.IGNORE_WITH_MASK_BEFORE_CRF:
            if self.lstm is not None:
                seq_repr = self.dropout(seq_repr)
                seq_repr, _ = self.lstm(seq_repr)
            # apply mask to LSTM output
            seq_repr, label_mask = apply_mask(seq_repr, label_mask)
        elif self.strategy == JunctionStrategy.TOKEN_WISE_AVERAGE_BEFORE_CRF:
            raise NotImplementedError  # TODO
        else:
            raise NotImplementedError

        seq_repr = self.dropout(seq_repr)
        label_scores = self.hidden2labels(seq_repr)

        marginal_labels = self.crf.marginal_probabilities(label_scores).transpose(0, 1)

        outputs = marginal_labels, label_mask

        if labels is not None:
            if labels.shape != marginal_labels.shape:
                # convert hard labels into one-hots
                labels = convert_hard_to_soft_labels(labels, self.num_labels)

            if self_training and use_kldiv_loss:
                kld_loss = KLDivLoss()
                raveled_marginal_labels = marginal_labels.contiguous().view(-1, self.num_labels)
                raveled_gold_labels = labels.contiguous().view(-1, self.num_labels)
                raveled_mask = (label_mask.contiguous().view(-1) > 0)
                loss = kld_loss(raveled_marginal_labels[raveled_mask], raveled_gold_labels[raveled_mask])
            else:
                loss = self.crf.forward(label_scores, marginal_tags=labels, mask=(label_mask > 0))

            outputs = (loss,) + outputs

        return outputs  # (loss), scores, label_mask


class RobertaCRFForTokenClassification(BertPreTrainedModel):
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

    def __init__(self, config: RobertaConfig, junction_strategy: JunctionStrategy = JunctionStrategy.IGNORE_WITH_MASK,
                 add_lstm: bool = False, lstm_layers: int = 2, lstm_hidden: int = 128):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.head = CRFForBERT(num_labels=self.num_labels, hidden_size=config.hidden_size, dropout_prob=config.hidden_dropout_prob,
                               junction_strategy=junction_strategy, add_lstm=add_lstm, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers)

        self.init_weights()

    @property
    def returns_probs(self) -> bool:
        return True

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, token_type_ids: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None, head_mask: Optional[Tensor] = None, inputs_embeds: Optional[Tensor] = None,
                labels: Optional[Tensor] = None, label_mask: Optional[Tensor] = None, self_training: bool = False,
                use_kldiv_loss: bool = False):

        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.roberta(input_ids, attention_mask=attention_mask,
                                                                             token_type_ids=token_type_ids, position_ids=position_ids,
                                                                             head_mask=head_mask, inputs_embeds=inputs_embeds)
        # outputs: final_embedding, pooler_output, (hidden_states), (attentions)

        final_embedding = outputs[0]
        head_outputs = self.head(final_embedding, labels=labels, label_mask=label_mask, self_training=self_training,
                                 use_kldiv_loss=use_kldiv_loss)  # (loss), scores, label_mask

        outputs = head_outputs + (final_embedding,) + outputs[2:]

        return outputs  # (loss), scores, label_mask, final_embedding, (hidden_states), (attentions)
