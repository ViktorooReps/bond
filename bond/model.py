from torch import FloatTensor
from torch.nn.functional import one_hot
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss

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

    def __init__(self, config):
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

        outputs = (logits, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:

            # Only keep active parts of the loss
            if attention_mask is not None or label_mask is not None:
                active_loss = True
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                if label_mask is not None:
                    active_loss = active_loss & (label_mask.view(-1) == 1)
                active_logits = logits.view(-1, self.num_labels)[active_loss]

            if labels.shape == logits.shape:
                loss_fct = KLDivLoss()
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

        return outputs  # (loss), scores, final_embedding, (hidden_states), (attentions)


def convert_hard_to_soft_labels(labels, num_labels: int) -> FloatTensor:
    # TODO: implement UNK tokens
    labels[labels < 0] = 0
    return one_hot(labels, num_labels)


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

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden2label = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = MarginalCRF(self.num_labels)

        self.init_weights()

    @property
    def returns_probs(self) -> bool:
        return True

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
        label_scores = self.hidden2label(sequence_output)

        marginal_labels = self.crf.marginal_probabilities(label_scores).transpose(0, 1)

        outputs = (marginal_labels, final_embedding,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # might be a problem when labels are not soft
            if labels.shape != marginal_labels.shape:
                labels = convert_hard_to_soft_labels(labels, self.num_labels)

            batch_size, seq_len, _ = labels.shape
            loss = self.crf.forward(label_scores, marginal_tags=labels, mask=(label_mask == 1))
            outputs = (loss,) + outputs

            # kld_loss = KLDivLoss()
            # loss_kld = kld_loss(marginal_labels.view(-1, self.num_labels)[mask], labels.view(-1, self.num_labels)[mask])
            # outputs = (loss_kld,) + outputs

        return outputs  # (loss), scores, final_embedding, (hidden_states), (attentions)
