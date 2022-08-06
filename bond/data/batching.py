from _operator import attrgetter
from copy import copy

from dataclasses import dataclass
from typing import Sequence, Union, Iterable

import torch
from torch import BoolTensor, LongTensor, Tensor
from torch.nn.utils.rnn import pad_sequence

from bond.data.example import Example


@dataclass
class BatchedExamples(Example):
    gold_entities_mask: BoolTensor
    token_padding_mask: BoolTensor
    label_mask: BoolTensor
    label_padding_mask: BoolTensor

    def without_labels(self) -> 'BatchedExamples':
        copy_ = copy(self)
        copy_.label_ids = None
        copy_.label_distributions = None
        return copy_

    def with_changes(
            self,
            *,
            token_ids: LongTensor = None,
            labeled_token_mask: BoolTensor = None,
            label_ids: LongTensor = None,
            label_distributions: Tensor = None,
            main_sentences_mask: BoolTensor = None,
            gold_entities_mask: BoolTensor = None,
            token_padding_mask: BoolTensor = None,
            label_mask: BoolTensor = None
    ) -> 'Example':

        new_attributes = dict(
            token_ids=token_ids,
            labeled_token_mask=labeled_token_mask,
            label_ids=label_ids,
            label_distributions=label_distributions,
            main_sentences_mask=main_sentences_mask,
            gold_entities_mask=gold_entities_mask,
            token_padding_mask=token_padding_mask,
            label_mask=label_mask
        )

        for arg_name, arg_value in new_attributes.items():
            if arg_value is None:
                # keep the previous value
                new_attributes[arg_name] = getattr(self, arg_name)

        return BatchedExamples(**new_attributes)

    @classmethod
    def from_examples(cls, examples: Sequence[Example], *, pad_token_id: int = -100, pad_label_id: int = -1):

        def collect_and_pad(attribute: str, *, pad_value: Union[float, int, bool] = 0.0) -> Tensor:
            all_values = list(map(attrgetter(attribute), examples))
            return pad_sequence(all_values, batch_first=True, padding_value=pad_value)

        token_ids: LongTensor = collect_and_pad('token_ids', pad_value=pad_token_id).long()
        labeled_token_mask: BoolTensor = collect_and_pad('labeled_token_mask', pad_value=False).bool()
        label_ids: LongTensor = collect_and_pad('label_ids', pad_value=pad_label_id).long()
        label_distributions: Tensor = collect_and_pad('label_distributions', pad_value=0.0)
        label_mask: BoolTensor = collect_and_pad('label_mask', pad_value=False).bool()
        main_sentences_mask: BoolTensor = collect_and_pad('main_sentences_mask', pad_value=False).bool()
        gold_entities_mask: BoolTensor = collect_and_pad('gold_entities_mask', pad_value=False).bool()

        token_padding_mask: BoolTensor = pad_sequence(
            [torch.ones(len(example), dtype=torch.bool) for example in examples],
            batch_first=True,
            padding_value=False
        ).bool()

        label_padding_mask: BoolTensor = pad_sequence(
            [torch.ones(len(example.label_ids), dtype=torch.bool) for example in examples],
            batch_first=True,
            padding_value=False
        ).bool()

        return BatchedExamples(
            token_ids=token_ids,
            labeled_token_mask=labeled_token_mask,
            label_ids=label_ids,
            label_distributions=label_distributions,
            main_sentences_mask=main_sentences_mask,
            gold_entities_mask=gold_entities_mask,
            token_padding_mask=token_padding_mask,
            label_mask=label_mask,
            label_padding_mask=label_padding_mask
        )


def collate_fn(batch: Iterable[Example], pad_token_id: int, pad_label_id: int) -> BatchedExamples:
    return BatchedExamples.from_examples(tuple(batch), pad_token_id=pad_token_id, pad_label_id=pad_label_id)
