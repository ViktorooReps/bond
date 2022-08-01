import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, List, Iterable, Dict, Any, Iterator

import torch
from torch import LongTensor, BoolTensor, Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from bond.utils import create_one_hot_encoding, recursive_tuple, wrap


@dataclass
class TokenizedSentence:
    token_ids: Tuple[int, ...]
    labeled_token_mask: Tuple[bool, ...]
    label_ids: Tuple[int, ...]
    label_distributions: Tuple[Tuple[float, ...], ...]

    def __len__(self):
        return len(self.token_ids)


@dataclass
class Example(TokenizedSentence):
    token_ids: LongTensor
    labeled_token_mask: BoolTensor
    label_ids: LongTensor
    label_distributions: Tensor
    label_mask: BoolTensor
    main_sentences_mask: BoolTensor  # label-level
    gold_entities_mask: Optional[BoolTensor]

    @classmethod
    def from_sentences(
            cls,
            sentences: Sequence[TokenizedSentence],
            *,
            left_context: Optional[TokenizedSentence] = None,
            right_context: Optional[TokenizedSentence] = None
    ) -> 'Example':

        torch_token_ids: List[LongTensor] = []
        torch_labeled_token_mask: List[BoolTensor] = []
        torch_label_ids: List[LongTensor] = []
        torch_label_distributions: List[Tensor] = []
        torch_main_sentences_mask: List[BoolTensor] = []

        is_main = wrap(left_context, False) + [True] * len(sentences) + wrap(right_context, False)
        all_sentences = wrap(left_context, left_context) + list(sentences) + wrap(right_context, right_context)

        for is_main_sentence, sent in zip(is_main, all_sentences):
            sent_len = len(sent.label_ids)

            torch_token_ids.append(torch.tensor(sent.token_ids, dtype=torch.long).long())
            torch_labeled_token_mask.append(torch.tensor(sent.labeled_token_mask, dtype=torch.bool).bool())
            torch_label_ids.append(torch.tensor(sent.label_ids, dtype=torch.long).long())
            torch_label_distributions.append(torch.tensor(sent.label_distributions, dtype=torch.float))
            torch_main_sentences_mask.append(torch.full([sent_len], fill_value=is_main_sentence, dtype=torch.bool).bool())

        return cls(
            token_ids=torch.concat(torch_token_ids).long(),
            labeled_token_mask=torch.concat(torch_labeled_token_mask).bool(),
            label_ids=torch.concat(torch_label_ids).long(),
            label_distributions=torch.concat(torch_label_distributions),
            label_mask=torch.ones(len(torch_label_ids), dtype=torch.bool).bool(),
            main_sentences_mask=torch.concat(torch_main_sentences_mask).bool(),
            gold_entities_mask=None
        )

    def with_changes(
            self,
            *,
            token_ids: LongTensor = None,
            labeled_token_mask: BoolTensor = None,
            label_ids: LongTensor = None,
            label_distributions: Tensor = None,
            label_mask: BoolTensor = None,
            main_sentences_mask: BoolTensor = None,
            gold_entities_mask: BoolTensor = None,
    ) -> 'Example':

        new_attributes = dict(
            token_ids=token_ids,
            labeled_token_mask=labeled_token_mask,
            label_ids=label_ids,
            label_distributions=label_distributions,
            label_mask=label_mask,
            main_sentences_mask=main_sentences_mask,
            gold_entities_mask=gold_entities_mask
        )

        for arg_name, arg_value in new_attributes.items():
            if arg_value is None:
                # keep the previous value
                new_attributes[arg_name] = getattr(self, arg_name)

        return Example(**new_attributes)


def get_examples(
        json_dataset: Iterable[List[Dict[str, Any]]],
        tokenizer: PreTrainedTokenizer,
        *,
        max_seq_length: int = 512
) -> Iterator[Example]:
    """Creates generator that outputs (token_ids, token_mask, labels) from json dataset"""

    def tokenize_word(word: str) -> Tuple[List[int], List[bool]]:
        token_ids = tokenizer.encode([word], add_special_tokens=False, is_split_into_words=True)
        if not len(token_ids):  # in case tokenizer fails to split token into sub tokens
            token_ids = [tokenizer.unk_token_id]
        mask = [True] + (len(token_ids) - 1) * [False]

        return token_ids, mask

    def tokenize_sentence(sent: dict) -> TokenizedSentence:
        token_ids: List[int] = []
        labeled_token_mask: List[bool] = []
        for word in sent:
            word_token_ids, word_labeled_token_mask = tokenize_word(word)
            token_ids.extend(word_token_ids)
            word_labeled_token_mask.extend(word_labeled_token_mask)

        label_ids = tuple(sent['label_ids'])

        if 'tag_distributions' not in sent:
            label_distributions = create_one_hot_encoding(label_ids)
        else:
            label_distributions = recursive_tuple(sent['tag_distributions'])

        return TokenizedSentence(
            token_ids=tuple(token_ids),
            labeled_token_mask=tuple(labeled_token_mask),
            label_ids=label_ids,
            label_distributions=label_distributions
        )

    for document in tqdm(json_dataset, leave=False):
        tokenized_sentences = tuple(map(tokenize_sentence, document))

        left_context: Optional[TokenizedSentence] = None
        left_context_length = 0

        example_sentences: List[TokenizedSentence] = []
        accumulated_length = 0

        right_context: Optional[TokenizedSentence] = None
        right_context_length = 0

        def example_builder(new_sentences: Iterable[TokenizedSentence]) -> Iterable[Example]:
            """Here is the general algorithm for building an example:

            We construct sentence sequences S_i = [LEFT_CONTEXT] + MAIN_BODY + [RIGHT_CONTEXT]. Square brackets meaning optionality.
            LEFT_CONTEXT is the last sentence of S_(i-1).
            RIGHT_CONTEXT is the first sentence os S_(i+1).
            MAIN_BODY consists of at least one sentence.

            (*) Left and right contexts are added only if main body already has at least one sentence. All sequence should be limited to
            maximum number of subtokens in one example.

            Algorithm A is applied to sentences s_0, ..., s_n: A(s_0, ..., s_n), and outputs target sequences S_0, ..., S_m.

            For S_j:
                1. Check s_0 and update left context and main body of S_j0 following (*). Output S_j and apply A(s_1, ..., s_n)
                if s_0 exceeds maximum example length limit.
                2. (iteratively for i = 1, 2, 3, ...) Let S_ji = S_j(i-1), then add current right context of S_ji to the main body
                and update right context with s_i. End iteration if S_ji length exceeds maximum example length.
                3. Let S_j = S_j(i-1). Output S_j and apply:
                    1) A(s_i, ..., s_n) if S_j did not have any right context.
                    2) A(s_(i-1), ..., s_n) if S_j did have right context.
            """
            # these might change
            nonlocal left_context
            nonlocal left_context_length
            nonlocal example_sentences
            nonlocal accumulated_length
            nonlocal right_context
            nonlocal right_context_length

            for current_sentence in new_sentences:
                sent_length = len(current_sentence)

                if not len(example_sentences):
                    # the case of the first sentence is unique

                    if sent_length > max_seq_length:
                        logging.warning(f'Too long sentence ({sent_length} subtokens) detected! This might cause memory issues!')

                        # left context of the next sentence will be too long either way
                        left_context = None
                        left_context_length = 0

                        # yield long sentence as separate example
                        yield Example.from_sentences([current_sentence])
                        continue

                    if left_context_length + sent_length > max_seq_length:
                        # left context is too long, so we omit it
                        left_context = None
                        left_context_length = 0

                    example_sentences = [current_sentence]
                    accumulated_length = len(current_sentence)
                    continue  # skip updating

                if left_context_length + accumulated_length + right_context_length + sent_length > max_seq_length:
                    yield Example.from_sentences(example_sentences, left_context=left_context, right_context=right_context)

                    # last sentence of current example will always be the left context of the next example
                    left_context = example_sentences[-1]
                    left_context_length = len(left_context)

                    example_sentences = []
                    accumulated_length = 0

                    next_sentences = ([right_context] if right_context is not None else []) + [current_sentence]

                    right_context = None
                    right_context_length = 0

                    yield from example_builder(next_sentences)
                    continue  # `example_builder` call will update everything, so we skip update on this iteration

                # updating (at this point `example_sentences` are at least one example long)

                if right_context is not None:
                    # shift right context to main sentences
                    example_sentences.append(right_context)
                    accumulated_length += right_context_length

                right_context = current_sentence
                right_context_length = len(current_sentence)

        yield from example_builder(tokenized_sentences)

        if len(example_sentences):
            yield Example.from_sentences(example_sentences, left_context=left_context, right_context=right_context)
