import json
from typing import List

from bond.data import load_tags_dict, DatasetName


def old_to_new_format():
    with open('dataset/data/conll03/distant/train.json') as distant_file:
        old_formatted_distant: List[dict] = json.load(distant_file)

    with open('dataset/data/conll03/gold/train.json') as gold_file:
        new_formatted_gold: List[List[dict]] = json.load(gold_file)

    old_total_sentences = len(old_formatted_distant)
    doc_lens = [len(new_formatted_doc) for new_formatted_doc in new_formatted_gold]
    new_total_sentences = sum(doc_lens)

    assert old_total_sentences == new_total_sentences

    docs = []
    prev_sent_idx = 0
    for doc_len in doc_lens:
        curr_sent_idx = prev_sent_idx + doc_len
        docs.append(old_formatted_distant[prev_sent_idx: curr_sent_idx])
        prev_sent_idx = curr_sent_idx

    new_doc_lens = [len(doc) for doc in docs]

    assert doc_lens == new_doc_lens

    with open('dataset/data/conll03/distant/train.json', 'w') as distant_file:
        json.dump(docs, distant_file)


def conll03_raw_to_json(raw_file_name: str, json_file_name: str) -> None:
    with open(raw_file_name) as raw_file:
        raw_text = raw_file.read()

    raw_docs = raw_text.split('-DOCSTART- -X- -X- O\n\n')[1:]  # [1:] to remove first empty document
    sentenced_raw_docs = [doc.split('\n\n')[:-1] for doc in raw_docs]  # [:-1] to remove last empty sentence

    tags_dict = load_tags_dict(DatasetName.CONLL2003)

    json_dataset = []
    for sentenced_doc in sentenced_raw_docs:
        document = []
        for sentence in sentenced_doc:
            words = sentence.split('\n')
            str_words = []
            tags = []
            for word in words:
                str_word, _, _, label = word.split(' ')
                str_words.append(str_word)
                tags.append(tags_dict[label])
            document.append({'str_words': str_words, 'tags': tags})
        json_dataset.append(document)

    with open(json_file_name, 'w') as json_file:
        json.dump(json_dataset, json_file)
