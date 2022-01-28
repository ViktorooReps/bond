import json
from typing import Any, Dict, List, Optional, Set

from bond.data import load_tags_dict, DatasetName

JsonEntry = Dict[str, list]
JsonDataset = List[JsonEntry]


def transform_dataset(dataset: JsonDataset,
                      key_conversion: Optional[Dict[str, str]] = None,
                      value_conversion: Optional[Dict[str, dict]] = None,
                      exclude_keys: Optional[Set[str]] = None) -> JsonDataset:

    new_dataset: JsonDataset = []
    key_conversion = key_conversion if key_conversion is not None else {}
    value_conversion = value_conversion if value_conversion is not None else {}
    exclude_keys = exclude_keys if exclude_keys is not None else set()

    for entry in dataset:
        new_entry: JsonEntry = {}
        for key in filter(lambda k: k not in exclude_keys, entry.keys()):

            def convert_value(value: Any) -> Any:
                value_converter = value_conversion.get(key, {})
                return value_converter.get(value, value)

            new_entry[key_conversion.get(key, key)] = list(map(convert_value, entry[key]))

        new_dataset.append(new_entry)

    return new_dataset


def conll03_raw_to_json(raw_file_name: str, json_file_name: str):
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
