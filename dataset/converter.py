from typing import Any, Dict, List, Optional, Set

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
