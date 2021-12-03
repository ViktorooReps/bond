from typing import Dict, Iterable, Optional, Tuple

Scores = Dict[str, float]


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
