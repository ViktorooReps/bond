## Dataset Summary

The shared task of CoNLL-2003 concerns language-independent named entity recognition. We will concentrate on four types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.

The CoNLL-2003 shared task data files contain four columns separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence. The first item on each line is a word, the second a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. The chunk tags and the named entity tags have the format I-TYPE which means that the word is inside a phrase of type TYPE. Only if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag B-TYPE to show that it starts a new phrase. A word with tag O is not part of a phrase. Note the dataset uses IOB2 tagging scheme, whereas the original dataset uses IOB1.

For more details see https://www.clips.uantwerpen.be/conll2003/ner/ and https://www.aclweb.org/anthology/W03-0419

## Corrected test set

Annotation errors in CoNLL-2003 dataset were estimated to be around 5% of the whole dataset by Zihan Wang et al. [CrossWeigh: Training Named Entity Tagger from Imperfect Annotations](https://arxiv.org/abs/1909.01441) (2019). They also released the corrected version of test set that is used in this repository.

## Distant labels

Distant labels were provided by Liang et al. [BOND: Bert-Assisted Open-Domain Named Entity Recognition with Distant Supervision](https://arxiv.org/abs/2006.15509) (2020)

Labelling quality:

| F1     | Precision | Recall | 
|--------|-----------|--------|
| 0.7097 | 0.8238    | 0.6233 |

With 0.05 added gold entities:

| F1     | Precision | Recall | 
|--------|-----------|--------|
| 0.7267 | 0.8361    | 0.6427 |

With 0.10 added gold entities:

| F1     | Precision | Recall | 
|--------|-----------|--------|
| 0.7406 | 0.8442    | 0.6597 |

With 0.15 added gold entities:

| F1     | Precision | Recall | 
|--------|-----------|--------|
| 0.7568 | 0.8546    | 0.6791 |

