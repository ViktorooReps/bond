import json

from tqdm import tqdm

if __name__ == '__main__':
    with open('dataset/data/conll03/distant/train.json') as df:
        corrupted = json.load(df)

    with open('dataset/data/conll03/gold/train.json') as gf:
        gold = json.load(gf)

    new_docs = []
    for corrupted_doc, gold_doc in tqdm(zip(corrupted, gold)):
        new_doc = []
        for corrupted_sentence, gold_sentence in zip(corrupted_doc, gold_doc):
            corrupted_sentence['str_words'] = gold_sentence['str_words']
            new_doc.append(corrupted_sentence)
        new_docs.append(new_doc)

    with open('dataset/data/conll03/distant/train.json', 'w') as cf:
        json.dump(new_docs, cf)
