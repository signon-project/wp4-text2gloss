from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, List

import pandas as pd
import penman


def get_en2vgtgloss_dict(fin: str) -> Dict[str, List[str]]:
    df = pd.read_csv(fin, sep="\t", encoding="utf-8")
    transl2gloss = defaultdict(list)
    # iterate over col1 and col2 simultaneously
    for index, row in df.iterrows():
        gloss = row["GLOSS"]
        transls = row["translations"]

        if pd.isna(transls):
            continue

        for transl in transls.split(", "):
            transl2gloss[transl].append(gloss)

    return transl2gloss


def get_amr_concepts(graph: penman.Graph) -> List[str]:
    concepts = []
    for _, rel, target in graph.triples:
        if rel == ":instance":
            target = re.sub(r"-\d+$", "", target)
            concepts.append(target)

    return concepts


def concept2gloss(concept: str, dictionary: Dict[str, List[str]]):
    try:
        print(concept, dictionary[concept])
        return dictionary[concept][0]
    except (KeyError, IndexError):
        return None


def main(dict_path: str, amr_path: str):
    dictionary = get_en2vgtgloss_dict(dict_path)
    with open(amr_path, encoding="utf-8") as fhin:
        for graph in penman.iterdecode(fhin):
            concepts = get_amr_concepts(graph)
            print(graph.metadata["snt"])
            glossed = " ".join([gloss for concept in concepts if (gloss := concept2gloss(concept, dictionary))])
            print(glossed)
            print()


if __name__ == '__main__':
    import argparse

    cparser = argparse.ArgumentParser()
    cparser.add_argument("dict_path", help="Path to VGT dictionary (tsv)")
    cparser.add_argument("amr_path", help="Path to sentences in AMR format with ::snt attribute")

    main(**vars(cparser.parse_args()))
