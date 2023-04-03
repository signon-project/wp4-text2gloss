from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import penman
from sentence_transformers import util, SentenceTransformer


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
        return dictionary[concept]
    except (KeyError, IndexError):
        return None


def similarity_selection(src_sent: str, glosses: List[str], embedder):
    glosses = [re.sub(r"-[A-Z]$", "", g).lower().replace("-", " ") for g in glosses]

    embeddings = embedder.encode([src_sent] + glosses)
    src_embedding = embeddings[0]
    embeddings = embeddings[1:]
    scores = [util.cos_sim(src_embedding, embed).item() for embed in embeddings]
    best_score = np.argmax(scores)
    most_sim_glos = glosses[best_score].upper().replace(" ", "-")

    return most_sim_glos


def main(dict_path: str, amr_path: str):
    dictionary = get_en2vgtgloss_dict(dict_path)
    embedder = SentenceTransformer("distiluse-base-multilingual-cased-v1")

    with open(amr_path, encoding="utf-8") as fhin:
        for graph in penman.iterdecode(fhin):
            concepts = get_amr_concepts(graph)
            src_sent = graph.metadata["snt"]

            print(graph.metadata["snt"])
            print("=" * len(graph.metadata["snt"]))
            print(f"AMR CONCEPTS: {concepts}")
            gloss_cands = {concept: glosses for concept in concepts if (glosses := concept2gloss(concept, dictionary))}

            selected_glosses = []
            for concept, gloss_candidates in gloss_cands.items():
                print(f"\t- OPTIONS FOR {concept}: {gloss_candidates}")
                most_sim_cand = similarity_selection(src_sent, gloss_candidates, embedder=embedder)
                print(f"\t- BEST OPTION FOR {concept}: {most_sim_cand}\n")
                selected_glosses.append(most_sim_cand)

            print(">>> FINAL GLOSSES:", " ".join(selected_glosses))
            print()


if __name__ == '__main__':
    import argparse

    cparser = argparse.ArgumentParser()
    cparser.add_argument("dict_path", help="Path to VGT dictionary (tsv)")
    cparser.add_argument("amr_path", help="Path to sentences in AMR format with ::snt attribute")

    main(**vars(cparser.parse_args()))
