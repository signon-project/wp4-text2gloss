from functools import lru_cache

import pandas as pd
import wn
from tqdm import tqdm


@lru_cache
def find_word_synonyms(word, wordnet):
    syns = []
    for synset in wordnet.synsets(word):
        syns.extend([w.lemma().lower() for w in synset.words() if w.lemma().lower() != word])

    return syns


def find_all_gloss_synonyms(nl_translations: str, wordnet):
    nl_translations = nl_translations.split(", ")
    syns = set()
    for transl in nl_translations:
        syns.update(find_word_synonyms(transl, wordnet))

    return ", ".join(sorted(syns))


@lru_cache
def find_word_translations(word, wordnet):
    syns = []
    for synset in wordnet.synsets(word):
        for w in synset.words():
            for _, words in w.translate(lang="en").items():
                syns.extend([_w.lemma().lower() for _w in words if _w.lemma().lower() != word])

    return syns


def find_all_translations(nl_translations: str, wordnet):
    nl_translations = nl_translations.split(", ")
    transls = set()
    for transl in nl_translations:
        transls.update(find_word_translations(transl, wordnet))

    return ", ".join(sorted(transls))


def main():
    df = pd.read_csv(r"C:\Python\projects\amr-to-sl-repr\data\vgt-woordenboek-27_03_2023.tsv", sep="\t", encoding="utf-8")

    wordnet = wn.Wordnet("omw-nl:1.4")
    tqdm.pandas()
    df["synonyms"] = df.iloc[:, 2].progress_apply(find_all_gloss_synonyms, args=(wordnet,))
    df["translations"] = df.iloc[:, 2].progress_apply(find_all_translations, args=(wordnet,))

    cols = df.columns.tolist()
    reordered_cols = cols[:3] + ["synonyms", "translations"] + cols[3:-2]
    df = df[reordered_cols]

    print(df.head())
    df.to_csv(r"C:\Python\projects\amr-to-sl-repr\data\vgt-woordenboek-27_03_2023+syns.tsv", sep="\t", encoding="utf-8")


if __name__ == "__main__":
    main()
