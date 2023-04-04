import json
import logging
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import wn
from gensim.models.keyedvectors import load_word2vec_format
from numpy import dot
from numpy.linalg import norm
from pandas import DataFrame
from tqdm import tqdm

from amr_to_sl_repr.pipeline.utils import standardize_gloss

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.DEBUG
)


def add_en_translations(df: DataFrame):
    """Add "translations" to the dataframe in the "en" column. These translations are retrieved
    by looking up the "possible Dutch translations" in the "nl" column in Open Multilingual Wordnet (Dutch)
    and finding their equivalent in the English WordNet. This means these English translations will be _very_ broad.

    :param df: input Dataframe that must have an "nl" column
    :return: updated DataFrame that now also includes an "en" column
    """
    nwn = wn.Wordnet("omw-nl:1.4")

    @lru_cache
    def translate(nl: str):
        # One gloss has multiple Dutch "translations"
        # For each word, we find all of its possible translations through open multilingual Wordnet
        if not nl:
            return ""

        nl = remove_brackets(nl)
        nl = nl.split(", ")

        en_translations = set()
        for word in nl:
            if word[0].isupper():
                # Probably a city, country or name
                en_translations.add(word)
                continue

            word = word.strip()
            nl_words = nwn.words(word)

            if nl_words:
                for nl_word in nl_words:
                    for _, en_words in nl_word.translate("omw-en:1.4").items():
                        en_translations.update([en_w.lemma() for en_w in en_words])
            else:
                # TODO: use DEEPL to translate
                pass

        return ", ".join(sorted(en_translations))

    tqdm.pandas(desc="Translating NL 'translations' to EN with WordNet")
    df["en"] = df["nl"].progress_apply(translate)

    return df


def remove_brackets(word: str):
    """Remove ending brackets from a word. Sometimes, the "possible translations" end in brackets to further
    specify the context of the word. But for looking things up in WordNet, that will lead to issues.

    :param word: a given string
    :return: the modified string where open/closing brackets and anything between are removed
    """
    word = re.sub(r"\([^)]*\)", "", word)
    word = " ".join(word.split())  # fix white-spaces
    return word


def maybe_download_ft_vectors() -> Tuple[Path, Path]:
    """Download the fastText aligned vectors if they are not on disk yet. Will download to models/fasttext in the root
    directory of this repository.
    :return: a tuple of Paths to the Dutch and English downloaded vectors (.vec files)
    """
    dout = Path(__file__).resolve().parents[3].joinpath("models/fasttext")
    dout.mkdir(exist_ok=True, parents=True)

    def download(url: str):
        fname = url.split("/")[-1]
        pfout = dout.joinpath(fname)

        if pfout.exists():
            return pfout

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with pfout.open("wb") as fhout:
            with tqdm(total=total_size / (32*1024.0),
                      unit="B",
                      unit_scale=True,
                      unit_divisor=1024,
                      desc=f"Downloading {pfout.name}") as pbar:
                for data in response.iter_content(32*1024):
                    fhout.write(data)
                    pbar.update(len(data))

        return pfout

    url_nl = "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.nl.align.vec"
    url_en = "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec"

    local_nl = download(url_nl)
    local_en = download(url_en)

    return local_nl, local_en


def filter_en_translations(df: DataFrame, threshold: float = 0.1):
    """Uses fastText embeddings and calculates the centroid of the words in the verified "possible translations" (Dutch) column,
    and for each suggested WordNet translation (English) we calculate the cosine similarity to this centroid, so output value
    is between -1 and +1, NOT between 0 and 1!

    :param df: dataframe with "en" (WordNet translations) and "nl" columns ("verified" translations)
    :param threshold: minimal value. If cosine similarity is below this, the word will not be included
    :return: the updated DataFrame where potentially items have been removed from the "en" column
    """
    ft_nl_path, ft_en_path = maybe_download_ft_vectors()

    logging.info("Loading fastText word vectors")
    ft_nl = load_word2vec_format(str(ft_nl_path), binary=False)
    ft_en = load_word2vec_format(str(ft_en_path), binary=False)
    cos_sim = lambda a, b: dot(a, b) / (norm(a) * norm(b))

    def filter_translations(row):
        if not row["nl"] or not row["en"]:
            return row["en"]
        # Do this for the list, of, words because sometimes
        # there is a comma in the parentheses, which leads to unexpected results such as
        # bewerkingsteken (+, -...) -> -...)
        nl_words = remove_brackets(row["nl"])
        nl_words = [w.strip() for w in nl_words.split(", ")]
        nl_vecs = [ft_nl[w] for w in nl_words if w in ft_nl]

        if not nl_vecs:
            return row["en"]

        nl_centroid = np.mean(nl_vecs, axis=0)

        en_words = [w.strip() for w in row["en"].split(", ")]
        filtered_en = []
        for en in en_words:
            if en not in ft_en:
                # Do not include it if it is not found!
                continue
            en_vec = ft_en[en]
            sim = cos_sim(en_vec, nl_centroid)

            if sim >= threshold:
                filtered_en.append(en)

        return ", ".join(filtered_en)

    tqdm.pandas(desc="Filtering EN translations by semantic comparison")
    df["en"] = df.progress_apply(filter_translations, axis=1)
    return df


def main(fin: str, threshold: float = 0.1) -> Path:
    """Generate WordNet translations for the words in the "possible translation" column in the VGT dictionary.
    Then, filter those possible translations by comparing them with the centroid fastText vector. English translations
    that have a cosine similarity (between -1 and +1) of less than the threshold will not be included in the final
    DataFrame.

    "Dictionary" mappings will be created for en<>gloss and nl<>gloss and saved in a JSON file at the same location
    as the input file.

    Returns the path to the written JSON file

    :param fin: path to the VGT dictionary in TSV format
    :param threshold: similarity threshold. Lower similarity English words will not be included
    """
    pfin = Path(fin).resolve()

    df = pd.read_csv(fin, sep="\t")
    df = df.iloc[:, [1, 2]]
    df = df.rename(columns={df.columns[0]: "gloss", df.columns[1]: "nl"})
    df["nl"] = df["nl"].apply(lambda nl: ", ".join(map(str.strip, nl.split(", "))))  # clean possible white-space issues
    df = add_en_translations(df)
    df = filter_en_translations(df, threshold=threshold)

    gloss2en = defaultdict(set)
    gloss2nl = defaultdict(set)
    en2gloss = defaultdict(set)
    nl2gloss = defaultdict(set)

    logging.info("Building final JSON")
    for item in df.to_dict(orient="records"):
        gloss = standardize_gloss(item["gloss"])
        en_words = sorted([en_strip for en in item["en"].split(", ") if (en_strip := en.strip())])
        nl_words = sorted([nl_strip for nl in item["nl"].split(", ") if (nl_strip := nl.strip())])

        gloss2en[gloss].update(en_words)
        gloss2nl[gloss].update(nl_words)

        for word in en_words:
            en2gloss[word].add(gloss)

        for word in nl_words:
            nl2gloss[word].add(gloss)

    all_dicts = {
        "gloss2en": {k: sorted(gloss2en[k]) for k in sorted(gloss2en)},
        "gloss2nl": {k: sorted(gloss2nl[k]) for k in sorted(gloss2nl)},
        "en2gloss": {k: sorted(en2gloss[k]) for k in sorted(en2gloss)},
        "nl2gloss": {k: sorted(nl2gloss[k]) for k in sorted(nl2gloss)},
    }

    pfout = pfin.with_name(f"{pfin.stem}.json")

    with pfout.open("w", encoding="utf-8") as fhout:
        json.dump(all_dicts, fhout, indent=4)

    return pfout


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Generate WordNet translations for the words in the 'possible"
                                                  " translation' column in the VGT dictionary. Then, filter those"
                                                  " possible translations by comparing them with the centroid fastText"
                                                  " vector. English translations that have a cosine similarity (between"
                                                  " -1 and +1) of less than the threshold will not be included in the"
                                                  " final DataFrame.\nNOTE: this script may take a long time to run"
                                                  " because it needs to load the huge fastText models. You'll also need"
                                                  " at least 8GB of RAM. On a fast machine the process will take around"
                                                  " 10 minutes.")

    cparser.add_argument("fin", help="VGT dictionary in TSV format")
    cparser.add_argument("-t", "--threshold", type=float, default=0.1,
                         help="similarity threshold. Lower similarity English words will not be included")

    main(**vars(cparser.parse_args()))
