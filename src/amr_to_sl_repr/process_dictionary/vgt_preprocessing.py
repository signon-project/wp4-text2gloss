import json
import logging
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
import wn
from amr_to_sl_repr.utils import standardize_gloss
from amr_to_sl_repr.vec_similarity import cos_sim, get_vec_from_api, get_token_exists_in_ft, get_centroid
from pandas import DataFrame, Series
from tqdm import tqdm


def add_en_translations(df: DataFrame):
    """Add "translations" to the dataframe in the "en" column. These translations are retrieved
    by looking up the "possible Dutch translations" in the "nl" column in Open Multilingual Wordnet (Dutch)
    and finding their equivalent in the English WordNet. This means these English translations will be _very_ broad.

    :param df: input Dataframe that must have an "nl" column
    :return: updated DataFrame that now also includes an "en" column
    """
    nwn = wn.Wordnet("omw-nl:1.4")

    @lru_cache(maxsize=256)
    def translate_word(nl_word: str):
        en_translations = set()

        if nl_word[0].isupper():
            # Probably a city, country or name
            en_translations.add(nl_word)
        else:
            nl_word = nl_word.strip()
            nl_wn_words = nwn.words(nl_word)

            if nl_wn_words:
                for nl_wn_word in nl_wn_words:
                    for _, en_wn_words in nl_wn_word.translate("omw-en:1.4").items():
                        en_translations.update([en_w.lemma() for en_w in en_wn_words])

        return en_translations

    def translate(row: Series):
        nl = row["nl"]
        # One gloss has multiple Dutch "translations"
        # For each word, we find all of its possible translations through open multilingual Wordnet
        if not nl:
            return ""

        nl = remove_brackets(nl)
        nl_words = nl.split(", ")

        if "en" in row:
            en_translations = set(row["en"].split(", "))
        else:
            en_translations = set()

        for nl_word in nl_words:
            en_translations.update(translate_word(nl_word))

        return ", ".join(sorted(en_translations))

    tqdm.pandas(desc="Translating NL 'translations' to EN with WordNet")
    df["en"] = df.progress_apply(translate, axis=1)

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


def filter_en_translations(df: DataFrame, threshold: float = 0.1):
    """Uses fastText embeddings and calculates the centroid of the words in the verified "possible translations" (Dutch) column,
    and for each suggested WordNet translation (English) we calculate the cosine similarity to this centroid, so output value
    is between -1 and +1, NOT between 0 and 1!

    :param df: dataframe with "en" (WordNet translations) and "nl" columns ("verified" translations)
    :param threshold: minimal value. If cosine similarity is below this, the word will not be included
    :return: the updated DataFrame where potentially items have been removed from the "en" column
    """
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    def filter_translations(row):
        if not row["nl"] or not row["en"]:
            return row["en"]
        # Do this for the list, of, words because sometimes
        # there is a comma in the parentheses, which leads to unexpected results such as
        # `bewerkingsteken (+, -...)` -> -...)
        nl_words = remove_brackets(row["nl"])
        nl_words = tuple([w.strip() for w in nl_words.split(", ")])
        nl_centroid = get_centroid(nl_words, lang="Dutch", session=session)

        if nl_centroid is None:
            return row["en"]

        en_words = [w.strip() for w in row["en"].split(", ")]
        filtered_en = []
        for en in en_words:
            if not get_token_exists_in_ft(en, lang="English", session=session):
                # Do not include it if it is not found in the ft vectors!
                continue
            en_vec = get_vec_from_api(en, lang="English", session=session)

            sim = cos_sim(en_vec, nl_centroid)

            if sim >= threshold:
                filtered_en.append(en)
            else:
                logging.info(f"Dropping {en}. Too distant from {nl_words}! (sim={sim:.2f}; threshold={threshold})")

        return ", ".join(filtered_en)

    tqdm.pandas(desc="Filtering EN translations by semantic comparison")
    df["en"] = df.progress_apply(filter_translations, axis=1)
    session.close()
    return df


def main(fin: str, threshold: float = 0.1, no_fasttext: bool = False) -> Path:
    """Generate WordNet translations for the words in the "possible translation" column in the VGT dictionary.
    Then, filter those possible translations by comparing them with the centroid fastText vector. English translations
    that have a cosine similarity (between -1 and +1) of less than the threshold will not be included in the final
    DataFrame.

    "Dictionary" mappings will be created for en<>gloss and nl<>gloss and saved in a JSON file at the same location
    as the input file.

    Returns the path to the written JSON file

    :param fin: path to the VGT dictionary in TSV format
    :param no_fasttext: whether to disable fastText filtering of results
    :param threshold: similarity threshold. Lower similarity English words will not be included. Will not be used if
    'no_fasttext is True
    """
    pfin = Path(fin).resolve()

    df = pd.read_csv(fin, sep="\t")
    had_en_column = "en" in df.columns
    df = df.rename(columns={df.columns[1]: "gloss", df.columns[2]: "nl"})
    df["nl"] = df["nl"].apply(
        lambda nl: ", ".join(map(str.strip, re.split(r"\s*,\s*", nl)))
    )  # clean possible white-space issues
    df = add_en_translations(df)

    # Filter/disambiguate translations
    if not no_fasttext:
        df = filter_en_translations(df, threshold=threshold)

    if not had_en_column:
        cols = list(df.columns)
        reordered_cols = cols[:3] + [cols[-1]] + cols[3:-1]
        df = df[reordered_cols]

    pfout_tsv = pfin.with_name(f"{pfin.stem}+wn_transls{pfin.suffix}")
    df.to_csv(pfout_tsv, index=False, sep="\t")
    logging.info(f"Saved updated TSV in {pfout_tsv.resolve()}")

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

    pfout = pfin.with_name(f"{pfin.stem}+wn_transls.json")

    with pfout.open("w", encoding="utf-8") as fhout:
        json.dump(all_dicts, fhout, indent=4)

    logging.info(f"Saved updated JSON in {pfout.resolve()}")
    return pfout


if __name__ == "__main__":
    import argparse
    # TODO: update documentation
    cparser = argparse.ArgumentParser(description="")

    cparser.add_argument("fin", help="VGT dictionary in TSV format")
    cparser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        help="similarity threshold. Lower similarity English words will not be included",
    )
    cparser.add_argument(
        "--no_fasttext", action="store_true", help="whether to disable the ambiguity filtering with fastText"
    )

    main(**vars(cparser.parse_args()))
