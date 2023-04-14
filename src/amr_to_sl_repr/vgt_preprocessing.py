import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional

import numpy as np
import openai
import pandas as pd
import wn
from amr_to_sl_repr.utils import standardize_gloss
from openai.error import RateLimitError
from openai.error import Timeout as OAITimeout
from pandas import DataFrame
from requests.exceptions import Timeout as ReqTimeout
from tqdm import tqdm

from amr_to_sl_repr.vec_similarity import cos_sim, load_fasttext_models

openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_response(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> Optional[str]:
    """Post a request to the OpenAI ChatCompletion API.
    :param messages: a list of dictionaries with keys "role" and "content"
    :param model: the OpenAI model to use for translation
    :return: the model's translations
    """
    num_retries = 3
    while num_retries > 0:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=128,
                temperature=0,
            )
        except Exception as exc:
            if isinstance(exc, RateLimitError):
                logging.exception(f"Rate limit reached on {time.ctime()}! Error trace:")
                break
            else:
                num_retries -= 1
                logging.exception(f"Error occurred on {time.ctime()}! ({num_retries} retries left)... Error trace:")
                if isinstance(exc, (OAITimeout, ReqTimeout, TimeoutError)):
                    sleep(60)
                elif isinstance(exc, openai.error.APIError):
                    sleep(30)
                else:
                    sleep(10)
        else:
            assistant_response = response["choices"][0]["message"]["content"]
            return assistant_response

    return None


@lru_cache
def openai_translate(nl_words: str) -> List[str]:
    """Translate a sequence of Dutch words (separated with a comma) with the OpenAI API. The idea is that the "cluster"
    of words is translated (as if it were a synset) and that the models avoids to translate word-per-word but instead
    comes up with translations that match all the given words.

    :param nl_words: concatenation of Dutch words, separated by comma
    :return: a (possible empty) list of English translations
    """
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant that translates Dutch to English according to the requirements that are"
        " given to you.",
    }

    user_prompt = {
        "role": "user",
        "content": f"Consider the following list of Dutch words. They are synonyms. Provide one or more translations of"
        f" this Dutch 'synset' of concepts in English. Format the resulting list of possible translations as"
        f" a JSON list (without markdown ```json``` marker) and do not add an explanation or extra"
        f" information. \n\n{nl_words}",
    }

    translation = get_response([system_prompt, user_prompt]).strip()

    if not translation:
        return []

    try:
        translation = re.sub(r".*(\[.*\]).*", "\\1", translation)
        return json.loads(translation)
    except Exception:
        logging.error(
            f"Could not parse translation '{translation}' (input: {nl_words}) as JSON. Returning empty"
            " list instead."
        )
        return []


def add_en_translations(df: DataFrame, no_openai: bool = False):
    """Add "translations" to the dataframe in the "en" column. These translations are retrieved
    by looking up the "possible Dutch translations" in the "nl" column in Open Multilingual Wordnet (Dutch)
    and finding their equivalent in the English WordNet. This means these English translations will be _very_ broad.

    :param df: input Dataframe that must have an "nl" column
    :param no_openai: whether to disable OpenAI translation
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
        nl_words = nl.split(", ")

        en_translations = set()
        for nl_word in nl_words:
            if nl_word[0].isupper():
                # Probably a city, country or name
                en_translations.add(nl_word)
                continue

            nl_word = nl_word.strip()
            nl_wn_words = nwn.words(nl_word)

            if nl_wn_words:
                for nl_wn_word in nl_wn_words:
                    for _, en_wn_words in nl_wn_word.translate("omw-en:1.4").items():
                        en_translations.update([en_w.lemma() for en_w in en_wn_words])

        if not no_openai:
            translated_words = openai_translate(nl)
            en_translations.update(translated_words)

        return ", ".join(sorted(en_translations))

    tqdm.pandas(desc="Translating NL 'translations' to EN with WordNet and OpenAI")
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


def filter_en_translations(df: DataFrame, threshold: float = 0.1):
    """Uses fastText embeddings and calculates the centroid of the words in the verified "possible translations" (Dutch) column,
    and for each suggested WordNet translation (English) we calculate the cosine similarity to this centroid, so output value
    is between -1 and +1, NOT between 0 and 1!

    :param df: dataframe with "en" (WordNet translations) and "nl" columns ("verified" translations)
    :param threshold: minimal value. If cosine similarity is below this, the word will not be included
    :return: the updated DataFrame where potentially items have been removed from the "en" column
    """
    ft_nl, ft_en = load_fasttext_models()

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
                # Do not include it if it is not found in the ft vectors!
                continue
            en_vec = ft_en[en]
            sim = cos_sim(en_vec, nl_centroid)

            if sim >= threshold:
                filtered_en.append(en)
            else:
                logging.info(f"Dropping {en}. Too distant from {nl_words}! (sim={sim:.2f}; threshold={threshold})")

        return ", ".join(filtered_en)

    tqdm.pandas(desc="Filtering EN translations by semantic comparison")
    df["en"] = df.progress_apply(filter_translations, axis=1)
    return df


def main(fin: str, threshold: float = 0.1, no_openai: bool = False, no_fasttext: bool = False) -> Path:
    """Generate WordNet translations for the words in the "possible translation" column in the VGT dictionary.
    Then, filter those possible translations by comparing them with the centroid fastText vector. English translations
    that have a cosine similarity (between -1 and +1) of less than the threshold will not be included in the final
    DataFrame.

    "Dictionary" mappings will be created for en<>gloss and nl<>gloss and saved in a JSON file at the same location
    as the input file.

    Returns the path to the written JSON file

    :param fin: path to the VGT dictionary in TSV format
    :param no_openai: whether to disable OpenAI translation
    :param no_fasttext: whether to disable fastText filtering of results
    :param threshold: similarity threshold. Lower similarity English words will not be included. Will not be used if
    'no_fasttext is True
    """
    pfin = Path(fin).resolve()

    df = pd.read_csv(fin, sep="\t")
    df = df.rename(columns={df.columns[1]: "gloss", df.columns[2]: "nl"})
    df["nl"] = df["nl"].apply(
        lambda nl: ", ".join(map(str.strip, nl.split(", ")))
    )  # clean possible white-space issues
    df = add_en_translations(df, no_openai=no_openai)
    if not no_fasttext:
        df = filter_en_translations(df, threshold=threshold)

    cols = list(df.columns)
    reordered_cols = cols[:3] + [cols[-1]] + cols[3:-1]
    df = df[reordered_cols]

    pfout_tsv = pfin.with_name(f"{pfin.stem}+en_transls{pfin.suffix}")
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

    pfout = pfin.with_name(f"{pfin.stem}+en_transls.json")

    with pfout.open("w", encoding="utf-8") as fhout:
        json.dump(all_dicts, fhout, indent=4)

    logging.info(f"Saved updated JSON in {pfout.resolve()}")
    return pfout


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        description="Generate WordNet and OpenAI translations for the words in the 'possible"
        " translation' column in the VGT dictionary. (To use the OpenAI "
        " translation, your key must be set as an environment variable in "
        " 'OPENAI_API_KEY'. If you not have an OpenAI account, you can disable"
        " this option with the '--no_openai' flag) Then, filter those possible"
        " translations by comparing them with the centroid fastText"
        " vector. English translations that have a cosine similarity (between"
        " -1 and +1) of less than the threshold will not be included in the"
        " final DataFrame.\nNOTE: this script may take a long time to run"
        " because it needs to query the OpenAI API for translations and it"
        " also needs to load the huge fastText models. You'll need"
        " at least 8GB of RAM to run this process."
    )

    cparser.add_argument("fin", help="VGT dictionary in TSV format")
    cparser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        help="similarity threshold. Lower similarity English words will not be included",
    )
    cparser.add_argument("--no_openai", action="store_true", help="whether to disable OpenAI translation")
    cparser.add_argument(
        "--no_fasttext", action="store_true", help="whether to disable the ambiguity filtering with fastText"
    )

    main(**vars(cparser.parse_args()))
