import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from multiprocessing import Manager
from multiprocessing.managers import DictProxy
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional

import openai
import pandas as pd
from openai.error import RateLimitError
from openai.error import Timeout as OAITimeout
from pandas import DataFrame
from requests.exceptions import Timeout as ReqTimeout
from tqdm import tqdm


openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_response(messages: List[Dict[str, str]], mgr_flags: DictProxy, model: str = "gpt-3.5-turbo") -> Optional[str]:
    """Post a request to the OpenAI ChatCompletion API.
    :param mgr_flags: shared manager dict to keep track of errors that we may encounter
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
                mgr_flags["rate_limit_reached"] = True
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

    mgr_flags["total_failures"] += 1
    return None


@lru_cache
def openai_translate(item_idx: int, nl_words: str, mgr_flags: DictProxy, model: str = "gpt-3.5-turbo") -> Dict:
    """Translate a sequence of Dutch words (separated with a comma) with the OpenAI API. The idea is that the "cluster"
    of words is translated (as if it were a synset) and that the models avoids to translate word-per-word but instead
    comes up with translations that match all the given words.

    :param item_idx: the ID of this word ('ID' column in the dictionary) -- does not always correspond with the index
    in the DataFrame!
    :param nl_words: concatenation of Dutch words, separated by comma
    :param mgr_flags: shared manager dict to keep track of errors that we may encounter
    :param model: the OpenAI model to use for translation
    :return: a (possible empty) list of English translations, or None in case of error
    """
    response = {"ID": item_idx, "nl_words": nl_words, "translations": None}

    if mgr_flags["rate_limit_reached"] or mgr_flags["total_failures"] >= 3:
        return response
    elif not nl_words:
        response["translations"] = []
        return response

    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant that translates Dutch to English according to the requirements that are"
        " given to you.",
    }

    user_prompt = {
        "role": "user",
        "content": f"Consider the following list of one or more Dutch words. They are synonyms. Provide one or more"
        f" translations of this Dutch 'synset' of concepts in English. Format the resulting list of possible"
        f" translations as a JSON list (without markdown ```json``` marker) and do not add an explanation or"
        f" extra information. If you cannot translate a word (such as a city or a name), simply copy the"
        f" input word.\n\n{nl_words}",
    }

    translation = get_response([system_prompt, user_prompt], mgr_flags, model).strip()

    if not translation:
        response["translations"] = []
        return response

    try:
        translation = re.sub(r".*(\[.*\]).*", "\\1", translation)
        response["translations"] = json.loads(translation)
        return response
    except Exception:
        try:
            splut = re.split(r"\",\s*\"", translation)
            splut = [re.sub(r"\[?\"?", "", t) for t in splut]
            splut = [re.sub(r"\"?\]?", "", t) for t in splut]
            splut = [re.sub(r"(.*)\n.*", "\\1", t) for t in splut]
            response["translations"] = splut
            return response
        except Exception:
            # It is expected that some city/people names cannot be translated. In all other cases, this indicates an error.
            if not nl_words[0].isupper():
                logging.error(
                    f"Could not parse translation '{translation}' as list of translations (input: {nl_words}). Returning empty"
                    " list instead."
                )
            return response


def get_translated_idxs(df: DataFrame) -> List[int]:
    if "en" not in df.columns:
        return []

    has_translations = df[~df["en"].isnull()]

    return has_translations["ID"].tolist()


def translate_vgt_with_openai(
    fin: str, fout: str, model: str = "gpt-3.5-turbo", max_parallel_requests: int = 16, first_n: Optional[int] = None
):
    skip_idxs = []
    if Path(fout).exists():
        df_out = pd.read_csv(fout, sep="\t", encoding="utf-8")
        skip_idxs += get_translated_idxs(df_out)
    else:
        df_out = None

    df = pd.read_csv(fin, sep="\t", encoding="utf-8")

    with Manager() as manager:
        mgr_flags = manager.dict({"rate_limit_reached": False, "total_failures": 0})

        with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = {}
            processed_items = 0
            # Submit jobs
            for _, item in df.iterrows():
                item_idx = item["ID"]
                if item_idx in skip_idxs:
                    # Replace cell value with existing translations that were already found
                    if df_out is not None:
                        df.loc[df["ID"] == item_idx, "en"] = df_out.loc[df_out["ID"] == item_idx, "en"]
                    continue

                nl_words = item[2]
                futures[executor.submit(openai_translate, item_idx, nl_words, mgr_flags, model)] = item
                processed_items += 1
                if processed_items == first_n:
                    break

            # Read job results
            failed_counter = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
                result = future.result()
                item_idx = result["ID"]
                transls = result["translations"]
                if transls is None:
                    nl_words = result["nl_words"]
                    # If cities/people names were not translated correctly, just copy them from Dutch
                    if nl_words[0].isupper():
                        df.loc[df["ID"] == item_idx, "en"] = nl_words
                    else:
                        failed_counter += 1
                else:
                    translations = ", ".join(sorted(transls))
                    df.loc[df["ID"] == item_idx, "en"] = translations

            if failed_counter:
                logging.warning(f"Done processing. Had at least {failed_counter:,} failures. See the logs above.")

        if mgr_flags["rate_limit_reached"]:
            logging.error("Had to abort early due to the OpenAI rate limit. Seems like you hit your limit!")

        if mgr_flags["total_failures"] >= 3:
            logging.error("Had more than 3 catastrophic failures. Will stop processing. See the error messages above.")

    cols = df.columns.tolist()
    reordered_cols = cols[:3] + [cols[-1]] + cols[3:-1]
    df = df[reordered_cols]
    df.to_csv(fout, sep="\t", index=False, encoding="utf-8")


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        "Translate the 'possible translations' (Dutch) in the VGT dictionary to English"
        " with a GPT model. This is useful because we can give it a cluster of words"
        " and ask for similar English words. As such, it is 'context-sensitive'; taking"
        " into account the whole cluster of words.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument("fin", help="VGT dictionary in TSV format")
    cparser.add_argument("fout", help="Output file to write the updated TSV to")
    cparser.add_argument("-m", "--model", default="gpt-3.5-turbo", help="Chat model to use")
    cparser.add_argument("-j", "--max_parallel_requests", default=16, type=int, help="Max. parallel requests to query")
    cparser.add_argument("-i", "--first_n", default=None, type=int, help="For debugging: only translate first n items")
    translate_vgt_with_openai(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
