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
    last_error = None
    while num_retries > 0:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=128,
                temperature=0,
            )
        except Exception as exc:
            last_error = exc
            # If ratelimiterror on last retry, stop
            if num_retries == 1 and isinstance(exc, RateLimitError):
                mgr_flags["rate_limit_reached"] = True
                logging.exception(f"Rate limit reached on {time.ctime()}! Error trace:")
                break
            else:
                num_retries -= 1
                if isinstance(exc, (OAITimeout, ReqTimeout, TimeoutError, RateLimitError)):
                    sleep(60)
                elif isinstance(exc, openai.error.APIError):
                    sleep(30)
                else:
                    sleep(10)
        else:
            assistant_response = response["choices"][0]["message"]["content"]
            return assistant_response

    mgr_flags["total_failures"] += 1

    if last_error:
        logging.error(
            f"Error occurred on {time.ctime()}! (this is failure #{mgr_flags['total_failures']})..."
            f" Error:\n{last_error}"
        )

    return None


@lru_cache
def openai_translate(
    item_gloss: str, lang_prompt: str, explanation: str, mgr_flags: DictProxy, model: str = "gpt-3.5-turbo"
) -> Dict:
    """Translate a sequence of Dutch words (separated with a comma) with the OpenAI API. The idea is that the "cluster"
    of words is translated (as if it were a synset) and that the models avoids to translate word-per-word but instead
    comes up with translations that match all the given words.

    :param item_gloss: the gloss
    :param lang_prompt: an explanation of the language to be used inthe prompt. Can be as simple as 'Dutch'
    :param explanation: concatenation of words that explain the gloss, separated by comma
    :param mgr_flags: shared manager dict to keep track of errors that we may encounter
    :param model: the OpenAI model to use for translation
    :return: a (possible empty) list of English translations, or None in case of error
    """
    response = {"gloss": item_gloss, "explanation": explanation, "translations": None}

    if mgr_flags["rate_limit_reached"] or mgr_flags["total_failures"] >= 3:
        return response
    elif not explanation:
        response["translations"] = []
        return response

    system_prompt = {
        "role": "system",
        "content": f"You are a helpful assistant that translates {lang_prompt} to English.",
    }

    user_prompt = {
        "role": "user",
        "content": f"Consider the following list of one or more {lang_prompt} words. They are synonyms."
        f" Provide one or , preferably, multiple translations of this Dutch 'synonym set' of concepts in English."
        f" Format the resulting list of possible translations as a JSON list (without markdown ```json``` marker) and"
        f" do not add an explanation or any extra information. If you cannot translate a word (such as a city or a"
        f" name), simply copy the input word.\n\n{explanation}",
    }

    translation = get_response([system_prompt, user_prompt], mgr_flags, model)

    if not translation or not translation.strip():
        response["translations"] = []
        return response

    translation = translation.strip()

    try:
        translation = re.sub(r".*(\[.*\]).*", "\\1", translation)
        response["translations"] = json.loads(translation)
    except Exception:
        try:
            splut = re.split(r"\",\s*\"", translation)
            splut = [re.sub(r"\[?\"?", "", t) for t in splut]
            splut = [re.sub(r"\"?\]?", "", t) for t in splut]
            splut = [re.sub(r"(.*)\n.*", "\\1", t) for t in splut]
            response["translations"] = splut
        except Exception:
            # It is expected that some city/people names cannot be translated. In that case, ignore it here and copy
            # the name in the main process during collection. In all other cases, this indicates an error.
            if not explanation[0].isupper():
                logging.exception(
                    f"Could not parse translation '{translation}' as list of translations (input: '{explanation}')."
                    f" Returning empty list instead."
                )

    return response


def get_pretranslations(df: DataFrame) -> List[str]:
    if "en" not in df.columns:
        return []

    has_translations = df[~df["en"].isnull()]

    return has_translations["gloss"].tolist()


def translate_with_openai(
    fin: str,
    lang_prompt: str,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    max_parallel_requests: int = 16,
    first_n: Optional[int] = None,
):
    openai.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")

    pfin = Path(fin).resolve()
    pfout = pfin.with_stem(pfin.stem + "-openai").with_suffix(".tsv")

    skip_glosses = []
    if pfout.exists():
        df_out = pd.read_csv(pfout, sep="\t", encoding="utf-8")
        skip_glosses += get_pretranslations(df_out)
    else:
        df_out = None

    df = pd.read_csv(fin, sep="\t", encoding="utf-8").fillna("")

    with Manager() as manager:
        mgr_flags = manager.dict({"rate_limit_reached": False, "total_failures": 0})

        with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = {}
            processed_items = 0
            # Submit jobs
            for _, item in df.iterrows():
                item_gloss = item["gloss"]
                explanation = item[1].strip()

                if not explanation:
                    continue

                if item_gloss in skip_glosses:
                    # Replace cell value with existing translations that were already found
                    if df_out is not None:
                        df.loc[df["gloss"] == item_gloss, "en"] = df_out.loc[
                            df_out["gloss"] == item_gloss, "en"
                        ].copy()
                    continue

                futures[
                    executor.submit(openai_translate, item_gloss, lang_prompt, explanation, mgr_flags, model)
                ] = item
                processed_items += 1
                if processed_items == first_n:
                    break

            # Read job results
            failed_counter = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
                result = future.result()
                item_gloss = result["gloss"]
                transls = result["translations"]
                if transls is None:
                    explanation = result["explanation"]
                    # If cities/people names were not translated correctly, just copy them from Dutch
                    if explanation[0].isupper():
                        df.loc[df["gloss"] == item_gloss, "en"] = explanation
                    else:
                        failed_counter += 1
                else:
                    translations = ", ".join(sorted(transls))
                    df.loc[df["gloss"] == item_gloss, "en"] = translations

            if failed_counter:
                logging.warning(f"Done processing. Had at least {failed_counter:,} failures. See the logs above.")

        if mgr_flags["rate_limit_reached"]:
            logging.error(
                "Had to abort early due to the OpenAI rate limit. Seems like you hit your limit or the server was"
                " overloaded! The generated translations have been saved. You can run the script again the continue"
                " where you left off."
            )

        if mgr_flags["total_failures"] >= 3:
            logging.error(
                "Had more than 3 catastrophic failures. Will stop processing. See the error messages above."
                " The generated translations have been saved. You can run the script again the continue"
                " where you left off."
            )

    df = df.dropna()
    # Reorder columns
    extra_cols = [c for c in df.columns[2:].tolist() if c != "en"]
    df = df[df.columns[:2].tolist() + ["en"] + extra_cols]
    df.to_csv(pfout, sep="\t", index=False, encoding="utf-8")


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        "Translate the 'possible translations' in the dictionary to English"
        " with a GPT model. This is useful because we can give it a cluster of words"
        " and ask for similar English words. As such, it is 'context-sensitive'; taking"
        " into account the whole cluster of words.\nIf you get a RateLimitError concerning using more tokens per minute"
        " than is accepted, you can try lowering --max_parallel_requests to a smaller number.\n"
        " To use this script, you need access to the OpenAI API. Make sure your API key is set as an environment"
        " variable OPENAI_API_KEY (or use --api_key). Note: THIS WILL INCUR COSTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument("fin", help="Reformatted dictionary in TSV format")
    cparser.add_argument(
        "lang_prompt",
        help="Language description to be used in the prompt. This can just be 'Dutch' but also more elaborate, e.g."
        " 'Flemish, the variant of Dutch spoken in Flanders'",
    )
    cparser.add_argument("-m", "--model", default="gpt-3.5-turbo", help="Chat model to use")
    cparser.add_argument(
        "-j",
        "--max_parallel_requests",
        default=6,
        type=int,
        help="Max. parallel requests to query. Lower this if you are getting RateLimit issues.",
    )
    cparser.add_argument(
        "--api_key",
        help="OpenAI API key. By default the environment variable 'OPENAI_API_KEY' will be used, but you can override"
        " that with this option.",
    )
    cparser.add_argument("-i", "--first_n", default=None, type=int, help="For debugging: only translate first n items")
    translate_with_openai(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
