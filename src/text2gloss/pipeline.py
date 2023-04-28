import logging
import re
import sys
from typing import Dict, List, Literal

import numpy as np
import penman
import requests

from text2gloss.utils import standardize_gloss
from text2gloss.vec_similarity import cos_sim, get_token_exists_in_ft, get_vec_from_api


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

logging.getLogger("penman").setLevel(logging.WARNING)


def extract_concepts_from_invalid_penman(penman_str):
    # TODO: probably a regex/string-based extraction
    return []


def extract_concepts(penman_str: str) -> List[str]:
    """Extract concepts from a given penman string
    :param penman_str: AMR in penman format
    :return: a list of concepts (str)
    """
    try:
        graph = penman.decode(penman_str)
    except Exception:
        return extract_concepts_from_invalid_penman(penman_str)
    else:
        print("AMR GRAPH", graph)
        tokens = []
        for source, role, target in graph.triples:
            if source is None or target is None:
                continue

            if role == ":location":
                tokens.append("in")
            elif role == ":instance":
                target = re.sub(r"(\w)-\d+$", "\\1", target)  # Remove frame/concept sense ID
                tokens.append(target)
            elif role == ":polarity" and target == "-":
                tokens.append("NIET")
            elif role == ":quant":
                tokens.append(target)

        return tokens


def find_closest(amr_en_concept: str, gloss_options: List[str]) -> str:
    """If one English concept is linked to multiple VGT glosses, we select the "right" gloss by selecting the one
    whose semantic similarity is closest to the English concept. To do so, the gloss is preprocessed though (removing
    regional variety identifiers -A, -B and removing extra information in brackets, lowercased) so that, e.g.,
    'ABU-DHABI(UAR)-A' becomes 'abu-dhabi'. If the gloss does not exist in FastText it is not considered as an option
    (unless it is the only option).

    :param amr_en_concept: concept extracted from AMR
    :param gloss_options: list of possible gloss options
    :return: the 'best' gloss
    """
    if len(gloss_options) == 1:
        return gloss_options[0]

    gloss_options = [
        gloss for gloss in gloss_options if get_token_exists_in_ft(standardize_gloss(gloss).lower(), "Dutch")
    ]
    gloss_opts_std = [standardize_gloss(gloss).lower() for gloss in gloss_options]

    amr_en_vec = get_vec_from_api(amr_en_concept, "English")
    nl_vecs = [get_vec_from_api(g, "Dutch") for g in gloss_opts_std]
    sims = [cos_sim(amr_en_vec, nl_vec) for nl_vec in nl_vecs]
    best_cand_idx = np.argmax(sims, axis=0)
    print(f"gloss_opts_std: {gloss_opts_std} -- token: {amr_en_concept}")
    print(f"similaritiess: {sims}")

    return gloss_options[best_cand_idx]


def concepts2glosses(tokens: List[str], dictionary: Dict[str, List[str]]) -> List[str]:
    """Convert a list of tokens/concepts (extracted from AMR) to glosses by using
    mappings that were collected from the VGT dictionary.

    :param tokens: list of AMR tokens/concepts
    :param dictionary: dictionary of English words to VGT glosses
    :return: a list of glosses
    """
    glosses = []
    skip_extra = 0
    for token_idx in range(len(tokens)):
        if skip_extra:
            skip_extra -= 1
            continue

        token = tokens[token_idx]
        # We can ignore the special "quantity" identifier
        if token.endswith(("-quantity",)):
            continue
        elif token == "cause":
            glosses.append("[PU]")
        elif token in "i":
            glosses.append("WG-1")
        elif token == "you":
            glosses.append("WG-2")
        elif token in ("he", "she", "they"):
            glosses.append("WG-3")
        elif token == "we":
            glosses.append("WG-4")
        elif token.isdigit():  # Copy digits
            glosses.append(token)
        elif token.startswith('"') and token.endswith('"'):  # Copy literal items but remove quotes
            glosses.append(token[1:-1])
        else:  # Conditions that require info about the next token
            next_token = tokens[token_idx + 1] if token_idx < len(tokens) - 1 else None
            # If this token is "city" and the next token is the city name, we can ignore "city"
            if token in ("city", "station") and next_token and next_token[0].isupper():
                continue
            elif token == "person" and next_token and next_token == "have-rel-role":
                skip_extra = 1
                continue
            else:
                try:
                    best_match = find_closest(token, dictionary[token])
                    glosses.append(best_match)
                except KeyError:
                    glosses.append(token)

    return glosses


def run_pipeline(text: str, src_lang: Literal["Dutch", "English"], port: int = 5000):
    response = requests.post(rf"http://127.0.0.1:{port}/text2gloss/", json={"text": text, "lang": src_lang})
    if response:
        print(response.json())
        return response.json()


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        description="Convert a given sentence into glosses. The pipeline consists of gathering the AMR representation,"
        " extracting concepts from AMR, and then looking up those English concepts in the dictionary for"
        " a corresponding gloss. Some disambiguation may be done through FastText embeddings if one concept"
        " has multiple gloss candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument("text", help="Text to translate to glosses")
    cparser.add_argument("src_lang", help="Source language", choices=("English", "Dutch"))
    cparser.add_argument(
        "dictionary_file",
        help="Path to the VGT dictionary as JSON archive, containing 'nl2gloss'" " and 'en2gloss' keys",
    )
    cparser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Local port that the inference server is running on. Not used when" " '--no_fasttext' is in effect.",
    )

    run_pipeline(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
