import json
import logging
import re
import sys
from os import PathLike
from pathlib import Path
from typing import Dict, List, Literal, Union

import numpy as np
import penman
import requests
from text2gloss.utils import standardize_gloss
from text2gloss.vec_similarity import cos_sim


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def extract_concepts_from_invalid_penman(penman_str):
    # TODO: probably a regex/string-based extraction
    return []


def extract_concepts(penman_str):
    try:
        graph = penman.decode(penman_str)
    except Exception:
        return extract_concepts_from_invalid_penman(penman_str)
    else:
        tokens = []
        for source, role, target in graph.triples:
            if role == ":location":
                tokens.append("in")

            if role == ":instance":
                target = re.sub(r"(\w)-\d+$", "\\1", target)  # Remove frame/concept sense ID
                tokens.append(target)

        return tokens


def find_closest(amr_en_concept: str, gloss_options: List[str]):
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
    glosses = []
    for token_idx in range(len(tokens)):
        token = tokens[token_idx]
        # We can ignore the special "quantity" identifier
        if token.endswith(("-quantity",)):
            continue
        elif token == "cause":
            glosses.append("[PU]")
        elif token == "i":
            glosses.append("Wg1")
        elif token == "you":
            glosses.append("Wg2")
        elif token in ("he", "she", "they"):
            glosses.append("Wg3")
        elif token.isdigit():  # Copy digits
            glosses.append(token)
        elif token.startswith('"') and token.endswith('"'):  # Copy literal items but remove quotes
            glosses.append(token[1:-1])
        else:  # Conditions that require info about the next token
            next_token = tokens[token_idx + 1] if token_idx < len(tokens) - 1 else None
            # If this token is "city" and the next token is the city name, we can ignore "city"
            if token in ("city", "station") and next_token and next_token[0].isupper():
                continue
            else:
                try:
                    best_match = find_closest(token, dictionary[token])
                    glosses.append(best_match)
                except KeyError:
                    glosses.append(token)

    return glosses


def get_penman_from_api(text: str, src_lang: Literal["English", "Dutch"]):
    # TODO: for development only!
    response = requests.post(r"http://127.0.0.1:5000/penman", json={"text": text, "src_lang": src_lang})
    return response.json()


def main(text: str, src_lang: Literal["Dutch", "English"], dictionary_file: Union[str, PathLike] = None):
    en2glosses = json.loads(Path(dictionary_file).read_text(encoding="utf-8"))["en2gloss"]
    penman_str = get_penman_from_api(text, src_lang=src_lang)[0]
    print(penman_str)
    concepts = extract_concepts(penman_str)
    print(concepts)

    if concepts:
        glosses = concepts2glosses(concepts, en2glosses)
        print(glosses)


if __name__ == "__main__":
    main(
        "The train to Antwerp-Central is arriving on track 10",
        src_lang="English",
        dictionary_file=r"/data/vgt-woordenboek-27_03_2023+en_transls.json",
    )
