import logging
import re
import sys
from typing import Dict, List, Literal

import penman
import requests
from text2gloss.utils import send_request


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
        logging.info("AMR GRAPH")
        logging.info(graph)
        tokens = []
        for source, role, target in graph.triples:
            if source is None or target is None:
                continue

            if role == ":location":
                tokens.append("in")
            elif role == ":instance":
                if target == "amr-unknown":
                    # Questions
                    continue
                target = re.sub(r"(\w)-\d+$", "\\1", target)  # Remove frame/concept sense ID
                tokens.append(target)
            elif role == ":polarity" and target == "-":
                tokens.append("NIET")
            elif role == ":quant":
                # :quant can sometimes occur as precursor to other quant, e.g.:
                #   ('c', ':quant', 'v'): [Push(v)],
                #     ('v', ':instance', 'volume-quantity'): [],
                #     ('v', ':quant', '2'): [],
                # Se we want to ignore the first quant
                if not (len(target) == 1 and target.isalpha()):
                    tokens.append(target)

        logging.info(f"Extracted concepts: {tokens}")
        return tokens


def concepts2glosses(tokens: List[str], dictionary: Dict[str, List[str]], src_sentence: str) -> List[str]:
    """Convert a list of tokens/concepts (extracted from AMR) to glosses by using
    mappings that were collected from the VGT dictionary.

    :param tokens: list of AMR tokens/concepts
    :param dictionary: dictionary of English words to VGT glosses
    :param src_sentence: full input sentence
    :return: a list of glosses
    """
    glosses = []
    skip_extra = 0
    session = requests.Session()

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
                    best_match = send_request(
                        "closest", session=session, params={"text": src_sentence, "candidates": dictionary[token]}
                    )
                    glosses.append(best_match)

                    logging.info(f"Best gloss for {token} (out of {dictionary[token]}): {best_match}")
                except KeyError:
                    glosses.append(token)

    logging.info(f"Glosses: {glosses}")

    return glosses


def run_pipeline(text: str, src_lang: Literal["Dutch", "English"], port: int = 5000, verbose: bool = True):
    response = send_request("text2gloss", port=port, params={"text": text, "src_lang": src_lang})
    if verbose:
        print(response)
    return response


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
    cparser.add_argument("--src_lang", help="Source language", default="English", choices=("English", "Dutch"))
    cparser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Local port that the inference server is running on. Not used when" " '--no_fasttext' is in effect.",
    )

    run_pipeline(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
