import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import penman
import torch.cuda
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from mbart_amr.data.linearization import linearized2penmanstr
from pydantic import BaseSettings
from sentence_transformers import SentenceTransformer, util
from text2gloss.text2amr import get_resources, translate
from text2gloss.utils import standardize_gloss
from transformers import LogitsProcessorList
from typing_extensions import Annotated


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

logging.getLogger("penman").setLevel(logging.WARNING)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Settings(BaseSettings):
    json_vgt_dictionary: str = r"vgt-woordenboek-27_03_2023+openai+wn_transls.json"
    sbert_model_name: str = "sentence-transformers/LaBSE"
    sbert_device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

    mbart_input_lang: Literal["English", "Dutch"] = "English"
    mbart_device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    mbart_quantize: bool = True
    mbart_num_beams: int = 3


settings = Settings()
app = FastAPI()

stransformer = SentenceTransformer(settings.sbert_model_name, device=settings.sbert_device)
amr_model, amr_tokenizer, amr_logitsprocessor = get_resources(
    multilingual=settings.mbart_input_lang != "English",
    quantize=settings.mbart_quantize,
    no_cuda=settings.mbart_device == "cpu",
)
amr_gen_kwargs = {
    "max_length": amr_model.config.max_length,
    "num_beams": settings.mbart_num_beams if settings.mbart_num_beams else amr_model.config.num_beams,
    "logits_processor": LogitsProcessorList([amr_logitsprocessor]),
}

en2glosses = json.loads(Path(settings.json_vgt_dictionary).read_text(encoding="utf-8"))["en2gloss"]


@app.get("/centroid/")
def build_tokens_centroid(
    tokens: Annotated[
        List[str],
        Query(
            title="Tokens",
            description="List of tokens whose vectors to retrieve and average (centroid)",
        ),
    ]
):
    vectors = get_tokens_vectors(tokens)
    return np.mean(vectors, axis=0).tolist()


@lru_cache(128)
def encode_texts(tokens: Tuple[str, ...]):
    return stransformer.encode(tokens)


@app.get("/vectors/")
def get_tokens_vectors(
    tokens: Annotated[
        List[str],
        Query(
            title="Tokens",
            description="List of tokens whose vectors to retrieve",
        ),
    ]
):
    return encode_texts(tuple(tokens)).tolist()


@app.get("/similarity/")
def get_tokens_similarity(
    left_token: Annotated[
        str,
        Query(
            title="First token",
        ),
    ],
    right_token: Annotated[
        str,
        Query(
            title="Second token",
        ),
    ],
):
    vecs = get_tokens_vectors([left_token, right_token])
    return util.cos_sim(vecs[0], vecs[1]).squeeze(dim=0).item()


@app.get("/closest/")
def find_closest(
    text: Annotated[
        str,
        Query(
            title="Anchor token or sentence",
        ),
    ],
    candidates: Annotated[
        List[str],
        Query(
            title="Token candidates",
        ),
    ],
) -> str:
    """If one English concept is linked to multiple VGT glosses, we select the "right" gloss by selecting the one
    whose semantic similarity is closest to the English concept _or_ closer to a given sentence, like the whole input
     sentences. To do so, the gloss is preprocessed though (removing regional variety identifiers -A, -B and removing
      extra information in brackets, lowercased) so that, e.g.,

      'ABU-DHABI(UAR)-A' becomes 'abu-dhabi'. If the gloss does not exist in FastText it is not considered as an option
    (unless it is the only option).

    :param text: concept extracted from AMR or sentence
    :param candidates: list of possible gloss options
    :return: the 'best' gloss
    """
    if len(candidates) == 1:
        return candidates[0]

    candidates_std = [standardize_gloss(gloss).lower() for gloss in candidates]
    vecs = get_tokens_vectors([text] + candidates_std)
    sims = util.cos_sim(vecs[0], vecs[1:]).squeeze(dim=0)
    best_cand_idx = sims.argmax(axis=0).item()

    return candidates[best_cand_idx]


@app.get("/text2gloss/")
def run_pipeline(
    texts: Annotated[
        List[str],
        Query(
            title="Texts to convert to a penman representation",
        ),
    ]
) -> List[List[str]]:
    linearizeds = translate(texts, settings.mbart_input_lang, amr_model, amr_tokenizer, **amr_gen_kwargs)
    penman_strs = [linearized2penmanstr(lin) for lin in linearizeds]
    batch_concepts = [extract_concepts(penman_str) for penman_str in penman_strs]

    glosses = []
    for concepts, text in zip(batch_concepts, texts):
        glosses.append(concepts2glosses(concepts, src_sentence=text))

    return glosses


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


def concepts2glosses(tokens: List[str], src_sentence: str) -> List[str]:
    """Convert a list of tokens/concepts (extracted from AMR) to glosses by using
    mappings that were collected from the VGT dictionary.

    :param tokens: list of AMR tokens/concepts
    :param src_sentence: full input sentence
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
                    best_match = find_closest(text=src_sentence, candidates=en2glosses[token])
                    glosses.append(best_match)

                    logging.info(f"Best gloss for {token} (out of {en2glosses[token]}): {best_match}")
                except KeyError:
                    glosses.append(token)

    logging.info(f"Glosses: {glosses}")

    return glosses
