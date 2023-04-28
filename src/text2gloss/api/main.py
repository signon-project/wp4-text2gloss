import json
from functools import lru_cache
from pathlib import Path
from traceback import print_exception
from typing import List, Literal, Tuple, Union

import torch.cuda
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from sentence_transformers import SentenceTransformer, util
from text2gloss.pipeline import concepts2glosses, extract_concepts
from text2gloss.text2amr import get_resources, text2penman
from text2gloss.utils import standardize_gloss
from typing_extensions import Annotated


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
    sbert_device: str = "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
app = FastAPI()

stransformer = SentenceTransformer(settings.sbert_model_name, device=settings.sbert_device)
stransformer = SentenceTransformer(settings.sbert_model_name, device=settings.sbert_device)
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
    return vectors.mean(axis=0)


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
    return encode_texts(tuple(tokens))


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


@app.get("/penman/")
def get_penman(
    text: Annotated[
        str,
        Query(
            title="Text to convert to a penman representation",
        ),
    ],
    src_lang: Annotated[
        Literal["English", "Dutch"],
        Query(
            title="Language of the given text",
        ),
    ],
    quantize: Annotated[
        bool,
        Query(
            title="Whether to quantize the MBART model (faster)",
        ),
    ] = True,
    no_cuda: Annotated[
        bool,
        Query(
            title="Whether to disable CUDA",
        ),
    ] = True,
) -> List[str]:
    try:
        return text2penman([text], src_lang=src_lang, quantize=quantize, no_cuda=no_cuda)
    except Exception as exc:
        print_exception(exc)
        raise HTTPException(status_code=500, detail="Internal server error when generating penman AMR representation.")


@app.get("/text2gloss/")
def run_pipeline(
    text: Annotated[
        str,
        Query(
            title="Text to convert to a penman representation",
        ),
    ],
    src_lang: Annotated[
        Literal["English", "Dutch"],
        Query(
            title="Language of the given text",
        ),
    ],
    quantize: Annotated[
        bool,
        Query(
            title="Whether to quantize the MBART model (faster)",
        ),
    ] = True,
    no_cuda: Annotated[
        bool,
        Query(
            title="Whether to disable CUDA",
        ),
    ] = False,
) -> List[str]:
    penman_str = get_penman(text, src_lang, quantize=quantize, no_cuda=no_cuda)[0]
    concepts = extract_concepts(penman_str)

    if concepts:
        return concepts2glosses(concepts, en2glosses, src_sentence=text)
    else:
        raise HTTPException(
            status_code=204,
            detail="No concepts could be extracted from the AMR so no glosses could be generated either.",
        )
