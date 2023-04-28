import json
import os
from pathlib import Path
from traceback import print_exception
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, BaseSettings

from text2gloss.pipeline import extract_concepts, concepts2glosses
from text2gloss.text2amr import text2penman, get_resources
from text2gloss.vec_similarity import load_fasttext_models


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi import FastAPI, HTTPException


# TODO: use sentence transformers
# TODO use Annotated. e.g. to add descriptions and titles, see: https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#deprecating-parameters


class Settings(BaseSettings):
    load_ft: bool = False  # FastText is only needed during pre-processing, not during inference
    load_mbart: bool = False
    json_vgt_dictionary: str = r"vgt-woordenboek-27_03_2023+openai+wn_transls.json"


settings = Settings()
app = FastAPI()

ft_nl, ft_en = load_fasttext_models() if settings.load_ft else (None, None)

_ = get_resources(multilingual=False) if settings.load_mbart else None  # Preload MBART once
en2glosses = json.loads(Path(settings.json_vgt_dictionary).read_text(encoding="utf-8"))["en2gloss"] if settings.load_mbart else None


class FastTextItem(BaseModel):
    token: str
    lang: Literal["Dutch", "English"]


@app.post("/token_exists_in_ft/")
def get_token_exists_in_ft(item: FastTextItem) -> bool:
    if item.lang == "English":
        if ft_en is None:
            raise HTTPException(status_code=503, detail="English FastText is not available.")
        else:
            return item.token in ft_en
    elif item.lang == "Dutch":
        if ft_nl is None:
            raise HTTPException(status_code=503, detail="Dutch FastText is not available.")
        else:
            return item.token in ft_nl


@app.post("/token_vector/")
def get_token_vector(item: FastTextItem):
    if item.lang == "English":
        if ft_en is None:
            raise HTTPException(status_code=503, detail="English FastText is not available.")
        else:
            return ft_en[item.token] if item.token in ft_en else None
    elif item.lang == "Dutch":
        if ft_nl is None:
            raise HTTPException(status_code=503, detail="Dutch FastText is not available.")
        else:
            return ft_nl[item.token] if item.token in ft_nl else None


class PenmanItem(BaseModel):
    text: str
    src_lang: Literal["Dutch", "English"]
    quantize: bool = True
    no_cuda: bool = False


@app.post("/penman/")
def get_penman(item: PenmanItem) -> List[str]:
    try:
        return text2penman([item.text], src_lang=item.src_lang, quantize=item.quantize, no_cuda=item.no_cuda)
    except Exception as exc:
        print_exception(exc)
        raise HTTPException(status_code=500, detail="Internal server error when generating penman AMR representation.")


@app.post("/text2gloss/")
def run_pipeline(item: PenmanItem):
    penman_str = get_penman(item)[0]
    concepts = extract_concepts(penman_str)

    if concepts:
        return concepts2glosses(concepts, en2glosses)
    else:
        raise HTTPException(status_code=204, detail="No concepts could be extracted from the AMR so no glosses could be generated either.")
