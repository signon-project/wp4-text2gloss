from typing import List, Literal

from amr_to_sl_repr.text2amr import text2penman
from amr_to_sl_repr.vec_similarity import load_fasttext_models
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ft_nl, ft_en = load_fasttext_models()


class FastTextItem(BaseModel):
    token: str
    lang: Literal["Dutch", "English"]


@app.post("/token_exists_in_ft/")
async def get_token_exists_in_ft(item: FastTextItem) -> bool:
    if item.lang == "English":
        return item.token in ft_en
    elif item.lang == "Dutch":
        return item.token in ft_nl


@app.post("/token_vector/")
async def get_token_vector(item: FastTextItem):
    if item.lang == "English":
        return ft_en[item.token].tolist() if item.token in ft_en else None
    elif item.lang == "Dutch":
        return ft_nl[item.token].tolist() if item.token in ft_nl else None


class PenmanItem(BaseModel):
    text: str
    src_lang: Literal["Dutch", "English"]
    quantize: bool = False
    no_cuda: bool = False


@app.post("/penman/")
async def get_penman(item: PenmanItem) -> List[str]:
    return text2penman([item.text], src_lang=item.src_lang, quantize=item.quantize, no_cuda=item.no_cuda)
