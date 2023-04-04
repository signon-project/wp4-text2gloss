import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from os import PathLike
from typing import Union, Dict, List, Optional

import pandas as pd
from transformers import pipeline, Pipeline
import wn

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.DEBUG
)

# TODO: move translation and wordnet stuff to other script so that we have one script to preprocess the tsv ONCE
# and then this script to just load it


logger = logging.getLogger("amr")


def standardize_gloss(gloss: str) -> str:
    gloss = re.sub(r"-[A-Z]$", "", gloss)
    gloss = re.sub(r"\([^)]+\)$", "", gloss)
    return gloss


def gloss2str(gloss: str) -> str:
    gloss = standardize_gloss(gloss)
    gloss = gloss.replace("-", " ")

    return gloss.lower()

@lru_cache
def translate_word_to_en(translator, token) -> str:
    return translator(token, clean_up_tokenization_spaces=True)[0]["translation_text"]

@dataclass
class Dictionary:
    tsv_path: Union[str, PathLike]
    gloss2transls: Dict[str, List[str]] = field(default_factory=list, init=False)
    transl2glosses: Dict[str, List[str]] = field(default_factory=list, init=False)
    hf_translator: Optional[Pipeline] = field(default=None, init=False)

    def __post_init__(self):
        df = pd.read_csv(self.tsv_path, sep="\t")
        # Intended for use with the VGT woordenboek.tsv
        df = df.iloc[:, [1, 2]]
        self.df = df.rename(columns={df.columns[0]: "gloss", df.columns[1]: "nl"})

        try:
            self.hf_translator = pipeline("translation_nl_to_en")
        except Exception as exc:
            try:
                self.hf_translator = pipeline(model=f"Helsinki-NLP/opus-mt-nl-en", task="translation")
            except Exception:
                logger.error(f"{exc} Won't be able to use the Hugging Face translator so we do not have a back-off!")
                self.hf_translator = None

        self.add_wn_en_translations()

    def add_wn_en_translations(self):
        nwn = wn.Wordnet("omw-nl:1.4")

        def wn_translate(nl: str):
            # One gloss has multiple Dutch "translations"
            # For each word, we find all of its possible translations through open multilingual Wordnet
            nl = nl.split(", ")
            en_translations = set()
            for word in nl:
                nl_words = nwn.words(word)

                if nl_words:
                    for nl_word in nl_words:
                        for _, en_words in nl_word.translate("omw-en:1.4").items():
                            en_translations.update([en_w.lemma() for en_w in en_words])
                else:
                    # TODO: use DEEPL instead
                    pass

            return ", ".join(sorted(en_translations))

        self.df["en"] = self.df["nl"].apply(wn_translate)

    @cached_property
    def dict_gloss2nl(self):
        # Gloss (str) -> Dutch "translations" (List[str])
        gloss2nl = defaultdict(set)
        for r in self.df.to_dict(orient="records"):
            gloss = standardize_gloss(r["gloss"])
            nl = r["nl"].split(", ")
            gloss2nl[gloss].update(nl)
        gloss2nl = {gloss: sorted(nl) for gloss, nl in gloss2nl.items()}

        return gloss2nl

    @cached_property
    def dict_nl2gloss(self):
        # Dutch "translation" (str) -> corresponding glosses (List[str])
        nl2gloss = defaultdict(set)
        for gloss, nl in self.dict_gloss2nl.items():
            for transl in nl:
                nl2gloss[transl].update(gloss)

        nl2gloss = {nl: sorted(glosses) for nl, glosses in nl2gloss.items()}
        return nl2gloss

    @cached_property
    def dict_gloss2en(self):
        # Gloss (str) -> English "translations" (List[str])
        gloss2en = defaultdict(set)
        for r in self.df.to_dict(orient="records"):
            gloss = standardize_gloss(r["gloss"])
            en = r["en"].split(", ")
            gloss2en[gloss].update(en)
        gloss2en = {gloss: sorted(en) for gloss, en in gloss2en.items()}

        return gloss2en

    @cached_property
    def dict_en2gloss(self):
        # English "translation" (str) -> corresponding glosses (List[str])
        en2gloss = defaultdict(set)
        for gloss, en in self.dict_gloss2en.items():
            for transl in en:
                en2gloss[transl].update(gloss)

        en2gloss = {en: sorted(glosses) for en, glosses in en2gloss.items()}
        return en2gloss

    def word2gloss(self, token) -> str:
        transl = self.translate_word(token)

        print("TOKEN, TRANSL", token, transl)
        try:
            return self.transl2glosses[token][0]
        except KeyError:
            return transl

    def sequence2gloss(self, tokens: Union[str, List[str]]) -> str:
        if isinstance(tokens, str):
            tokens = tokens.split()

        transls = [self.word2gloss(token) for token in tokens]

        return " ".join(transls)

if __name__ == '__main__':
    dic = Dictionary(r"F:\python\amr-to-sl-repr\data\vgt-woordenboek-27_03_2023.tsv")
    print(dic.df.head(25))
