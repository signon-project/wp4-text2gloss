import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from os import PathLike
from typing import Union, Dict, List, Optional

import pandas as pd
from transformers import pipeline, Pipeline
from word2word import Word2word

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.DEBUG
)

@dataclass
class Dictionary:
    tsv_path: Union[str, PathLike]
    src_lang: str = "en"
    tgt_lang: str = "nl"  # The same language as the glosses
    gloss2transls: Dict[str, List[str]] = field(default_factory=list, init=False)
    transl2glosses: Dict[str, List[str]] = field(default_factory=list, init=False)
    word2word: Optional[Word2word] = field(default=None, init=False)
    hf_translator: Optional[Pipeline] = field(default=None, init=False)

    def __post_init__(self):
        df = pd.read_csv(self.tsv_path, sep="\t")
        # Intended for use with the VGT woordenboek.tsv
        df = df.iloc[:, [1, 2]]
        df = df.rename(columns={df.columns[0]: "gloss", df.columns[1]: "translation"})

        self.gloss2transls = {r["gloss"]: list(r["translation"].split(", ")) for r in df.to_dict(orient="records")}

        transl2glosses = defaultdict(list)
        for gloss, transls in self.gloss2transls.items():
            for transl in transls:
                transl2glosses[transl].append(gloss)
        self.transl2glosses = dict(transl2glosses)

        try:
            # self.word2word = Word2word(self.src_lang, self.tgt_lang)
            self.word2word = None
        except Exception as exc:
            logging.error(f"{exc} Won't be able to use bilingual dictionary look-up!")
            self.word2word = None

        try:
            self.hf_translator = pipeline(f"translation_{self.src_lang}_to_{self.tgt_lang}")
        except Exception as exc:
            try:
                self.hf_translator = pipeline(model=f"Helsinki-NLP/opus-mt-{self.src_lang}-{self.tgt_lang}",
                                              task="translation")
            except Exception:
                logging.error(f"{exc} Won't be able to use the Hugging Face translator!")
                self.hf_translator = None

        if self.word2word is None and self.hf_translator is None:
            raise ValueError("It would appear that neither the bilingual dictionary 'word2word' nor the translation"
                             " system 'hf_translator' could be initialized, so cannot continue. Check your 'src_lang'"
                             " and 'tgt_lang'")

    def translate_word(self, token) -> str:
        if self.word2word is not None:
            # Try translating the token
            try:
                print("hello")
                logging.debug(self.word2word(token))
                return self.word2word(token)[0]
            except KeyError:
                # Try translating the lower-cased token
                try:
                    logging.debug(self.word2word(token.lower()))
                    return self.word2word(token.lower())[0]
                except KeyError:
                    # Try translating the token with "-" replaced by a space
                    try:
                        logging.debug(self.word2word(token.lower().replace("-", " ")))
                        return self.word2word(token.lower().replace("-", " "))[0]
                    except KeyError:
                        pass

        # If we can't find a match in the dictionary, use brute translation
        if self.hf_translator is not None:
            logging.info(f"Could not find {token} in bilingual dictionary. Translating instead...")
            return self.hf_translator(token, clean_up_tokenization_spaces=True)[0]["translation_text"]

        logging.info(f"Could not find {token} in bilingual dictionary nor could translate successfully."
                     f" Using original token instead.")

        # If even that fails, just return the token itself
        return token

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
