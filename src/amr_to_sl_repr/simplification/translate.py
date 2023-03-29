import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Union
from collections import Counter

import requests
import wn
from Levenshtein import distance

from muss.simplify import simplify_sentences


def offset2omw_synset(wnet: wn.Wordnet, offset: str) -> Optional[wn.Synset]:
    offset = offset.replace("wn:", "")
    offset = "0" * (9-len(offset)) + offset
    wnid = f"omw-en-{offset[:-1]}-{offset[-1]}"
    wnid_s = None

    try:
        return wnet.synset(wnid)
    except wn.Error:
        if wnid[-1] == "a":
            wnid_s = f"omw-en-{wnid[:-2]}-s"
            try:
                return wnet.synset(wnid_s)
            except wn.Error:
                pass

    logging.warning(f"Could not find offset {offset} ({wnid}{' or ' + wnid_s if wnid_s else ''}) in {wnet._lexicons}")


@lru_cache
def get_amuse_wsd(texts: List[str], lang: str):
    data = [{"text": text, "lang": lang.upper()} for text in texts]
    response = requests.post("http://nlp.uniroma1.it/amuse-wsd/api/model",
                             json=data,
                             headers={"accept": "application/json", "Content-Type": "application/json"})
    return response.json()

def main(texts: Union[str, List[str]], src_lang: str = "en", tgt_lang: str = "nl"):
    """
    text (any language) -> translate src_lang to en -> simplify -> wsd -> translate en synset to tgt_lang
    :param texts:
    :param src_lang:
    :param tgt_lang:
    :return:
    """
    if isinstance(texts, str):
        texts = [texts]

    if src_lang != "en":
        # TODO: Do translation
        texts = texts

    # TODO: this does not perform well
    pred_sentences = simplify_sentences(texts, model_name="muss_en_wikilarge_mined")
    print(pred_sentences)
    exit()
    sents = get_amuse_wsd(texts, "en")
    # Amuse returns English synsets, even if the input language is something else like Dutch
    ewn = wn.Wordnet(f"omw-en:1.4")

    for sent in sents:
        glosses = []
        for token in sent:
            if token["pos"] in ["PROPN", "NUM"]:
                glosses.append(token["lemma"])
            elif token["lemma"].lower() == "niet":
                glosses.append(token["lemma"])
            else:
                # Get the synset from the (English) Open Multilingual WordNet
                # even if the text is in another language, it'll still be OMW-en
                try:
                    ofs = token["wnSynsetOffset"]
                    if ofs == "O":
                        raise KeyError
                except KeyError:
                    continue

                en_syn = offset2omw_synset(ewn, ofs, lang=src_lang)

                # No synset found
                if en_syn is None:
                    continue

                # Find WordNet translations for this English synset
                syns = en_syn.translate(f"omw-{src_lang}:1.4")

                # NO translations found
                if not syns:
                    continue

                # For all possibles words linked to the synsets, get the best one
                # That is, the one with the smallest edit distance compared to the parsed lemma
                candidates = {}
                for syn in syns:
                    for word in syn.words():
                        wn_lemma = word.lemma()
                        candidates[wn_lemma] = distance(wn_lemma, token["lemma"])

                best_candidate = min(candidates, key=candidates.get)
                # TODO: translate this to a gloss via the VGT dictionary!

                glosses.append(best_candidate)

        print(" ".join([t["text"] for t in sent]))
        print(" ".join(glosses))
        print()



if __name__ == '__main__':
    main("The train is arriving at track 5 instead of track 7.")
