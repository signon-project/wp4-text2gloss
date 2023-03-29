import json
from pathlib import Path

import penman
import requests

from muss.simplify import ALLOWED_MODEL_NAMES, simplify_sentences


def main(amr_in: str = r"F:\python\amr-to-sl-repr\data\vgt-amr-sentences.txt",
         jsonl_out: str = r"F:\python\amr-to-sl-repr\data\wsd\simplified.jsonl",
         lang: str = "nl"):
    pfin = Path(amr_in)
    lines = pfin.read_text(encoding="utf-8").splitlines()

    with Path(jsonl_out).open("w", encoding="utf-8") as fhout:
        simplified = []
        for graph in penman.iterdecode(lines):
            try:
                smpl = graph.metadata[f"{lang}-simplified"]
                simplified.append(smpl)
            except KeyError:
                continue

        data = [{"text": s, "lang": lang.upper()} for s in simplified]
        response = requests.post("http://nlp.uniroma1.it/amuse-wsd/api/model",
                                 json=data,
                                 headers={"accept": "application/json", "Content-Type": "application/json"})

        response = response.json()

        for item in response:
            fhout.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main(jsonl_out=r"F:\python\amr-to-sl-repr\data\wsd\simplified-nl.jsonl")
