from pathlib import Path
from typing import Literal

from text2gloss.utils import send_request


def run_pipeline(text: str, src_lang: Literal["English", "Dutch", "Spanish"], sign_lang: Literal["vgt", "ngt"] = "vgt", port: int = 5000, verbose: bool = True):
    pfin = Path(text)

    if pfin.exists() and pfin.is_file():
        texts = pfin.read_text(encoding="utf-8").splitlines()
        all_glosses = []
        for text in texts:
            glosses = send_request("text2gloss", port=port, params={"text": text, "sign_lang": sign_lang, "src_lang": src_lang})
            if verbose:
                print("TEXT:", text)
                print(f"{sign_lang.upper()}:", " ".join(glosses["glosses"]))
                print("META", glosses["meta"])
                print()
            all_glosses.append(glosses)

        glosses = all_glosses
    else:
        glosses = send_request("text2gloss", port=port, params={"text": text, "sign_lang": sign_lang, "src_lang": src_lang})
        if verbose:
            print("TEXT:", text)
            print(f"{sign_lang.upper()}:", " ".join(glosses["glosses"]))
            print("META", glosses["meta"])

    return glosses


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        description="Convert a given sentence into glosses. The pipeline consists of gathering the AMR representation,"
        " extracting concepts from AMR, and then looking up those English concepts in the dictionary for"
        " a corresponding gloss. Some disambiguation may be done through LABSE embeddings if one concept"
        " has multiple gloss candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument(
        "text",
        help="Text to translate to glosses. If a file is given, all of its lines will be translated separately.",
    )

    cparser.add_argument(
        "src_lang",
        choices=("English", "Dutch", "Spanish"),
        help="Spoken language to translate from",
    )

    cparser.add_argument(
        "sign_lang",
        choices=("vgt", "ngt"),
        default="vgt",
        help="Sign language (gloss representation) to translate to",
    )

    cparser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Local port that the inference server is running on",
    )

    run_pipeline(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
