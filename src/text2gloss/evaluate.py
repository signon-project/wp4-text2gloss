from enum import Enum
from math import ceil
from pathlib import Path
from typing import Annotated, Optional

import typer
from text2gloss.utils import send_request
from tqdm import tqdm
from typer import Argument, Option


app = typer.Typer()


class SignLang(str, Enum):
    vgt = "vgt"
    ngt = "ngt"


class SpokenLang(str, Enum):
    english = "English"
    dutch = "Dutch"
    spanish = "Spanish"


def filter_unsupported(input_glossed_sents: list[str], supported_glosses: set[str]) -> list[int]:
    """
    Filter out sentences whose glosses are not supported.
    :param input_glossed_sents: a list of glossed sentences
    :param supported_glosses: a list of supported glosses
    :return: a list of indices of sentences whose glosses are supported
    """
    filtered_idxs = []
    for sent_idx, sentence in enumerate(input_glossed_sents):
        glosses = sentence.split()
        if all(gloss in supported_glosses for gloss in glosses):
            filtered_idxs.append(sent_idx)

    return filtered_idxs


def batchify(iterable: list, batch_size: int = 8) -> list:
    """
    Batchify an iterable.
    :param iterable: an iterable
    :param batch_size: batch size
    :return: a list of batches
    """
    return [iterable[idx : idx + batch_size] for idx in range(0, len(iterable), batch_size)]


@app.command()
def evaluate_glosses(
    text_file: Annotated[
        Path,
        Argument(help="text file containing sentences", file_okay=True, exists=True, readable=True, resolve_path=True),
    ],
    gloss_file: Annotated[
        Path,
        Argument(help="text file containing glosses", file_okay=True, exists=True, readable=True, resolve_path=True),
    ],
    output_file: Annotated[
        Path,
        Argument(help="output file to write glosses to", exists=False),
    ],
    sign_lang: Annotated[SignLang, Option(help="which sign language to generate glosses for")] = SignLang.vgt,
    src_lang: Annotated[SpokenLang, Option(help="language of the input")] = SpokenLang.dutch,
    supported_glosses_file: Annotated[
        Optional[Path],
        Option(
            help="a file containing supported glosses, one gloss per line",
            file_okay=True,
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    only_use_supported: Annotated[
        bool, Option(help="whether to filter out sentences whose glosses are not supported")
    ] = False,
    batch_size: Annotated[int, Option(help="batch size to process text2amr with")] = 4,
    port: Annotated[int, Option(help="port where the inference server is running on")] = 5000,
):
    """
    Generate glosses for a given text file.
    """
    input_sentences = Path(text_file).read_text(encoding="utf-8").splitlines()
    input_glossed_sents = Path(gloss_file).read_text(encoding="utf-8").splitlines()

    if supported_glosses_file:
        supported_glosses = set(Path(supported_glosses_file).read_text(encoding="utf-8").splitlines())
        filtered_idxs = filter_unsupported(input_glossed_sents, supported_glosses)
        filtered_glossed_sents = [input_glossed_sents[idx] for idx in filtered_idxs]

        if only_use_supported:
            print("Filtered out", len(input_glossed_sents) - len(filtered_glossed_sents), "sentences")
            print(
                f"Filtered out {len(input_glossed_sents) - len(filtered_glossed_sents):,}"
                f" out of {len(input_glossed_sents):,} sentences!"
            )
            input_glossed_sents = filtered_glossed_sents
            input_sentences = [input_sentences[idx] for idx in filtered_idxs]
        else:
            print(
                f"Warning! {len(input_glossed_sents) - len(filtered_glossed_sents):,} out of"
                f" {len(input_glossed_sents):,} sentences have one or more unsppported glosses. Will use all of these!"
                f" If you want to drop sentences with glosses that are not supported, use the"
                f" --only-use-supported flag."
            )

    with Path(output_file).parent.joinpath("raw_predictions.txt").open("w", encoding="utf-8") as fhout:
        # Predict new glosses
        predictions = []
        for batch_input_sentences in tqdm(
            batchify(input_sentences, batch_size=batch_size), total=ceil(len(input_sentences) / batch_size)
        ):
            output = send_request(
                "batch_text2gloss",
                port=port,
                params={"texts": batch_input_sentences, "sign_lang": sign_lang, "src_lang": src_lang},
            )

            if output is not None:
                batch_preds = [" ".join(glosses) for glosses in output["glosses"]]
                predictions.extend(batch_preds)

                for glosses, meta in zip(output["glosses"], output["meta"]):
                    print(f"Text: {meta['text']}")
                    print(f"Preds: {' '.join(glosses)}")
                    print()

                    fhout.write(f"Text: {meta['text']}\n")
                    fhout.write(f"Preds: {' '.join(glosses)}\n")
                    fhout.write(meta['penman_str'])
                    fhout.write("\n\n")

    Path(output_file).write_text("\n".join(predictions), encoding="utf-8")
    Path(output_file).parent.joinpath("gold_glosses.txt").write_text("\n".join(input_glossed_sents), encoding="utf-8")
    Path(output_file).parent.joinpath("gold_sents.txt").write_text("\n".join(input_sentences), encoding="utf-8")
