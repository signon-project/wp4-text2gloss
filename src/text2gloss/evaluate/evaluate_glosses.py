import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import comet
import pandas as pd
import torch
import typer
from bleurt import score as bleurt_scorer
from comet import download_model as download_comet
from comet import load_from_checkpoint as load_comet
from sacrebleu.metrics import BLEU, CHRF, TER
from tqdm import trange
from typer import Argument, Option


app = typer.Typer()


def calculate_bleurt(
    predictions: list[str],
    references: list[str],
    batch_size: int = 4,
    bleurt_checkpoint: Path = Path(__file__).parents[3] / "bleurt" / "checkpoints" / "BLEURT-20",
) -> tuple[list, str]:
    """
    Calculate BLEURT score for each pair of predicted and gold glosses.
    :param predictions: predicted glosses
    :param references: gold glosses
    :param batch_size: batch size
    :param bleurt_checkpoint: path to the BLEURT checkpoint
    :return: a list of BLEURT scores
    """
    print("Calculating BLEURT scores")
    bleurt_checkpoint = Path(bleurt_checkpoint).resolve()
    if not bleurt_checkpoint.exists():
        raise ValueError(
            f"BLEURT checkpoint {bleurt_checkpoint} does not exist. By default, it is expected to be"
            f" inside the cloned BLEURT repository at the top level of the text2gloss library, in a newly"
            f" created subfolder `checkpoints`. I.e. 'text2gloss/bleurt/checkpoints/BLEURT-20'."
            f" You can download and unpack it from"
            f" https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip"
        )

    bleurt_metric = bleurt_scorer.BleurtScorer(bleurt_checkpoint)
    bleurt_scores = []
    # Use external batch loop so that we can add a progress bar
    for batch_idx in trange(0, len(predictions), batch_size, desc="Calculating BLEURT scores"):
        batch_references = references[batch_idx : batch_idx + batch_size]
        batch_predictions = predictions[batch_idx : batch_idx + batch_size]
        bleurt_scores.extend(
            bleurt_metric.score(references=batch_references, candidates=batch_predictions, batch_size=batch_size)
        )

    signature = f"ckpt:{bleurt_checkpoint.stem}|v:{version('bleurt')}"
    return bleurt_scores, signature


def calculate_comet(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    batch_size: int = 8,
) -> tuple[list, str]:
    """
    Calculate COMET score for each pair of predicted and gold glosses.
    :param sources: source sentences
    :param predictions: predicted glosses
    :param references: gold glosses
    :param batch_size: batch size
    :return: a list of COMET scores
    """
    print("Calculating COMET scores")
    data = [{"src": src, "mt": pred, "ref": ref} for src, pred, ref in zip(sources, predictions, references)]
    model_path = download_comet("Unbabel/wmt22-comet-da")
    comet_metric = load_comet(model_path)

    # On Windows, if num_workers is None, we get all zero scores. Specifically disabling multiprocessing will fix this.
    comet_scores = comet_metric.predict(
        data,
        batch_size=batch_size,
        gpus=1 if torch.cuda.is_available() else 0,
        num_workers=0 if os.name == "nt" else None,
    ).scores

    signature = f"ckpt:wmt22-comet-da|v:{comet.__version__}"
    return comet_scores, signature


def calculate_sacrebleu(
    predictions: list[str],
    references: list[str],
) -> dict[str, Any]:
    """
    Calculate SacreBLEU score for each pair of predicted and reference glosses.
    :param predictions: predicted glosses
    :param references: gold glosses
    :return: a list of SacreBLEU scores
    """
    print("Calculating SacreBLEU scores")
    references = [references]

    chrf_metric = CHRF()
    ter_metric = TER()

    results = {
        "chrf": chrf_metric.corpus_score(predictions, references).score,
        "chrf_signature": chrf_metric.get_signature().format(short=True),
        "ter": ter_metric.corpus_score(predictions, references).score,
        "ter_signature": ter_metric.get_signature().format(short=True),
    }

    for ngram_order in range(1, 5):
        bleu_metric = BLEU(max_ngram_order=ngram_order)
        results[f"bleu{ngram_order}"] = bleu_metric.corpus_score(predictions, references).score
        results[f"bleu{ngram_order}_signature"] = bleu_metric.get_signature().format(short=True)

    return results


@app.command()
def evaluate_glosses(
    pred_f: Annotated[
        Path,
        Argument(
            help="text file containing predicted glosses",
            file_okay=True,
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    ref_f: Annotated[
        Path,
        Argument(
            help="text file containing reference glosses",
            file_okay=True,
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    src_f: Annotated[
        Path,
        Argument(
            help="text file containing source text (writte, spoken language)",
            file_okay=True,
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    corpus_output_f: Annotated[
        Path,
        Argument(help="output file to write corpus results to as a JSON file", exists=False),
    ],
    sentence_output_f: Annotated[
        Path,
        Argument(help="output file to write sentence-level results to as a CSV", exists=False),
    ],
    batch_size: Annotated[int, Option(help="batch size to evaluate neural metrics")] = 4,
    lower_case: Annotated[bool, Option(help="lower case glosses")] = True,
):
    """
    Evaluate given glosses against reference glosses.
    """
    predictions = Path(pred_f).read_text(encoding="utf-8").splitlines()
    references = Path(ref_f).read_text(encoding="utf-8").splitlines()
    sources = Path(src_f).read_text(encoding="utf-8").splitlines()

    if lower_case:
        predictions = [p.lower() for p in predictions]
        references = [r.lower() for r in references]

    if len(predictions) != len(references) != len(sources):
        raise ValueError(
            f"Number of predicted glosses ({len(predictions)}), reference glosses ({len(references)}) and source texts"
            f" ({len(sources)}) must be equal."
        )

    bleurt_scores, bleurt_sig = calculate_bleurt(predictions=predictions, references=references, batch_size=batch_size)
    bleurt_score = sum(bleurt_scores) / len(bleurt_scores)

    comet_scores, comet_sig = calculate_comet(
        sources=sources, predictions=predictions, references=references, batch_size=batch_size
    )
    comet_score = sum(comet_scores) / len(comet_scores)
    sacrebleu_results = calculate_sacrebleu(predictions=predictions, references=references)

    corpus_results = {
        "bleurt": bleurt_score,
        "bleurt_signature": bleurt_sig,
        "comet": comet_score,
        "comet_signature": comet_sig,
        **sacrebleu_results,
    }

    Path(corpus_output_f).write_text(json.dumps(corpus_results, indent=4), encoding="utf-8")

    sentence_results = {
        "sources": sources,
        "references": references,
        "predictions": predictions,
        "bleurt": bleurt_scores,
        "comet": comet_scores,
    }
    df_sentence = pd.DataFrame(sentence_results)
    df_sentence.to_csv(sentence_output_f, index=False, encoding="utf-8")
