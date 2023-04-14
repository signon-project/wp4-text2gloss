import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import requests
from gensim.models.keyedvectors import load_word2vec_format
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm


def maybe_download_ft_vectors(download_nl: bool = True, download_en: bool = True) -> Tuple[Path, Path]:
    """Download the fastText aligned vectors if they are not on disk yet. Will download to models/fasttext in the root
    directory of this repository.
    :return: a tuple of Paths to the Dutch and English downloaded vectors (.vec files)
    """
    dout = Path(__file__).resolve().parents[2].joinpath("models/fasttext")
    dout.mkdir(exist_ok=True, parents=True)

    def download(url: str):
        fname = url.split("/")[-1]
        pfout = dout.joinpath(fname)

        if pfout.exists():
            return pfout

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with pfout.open("wb") as fhout:
            with tqdm(
                    total=total_size / (32 * 1024.0),
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {pfout.name}",
            ) as pbar:
                for data in response.iter_content(32 * 1024):
                    fhout.write(data)
                    pbar.update(len(data))

        return pfout

    url_nl = "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.nl.align.vec"
    url_en = "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec"

    local_nl = download(url_nl) if download_nl else None
    local_en = download(url_en) if download_en else None

    return local_nl, local_en


def cos_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors
    :param vec_a: vector a
    :param vec_b: vector b
    :return: cosine similarity between [-1, 1]
    """
    return dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))


def load_fasttext_models(load_nl: bool = True, load_en: bool = True):
    ft_nl_path, ft_en_path = maybe_download_ft_vectors(download_nl=load_nl, download_en=load_en)

    logging.info("Loading fastText word vectors")
    ft_nl = load_word2vec_format(str(ft_nl_path), binary=False) if ft_nl_path else None
    ft_en = load_word2vec_format(str(ft_en_path), binary=False) if ft_en_path else None

    return ft_nl, ft_en
