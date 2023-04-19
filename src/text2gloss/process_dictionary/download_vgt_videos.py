import logging
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import HTTPError
from tqdm import tqdm


def download_file(url: str, dout: str, fherror: typing.IO, force_overwrite: bool = False) -> typing.Optional[str]:
    """Given a URL to a video, download it to a given output directory. If errors occur, log those
    to the given error file-handle.

    :param url: URL to video to download
    :param dout: output directory
    :param fherror: open file handle to write errors to
    :param force_overwrite: whether to overwrite output files
    :return: a path to the local video
    """
    local_filename = url.split("/")[-1]
    pfout = Path(dout) / local_filename

    if pfout.exists() and not force_overwrite:
        return local_filename

    with requests.get(url, stream=True) as r:
        try:
            r.raise_for_status()
        except HTTPError as exc:
            logging.warning(f"Cannot find {url}: {exc}")
            fherror.write(f"{url}\n")
            fherror.flush()
            return None

        with pfout.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return local_filename


def download_vgt_videos(fin: str, dout: str, ferror: str, force_overwrite: bool = False, num_workers: int = 4):
    df = pd.read_csv(fin, sep="\t")
    Path(ferror).parent.mkdir(exist_ok=True, parents=True)
    Path(dout).mkdir(exist_ok=True, parents=True)

    with open(ferror, "w", encoding="utf-8") as fherror:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_url = {
                executor.submit(download_file, url, dout, fherror, force_overwrite): url for url in df["Video"]
            }

            for future in tqdm(as_completed(future_to_url), total=len(df.index)):
                _ = future.result()
                # If we still want to do anything with the return value (local path to video)


def main():
    import argparse

    cparser = argparse.ArgumentParser(description="Download all the videos from the URLs in the VGT dictionary",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cparser.add_argument("fin", help="VGT dictionary in TSV format. Must have a column called 'Video'")
    cparser.add_argument("dout", help="Output directory to write the videos to")
    cparser.add_argument("ferror", help="Output file to log errors to")
    cparser.add_argument("-f", "--force_overwrite", action="store_true", help="Whether to overwrite files")
    cparser.add_argument("-j", "--num_workers", type=int, default=4,
                         help="How many workers to use to download in parallel")

    download_vgt_videos(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
