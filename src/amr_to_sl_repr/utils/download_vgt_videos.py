import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from requests.exceptions import HTTPError

import pandas as pd
import requests

from tqdm import tqdm

error_file = open(r"C:\Python\projects\amr-to-sl-repr\data\download_errors.txt", "w", encoding="utf-8")


def download_file(url, dout, force_overwrite: bool = False):
    local_filename = url.split('/')[-1]
    pfout = Path(dout) / local_filename

    if pfout.exists() and not force_overwrite:
        return local_filename

    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        try:
            r.raise_for_status()
        except HTTPError as exc:
            logging.warning(f"Cannot find {url}: {exc}")
            error_file.write(f"{url}\n")
            error_file.flush()
            return local_filename

        with pfout.open('wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return local_filename


if __name__ == '__main__':
    # Hard-coded sorry-not-sorry
    df = pd.read_csv(r"C:\Python\projects\amr-to-sl-repr\data\vgt-woordenboek-27_03_2023.tsv", sep="\t")

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_url = {executor.submit(download_file, url, r"C:\Python\projects\amr-to-sl-repr\data\videos"): url for url in df["Video"]}

        for future in tqdm(as_completed(future_to_url), total=len(df.index)):
            url = future_to_url[future]
            data = future.result()


error_file.close()
