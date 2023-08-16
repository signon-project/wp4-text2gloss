import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

import pandas as pd
import requests
from text2gloss.utils import send_request
from tqdm import tqdm
from urllib3.exceptions import RequestError


def get_num_lines(fname: str) -> int:
    with open(fname, "rbU") as fhin:
        return sum(1 for _ in fhin)


def generate_glosses(text: str, port: int = 5000, min_length: int = 3, max_workers: int = 4):
    pfin = Path(text)
    if pfin.exists():

        def process_file(_pfin):
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            pfgloss = _pfin.with_stem(f"{_pfin.stem}-gloss")
            pfaligned = _pfin.with_stem(f"{_pfin.stem}-aligned")
            num_lines = get_num_lines(_pfin)
            print(f"Input {_pfin} is a file. Output will be written to {pfgloss}")
            # Maud's code sometimes adds "+++" and "sweep", but I am not sure what these are. They seem like
            # sign instructions or pluralization. We do not include those here
            # Also only include upper-case tokens. Sometimes Maud's code fails, e.g. `eentje` -> `ééntj`
            n_success = 0
            processed = set()
            with _pfin.open(encoding="utf-8") as fhin, pfgloss.open("w", encoding="utf-8") as fhout, pfaligned.open(
                "w", encoding="utf-8"
            ) as fhaligned:
                for line_idx, line in tqdm(
                    enumerate(fhin, 1), total=num_lines, leave=False, desc=f"Processing {_pfin.stem}", unit="line"
                ):
                    uniq_hash = hashlib.sha256(line.encode(encoding="utf-8"))
                    if uniq_hash in processed:
                        continue

                    processed.add(uniq_hash)

                    line = line.strip()
                    if not line or len(line.split()) < min_length:
                        continue

                    avail_retries = 3
                    while avail_retries > 0:
                        try:
                            response = send_request("rb_text2gloss", port=port, session=session, params={"text": line})
                            break
                        except Exception:
                            session.close()
                            session = requests.Session()
                            session.headers.update({"Content-Type": "application/json"})
                            sleep(30)
                            avail_retries -= 1

                    if avail_retries == 0:
                        print(f"Processing {_pfin} failed on line {line_idx:,}")
                        break

                    if response and "glosses" in response:
                        glosses = [
                            fixed_gloss
                            for gloss in response["glosses"]
                            if (fixed_gloss := gloss.replace("wg+++", "").replace("sweep", "").strip())
                            and gloss.isupper()
                        ]
                        if glosses and len(glosses) >= min_length:
                            glosses = " ".join(glosses)
                            fhout.write(glosses + "\n")
                            fhaligned.write(line + "\n")
                            n_success += 1

            session.close()

            gloss_lines = pfgloss.read_text(encoding="utf-8").splitlines()
            aligned_lines = pfaligned.read_text(encoding="utf-8").splitlines()

            df = pd.DataFrame(zip(aligned_lines, gloss_lines), columns=["text", "gloss"])
            pfout = pfgloss.with_stem(pfgloss.stem.replace("-gloss", "-text+gloss")).with_suffix(".tsv")

            print(f"Combined output file written to {pfout}")
            df.to_csv(pfout, index=False, sep="\t", encoding="utf-8")

            return n_success

        if pfin.is_file():
            process_file(pfin)
        elif pfin.is_dir():
            paths = [pchild for pchild in pfin.glob("*") if pchild.is_file()]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(process_file, pf): pf for pf in paths}
                for future in as_completed(future_to_path):
                    pf = future_to_path[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print(f"{pf.stem} generated an exception: {exc}")
                    else:
                        if data:
                            print(f"{pf.stem} processed: {data:,} lines glossified!")
    else:
        output = send_request("rb_text2gloss", port=port, params={"text": text})
        print(output)


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        description="'Translates' a given Dutch 'text', file or directory of files to VGT glosses."
        " Assumes that an inference server is running on the given 'port'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument(
        "text",
        help="Dutch text to translate to VGT glosses. If this is an existing file, the whole file will be translated"
        " and the output written to a file that is named the same but whose stem ends in -gloss. Because some"
        " lines can be discarded depending on the min_length, an -aligned file is also created with input lines"
        " that correspond line-to-line with the gloss file. If the item is a valid directory, all items in it"
        " (not recursively) will be translated to glosses. Duplicates will be removed on a per-file basis.",
    )
    cparser.add_argument(
        "--min_length",
        type=int,
        default=3,
        help="Sentences with less than 'min_length' tokens or glosses will not be included (split on white-space).",
    )
    cparser.add_argument(
        "-j",
        "--max_workers",
        type=int,
        default=6,
        help="How many parallel workers to use to process files in directories in parallel",
    )
    cparser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Local port that the inference server is running on.",
    )
    generate_glosses(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
