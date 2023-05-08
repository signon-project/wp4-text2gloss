from os import PathLike
from pathlib import Path
from typing import Literal, Union

import pandas as pd
from ftfy import fix_text
from pandas import DataFrame
from pandas._libs import lib


def reformat_common(df: DataFrame, lang_col: Literal["nl"]) -> DataFrame:
    # Only keep gloss, lang, video columns
    df = df.drop(columns=[c for c in df.columns if c not in ("gloss", lang_col, "video")]).reset_index(drop=True)
    # Strip all columns
    df = df.astype(str).apply(lambda x: x.str.strip())
    # Drop rows without gloss or language 'translations'
    df = df.dropna(subset=["gloss", lang_col])
    # Fix potential encoding issues
    df[["gloss", lang_col]] = df[["gloss", lang_col]].applymap(lambda x: fix_text(x))
    # Drop rows where 'gloss' column does not start with a number or capital letter
    df = df[df["gloss"].str.match(r"^[A-Z0-9]")]
    # Merge rows with the exact same gloss (so not regional variants). Meanings and videos are concatenated with ", "
    df = df.groupby("gloss").agg(", ".join).reset_index()
    return df


def reformat_vgt(df) -> DataFrame:
    df = df.rename(columns={df.columns[1]: "gloss", df.columns[2]: "nl", "Video": "video"})
    return reformat_common(df, "nl")


def reformat_ngt(df) -> DataFrame:
    df = df.rename(columns={df.columns[2]: "gloss", df.columns[6]: "nl"})
    return reformat_common(df, "nl")


def reformat_dictionary(fin: Union[str, PathLike], sign_language: Literal["vgt", "ngt"]) -> DataFrame:
    pfin = Path(fin).resolve()
    is_tab_sep = pfin.suffix.lower() in (".txt", ".tsv")
    df = pd.read_csv(fin, encoding="utf-8", sep="\t" if is_tab_sep else lib.no_default)

    if sign_language == "vgt":
        df = reformat_vgt(df)
    elif sign_language == "ngt":
        df = reformat_ngt(df)
    else:
        raise ValueError(f"'sign_language' must be one of 'vgt', 'ngt', but '{sign_language}' was given")

    pfout = pfin.with_stem(pfin.stem + "-reformat").with_suffix(".tsv")
    df.to_csv(pfout, index=False, encoding="utf-8", sep="\t")

    return df


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        description="Extracts only relevant columns and standardizes their names. Specifically, extracts the columns"
        " with glosses and their meaning in the source language (e.g. 'nl' for VGT and NGT). Also extracts"
        " the video column if it is present.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument(
        "fin",
        help="Sign language dictionary. If the extension is .tsv or .txt, will read it as tab-separated format,"
        " otherwise as comma-separated. The output will always be a tab-separated file.",
    )
    cparser.add_argument(
        "-l",
        "--sign_language",
        type=str.lower,
        choices=("vgt", "ngt"),
        required=True,
        help="Which sign language is this? VGT: Flemish Sign Language; NGT: Sign Language of the Netherlands",
    )

    reformat_dictionary(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
