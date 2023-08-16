import logging
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
import wn
from pandas import DataFrame, Series
from sentence_transformers import util
from sqlalchemy import create_engine
from text2gloss.utils import send_request, standardize_gloss
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d %H:%M:%S",
    filename=Path(__file__).parent.joinpath("preprocess.log"),
    level=logging.INFO,
)
# Set up logging to console in addition to logging to file
console = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
logger = logging.getLogger("text2gloss")


def add_en_translations(df: DataFrame, lang_col: str):
    """Add "translations" to the dataframe in the "en" column. These translations are retrieved
    by looking up the "possible translations" in the lang_col column in Open Multilingual Wordnet
    and finding their equivalent in the English WordNet. This means these English translations will be _very_ broad.

    :param df: input Dataframe that must have a `lang_col` column (e.g. 'nl_vgt')
    :param lang_col: key to the language column with 'explanations' (possible translations)
    :return: updated DataFrame that now also includes an "en" column
    """
    lang_code = lang_col.split("_")[0]
    nwn = wn.Wordnet(f"omw-{lang_code}:1.4")

    @lru_cache(maxsize=256)
    def translate_word(src_word: str):
        en_translations = set()

        if src_word[0].isupper():
            # Probably a city, country or name
            en_translations.add(src_word)
        else:
            wn_words = nwn.words(src_word)

            if wn_words:
                for wn_word in wn_words:
                    for _, en_wn_words in wn_word.translate("omw-en:1.4").items():
                        en_translations.update([en_w.lemma() for en_w in en_wn_words])

        return en_translations

    def translate(row: Series):
        src_words_str = row[lang_col]
        # One gloss has multiple "possible translations" (explanation words)
        # For each word, we find all of its possible translations through open multilingual Wordnet
        if not src_words_str:
            return ""

        src_words_str = remove_brackets(src_words_str)
        src_words = src_words_str.split(", ")

        en_translations = set(row["en"].split(", ")) if "en" in row else set()

        for src_word in src_words:
            if src_word := src_word.strip():
                en_translations.update(translate_word(src_word))

        return ", ".join(sorted(en_translations))

    tqdm.pandas(desc=f"Translating {lang_col.upper()} 'translations' to EN with WordNet")
    df["en"] = df.progress_apply(translate, axis=1)

    return df


def remove_brackets(word: str):
    """Remove ending brackets from a word. Sometimes, the "possible translations" end in brackets to further
    specify the context of the word. But for looking things up in WordNet, that will lead to issues.

    :param word: a given string
    :return: the modified string where open/closing brackets and anything between are removed
    """
    word = re.sub(r"\([^)]*\)", "", word)
    word = " ".join(word.split())  # fix white-spaces
    return word


def filter_en_translations(df: DataFrame, lang_col: str, threshold: float = 0.5, port: int = 5000):
    """Uses LABSE embeddings and calculates the centroid of the words in the verified "possible translations"
    column,and for each suggested WordNet translation (English) we calculate the cosine similarity to this centroid,
    so output value is between -1 and +1, NOT between 0 and 1!

    :param df: dataframe with "en" (WordNet translations) and a `lang_col` column (e.g. 'nl_vgt')
    :param lang_col: key to the language column with 'explanations' (possible translations)
    :param threshold: minimal value. If cosine similarity is below this, the word will not be included
    :param port: local port that the inference server is running on
    :return: the updated DataFrame where potentially items have been removed from the "en" column
    """
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    def find_similarities(row: Series):
        """On the one hand, discard English translations that do not meet the threshold similarity. On the other,
        collect similarities and gloss variants.
        - for each gloss, collect its standardized versions in `gloss_variants`
        - for each English word that meets the thershold, keep track of its similarity score and the std gloss
        for which this similarity was achieved
        """
        if not row[lang_col] or not row["en"]:
            return None

        # Do this for the list, of, words because sometimes
        # there is a comma in the parentheses, which leads to unexpected results such as
        # `bewerkingsteken (+, -...)` -> -...)
        src_words = remove_brackets(row[lang_col])
        src_words = tuple([w.strip() for w in src_words.split(", ")])
        src_centroid = send_request("centroid", session=session, port=port, params={"tokens": src_words})

        if src_centroid is None:
            return None

        en_words = [w.strip() for w in remove_brackets(row["en"]).split(", ")]
        en_words = [re.sub(r"^to\s+", "", en) for en in en_words]  # Remove "to" in infinitives
        en_vecs = send_request("vectors", session=session, port=port, params={"tokens": en_words})
        valid = []
        for en_vec, en_word in zip(en_vecs, en_words):
            sim = util.cos_sim(en_vec, src_centroid).squeeze(dim=0).item()
            if sim >= threshold:
                valid.append(en_word)
                logger.debug(f"Adding {en_word}. Close enough to {src_words}! (sim=~{sim:.2f}; threshold={threshold})")
            else:
                logger.debug(
                    f"Dropping {en_word}. Too distant from {src_words}! (sim=~{sim:.2f}; threshold={threshold})"
                )

        if valid:
            row["en"] = ", ".join(sorted(valid))
            return row
        else:
            return None

    tqdm.pandas(desc="Collecting semantic similarities")
    df = df.progress_apply(find_similarities, axis=1)
    session.close()

    df = df.dropna()
    return df


def build_en2gloss_database(df: DataFrame, db_path: str, lang_col: str):
    # Convert format to en->gloss
    en2gloss = []
    for item in df.to_dict(orient="records"):
        gloss = standardize_gloss(item["gloss"])
        en_words = sorted([en_strip for en in item["en"].split(", ") if (en_strip := en.strip())])

        for word in en_words:
            en2gloss.append({"en": word, "gloss": gloss})

    en2gloss_df = pd.DataFrame(en2gloss)
    en2gloss_df = en2gloss_df.drop_duplicates().reset_index(drop=True)

    sl_type = lang_col.split("_")[1] if "_" in lang_col else lang_col
    gloss_tbl = f"{sl_type}_en2gloss_tbl"

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    with engine.connect() as conn:
        en2gloss_df.to_sql(gloss_tbl, con=conn, if_exists="replace", index=False)
        conn.execute(f"CREATE INDEX idx_{sl_type}_en ON {gloss_tbl} (en);")


def process_dictionary(
    fin: str, dbout: str, lang_col: str, threshold: float = 0.5, only_db: bool = False, port: int = 5000
):
    """Generate WordNet translations for the words in the "possible translation" column in the dictionary.
    Then, filter those possible translations by comparing them with the centroid LABSE vector. English translations
    that have a cosine similarity (between -1 and +1) of less than the threshold will not be included in the final
    DataFrame.

    "Dictionary" mappings will be created in an SQLite table named after the lang_col, "lang_col_en2gloss_tbl"
    (e.g. "nl_vgt_en2gloss_tbl")

    :param fin: path to a formatted dictionary in TSV format
    :param dbout: path where to store the updated TSV as an SQLite database. The new data will be stored under a table
     called 'lang_col_en2gloss_tbl' where lang_col is replaced by the 'lang_col' argument given
    :param lang_col: key to the language column with 'explanations' (possible translations)
    :param threshold: similarity threshold. Lower similarity English words will not be included.
    :param only_db: whether to only generate the database. This assumes that the modified TSV file already exists!
    :param port: port where the inference server is running
    """

    pfin = Path(fin).resolve()
    if only_db:
        pfout_tsv = pfin.with_name(f"{pfin.stem}-prepr{pfin.suffix}")
        df = pd.read_csv(pfout_tsv, sep="\t", encoding="utf-8")
        df = df.dropna()
    else:
        df = pd.read_csv(fin, sep="\t", encoding="utf-8")
        had_en_column = "en" in df.columns
        df[lang_col] = df[lang_col].apply(
            lambda explanation: ", ".join(map(str.strip, re.split(r"\s*,\s*", explanation)))
        )  # clean possible white-space issues
        df = add_en_translations(df, lang_col=lang_col)

        # Filter/disambiguate translations
        df = filter_en_translations(df, lang_col=lang_col, threshold=threshold, port=port)
        df = df.dropna()

        if not had_en_column:
            # Reorder columns
            cols = list(df.columns)
            extra_cols = [c for c in cols[2:] if c != "en"]
            df = df[cols[:2] + ["en"] + extra_cols]

        pfout_tsv = pfin.with_name(f"{pfin.stem}-prepr{pfin.suffix}")
        df.to_csv(pfout_tsv, index=False, sep="\t", encoding="utf-8")
        logger.info(f"Saved updated TSV in {pfout_tsv.resolve()}")

    logger.info("Building SQLite DataBase")
    build_en2gloss_database(df=df, db_path=dbout, lang_col=lang_col)


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        description="'Translates' a dictionary via multilingual WordNet by finding synsets in Open"
        " Multilingual WordNet, their corresponding English synset, and all the words belonging to that"
        " synset. This can yield translations that are out-of-context. By comparing the vector of every English"
        " candidate translation with the centroid of the explanation words' vectors, we can filter out"
        " words whose similarity is too low.\nThe script will produce a modified TSV file with an"
        " 'en' column and an SQLite database. If the database already exists for another language, the new data"
        "will be simply added as a separate table. The tables are named 'lang_col_en2gloss_tbl' where lang_col"
        " is replaced by the respective input argument.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument("fin", help="dictionary in TSV format (after reformat and optional OpenAI translation)")
    cparser.add_argument(
        "dbout",
        help="path where to store the updated TSV as an SQLite database. The new data will be stored under a"
        " table called 'lang_col_en2gloss_tbl' where lang_col is replaced by the 'lang_col' argument given",
    )

    cparser.add_argument(
        "lang_col",
        help="name of the column that contains the 'explanations' of a gloss after reformatting, typically"
        " a language code followed by the sign language like 'nl_vgt'",
    )
    cparser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="similarity threshold. Lower similarity English words will not be included",
    )
    cparser.add_argument(
        "--only_db",
        action="store_true",
        help="whether to only generate the database. This assumes that the modified TSV file already exists!",
    )
    cparser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Local port that the inference server is running on.",
    )
    cparser.add_argument(
        "--logging_level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        default="INFO",
        help="Logging level.",
    )
    cargs = vars(cparser.parse_args())
    logger.setLevel(cargs.pop("logging_level").upper())
    process_dictionary(**cargs)


if __name__ == "__main__":
    main()
