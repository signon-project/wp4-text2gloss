# Idea
import json
import re
from operator import itemgetter
from os import PathLike
from pathlib import Path
from random import shuffle
from typing import Union, List, Literal, Optional, Dict

from ftfy import fix_text


# TODO:
#
# using: https://huggingface.co/yhavinga/ul2-large-dutch or https://huggingface.co/yhavinga/ul2-base-dutch
def filter_glosses(glosses: List[str], translation: str, remove_gestures: bool = False, remove_prefixes: bool = False,
                   remove_parentheses: bool = False,
                   simplify_wg: bool = False, simplify_c: bool = False, remove_specifiers: bool = False,
                   remove_ac: bool = False, normalize_vs: bool = False, remove_repetitions: bool = False,
                   remove_location: bool = False, remove_buoy: bool = False,
                   selective_removal_ng: bool = False):
    """
    :param glosses:
    :param translation:
    :param remove_gestures: remove gestures (starting with G:)
    :param selective_removal_ng: remove naamgebaren (NG) but only if it is not literally present in the translation
    :param remove_prefixes: remove prefixes such as NG:, G:, etc.
    :param remove_parentheses: remove parentheses and their content, e.g. "_(5,_palm_boven)_IK-WEET-NIET" -> "IK-WEET-NIET"
    :param simplify_wg: simplify wijsgebaren (WG), e.g. "WG-3bernard" -> "WG-3"; "WG-6_(B)_(vingers_lijstboei)_" -> "WG-6"
    :param simplify_c: simplify numbers (c; cijfers), e.g. "C:_6" -> "6"
    :param remove_ac: remove "actief" designators, e.g. "BUITENLAND-A_(ac)" -> "BUITENLAND-A"
    :param remove_specifiers: remove specifiers, e.g. "REKENEN-A" -> "REKENEN"
    :param normalize_vs: normalize finger spelling (vs), e.g. "VS:_T-L-I-B-N" -> "TLIBN" (Taliban)
    :param remove_repetitions: remove repetitions (one or more + signs), e.g. "JA-A_+++" -> "JA-A_"
    :param remove_location: remove location specificiers, e.g. "ELKE-DAG-wang" -> "ELKE-DAG"; "MENEER-Thand>5" -> "MENEER"
    :param remove_buoy: remove buoys (lijstboei) specificiers, e.g. lijstboei_(3)
    :return:
    """
    # Filter glosses where the annotator did not know the gloss
    # §: "valse start" , *: verspreking, ?(??): onduidelijk gebaren; "EUHM": aarzeling
    glosses = [fix_text(g) for g in glosses if
               not g.endswith(("?", "§", "*", "EUHM")) and not g.startswith("?") and "/" not in g]

    print("before remove_buoy", glosses)
    if remove_buoy:
        glosses = [g for g in glosses if not g.startswith("lijstboei")]

    # final digits can have special meanings: the digits often refer to specific persons. So to make it easier for
    # the model, we rephrase those here to Wg\d. E.g. "VAN-1" -> "VAN WG-1"; "3-GEVEN-D-1" -> "WG3 GEVEN-D WG1
    tmp_glosses = []
    for g in glosses:
        if match := re.search(r"(\d+)-?(.*?)-(\d+)", g):
            if match.group(1):
                tmp_glosses.append(f"WG-{match.group(1)}")

            tmp_glosses.append(match.group(2))

            if match.group(3):
                tmp_glosses.append(f"WG-{match.group(3)}")
        else:
            tmp_glosses.append(g)
    glosses = tmp_glosses
    print("before remove_repetitions", glosses)
    if remove_repetitions:
        glosses = [g.replace("+", "") for g in glosses]

    print("before remove_ac", glosses)
    if remove_ac:
        glosses = [re.sub(r"\(ac\)", "", g, flags=re.IGNORECASE) for g in glosses]

    print("before selective_removal_ng", glosses)
    if selective_removal_ng:
        glosses = [g for g in glosses if
                   not g.startswith("NG:") or re.sub(r"NG:_?", "", g).lower().replace("_", " ") in translation.lower()]

    print("before remove_parentheses", glosses)
    if remove_parentheses:
        glosses = [re.sub(r"\([^)]*\),?", "", g) for g in glosses]

    print("before normalize_vs", glosses)
    if normalize_vs:
        glosses = ["".join(g.replace("VS:", "").split("-")) if g.startswith("VS:") else g for g in glosses]

    print("before remove_prefixes", glosses)
    if remove_prefixes:
        glosses = [re.sub(r"^[A-Z]+[:_]+", "", g) for g in glosses]
        glosses = [re.sub(r"^Go?c[:_]+", "", g, flags=re.IGNORECASE) for g in glosses]

    print("before remove_gestures", glosses)
    if remove_gestures:
        glosses = [g for g in glosses if not g.startswith("G:")]

    print("before simplify_wg", glosses)
    if simplify_wg:
        glosses = [re.sub(r".*(W[Gg]-\d+).*", r"\1", g) for g in glosses]

    print("before simplify_c", glosses)
    if simplify_c:
        glosses = [re.sub(r"^C:_(\d+)", r"\1", g) for g in glosses]

    print("before remove_specifiers", glosses)
    if remove_specifiers:
        glosses = [re.sub(r"(.+)-[A-Z]\d?[\"']?_*$", r"\1", g) for g in glosses]
        glosses = [re.sub(r"(.+)-\d?[\"']?_*$", r"\1", g) if not g.startswith("WG") else g for g in glosses]
        # Do this again because after removing numbers, this might occur again
        glosses = [re.sub(r"(.+)-[A-Z]\d?[\"']?_*$", r"\1", g) for g in glosses]

    if remove_location:
        # reduce any specified location if it is present in the last part of the gloss
        # also "op" for cases like DANSEN-VopB
        locations = ("ruimte", "hand", "wang", "kin", "hoofd", "lichaam", "recht", "glos", "op", "herhaling", "goc")
        glosses = ["-".join(g.split("-")[:-1]) if any(loc in g.split("-")[-1].lower() for loc in locations) else g for g in
                   glosses]

    # At the end: trim preceding/trailing underscores
    print("before trimming whitespace", glosses)
    glosses = [re.sub(r"^[^a-zA-Z]*(.*?)[^a-zA-Z]*$", r"\1", g) for g in glosses]
    glosses = [g.strip().replace("_", "-") for g in glosses if g.strip()]
    glosses = filter_subsequent_duplicates(glosses)

    print("final", glosses)
    print()
    return glosses


def find_intersected_glosses(item) -> Optional[List[str]]:
    glosses = item["glosses"]["right"] + item["glosses"]["left"]

    if not glosses:
        return None
    elif len(glosses) == 1:
        return [glosses[0]["gloss"]]

    glosses.sort(key=itemgetter("start_ms"))

    new_glosses = []
    do_skip = False
    num_pairs = len(glosses) - 1
    # Iterate over pairs to see if two subsequent items are "the same"
    for pair_idx, (g1, g2) in enumerate(zip(glosses, glosses[1:]), 1):
        if do_skip:
            do_skip = False
            continue

        g1text = g1["gloss"].strip()
        g2text = g2["gloss"].strip()

        # Not at the same time, so skip
        if abs(g1["start_ms"] - g2["start_ms"]) > 1:
            new_glosses.append(g1text)
            # Last pair, so add last item
            if pair_idx == num_pairs:
                new_glosses.append(g2text)
            continue

        if g1text == f"{g2text}_(ac)" or g2text == f"{g1text}_(ac)":
            new_glosses.append(g2text.replace("_(ac)", ""))
            do_skip = True
        elif g1text == g2text:
            new_glosses.append(g2text)
            do_skip = True
        else:
            new_glosses.append(g1text)

        # Last pair, so add last item
        if pair_idx == num_pairs and not do_skip:
            new_glosses.append(g2text)

    return new_glosses


def filter_subsequent_duplicates(glosses: List[str]):
    if len(glosses) == 1:
        return glosses

    # Remove subsequent duplicates
    final_glosses = []
    do_skip = False
    num_pairs = len(glosses) - 1
    for pair_idx, (g1text, g2text) in enumerate(zip(glosses, glosses[1:]), 1):
        if do_skip:
            do_skip = False
            continue

        if g1text == g2text:
            do_skip = True

        final_glosses.append(g1text)
        # Last pair, so add last item
        if pair_idx == num_pairs and not do_skip:
            final_glosses.append(g2text)

    return final_glosses


def build_dataset(data: List[dict],
                  min_length: int = 3,
                  **kwargs):
    ds = []
    for item in data:
        translation = item["translation"].strip()

        if not translation:
            continue

        print(translation)
        glosses = find_intersected_glosses(item)
        if not glosses or len(glosses) < min_length:
            continue
        glosses = filter_glosses(glosses, translation, **kwargs)

        if not glosses or len(glosses) < min_length:
            continue
        glosses_str = " ".join(glosses)

        ds.append({"translation": translation, "glosses": glosses_str})
    return ds


def main(annotions_f: Union[str, PathLike], out_d: Union[str, PathLike],
         splits: Dict[str, float] = None, shuffle_before_split: bool = True):
    all_annotations = json.loads(Path(annotions_f).read_text(encoding="utf-8"))

    kwargs = {
        "remove_gestures": True,
        "remove_prefixes": True,
        "remove_parentheses": True,
        "simplify_wg": True,
        "simplify_c": True,
        "remove_specifiers": True,
        "remove_ac": True,
        "normalize_vs": True,
        "remove_repetitions": True,
        "remove_location": True,
        "remove_buoy": True,
        "selective_removal_ng": True
    }
    processed = build_dataset(all_annotations, **kwargs)
    jsonified = [json.dumps(sample) for sample in processed]
    num_items = len(jsonified)

    print(f"{len(all_annotations)} ANNOTATED SEGMENTS FOUND!\n({num_items} after filtering)")
    pdout = Path(out_d)
    pdout.mkdir(exist_ok=True, parents=True)
    pdout.joinpath("glosses.jsonl").write_text("\n".join(jsonified) + "\n")

    if not splits:
        return

    if sum(splits.values()) != 1:
        raise ValueError("'splits' values should sum to 1")

    if shuffle_before_split:
        shuffle(jsonified)

    cut_offs = {split_name: int(split_size * num_items) for split_name, split_size in splits.items()}
    data_splits = {}
    prev_cut_off = 0
    for split_name, cut_off in cut_offs.items():
        data_splits[split_name] = jsonified[prev_cut_off:prev_cut_off+cut_off]
        prev_cut_off += cut_off
        print(f"{split_name.upper()} SPLIT SIZE: {len(data_splits[split_name]):,}")
        pdout.joinpath(f"{split_name}.jsonl").write_text("\n".join(data_splits[split_name]) + "\n")

    return data_splits


if __name__ == '__main__':
    main(r"D:\corpora\sl\Corpus VGT\annotations.json",
         r"F:\python\amr-to-sl-repr\data\corpus-vgt",
         splits={"train": 0.8, "dev": 0.1, "test": 0.1},
         shuffle_before_split=True)
