# Idea
import json
from os import PathLike
from pathlib import Path
from typing import Union, List

# using: https://huggingface.co/yhavinga/ul2-large-dutch or https://huggingface.co/yhavinga/ul2-base-dutch
# input spoken text; output -> glosses but concatenated for both the left and right hand (separated by for instance
# `<n>` which was used during pretraining to indicate a new line)
# Eerst trainen as-is en nadien eventueel met constrained decoding? dan wel beslissen wat we doen met:
    # - NG: naamgebaar (weglaten?)
    # - G: gestures, bv. _(5,_palm_boven)_IK-WEET-NIET -> Tussen haakjes weglaten?
# maybe take ordered union of left/right? Sometimes negation in just one hand!

SEPARATOR = " <n> "
def build_dataset(data: List[dict], ltr: bool = False, include_goc: bool = True):
    ds = []
    exclude = ("goc", ) if include_goc else ()
    for item in data:
        translation = item["translation"].strip()
        left = " ".join([d["gloss"] for d in item["glosses"]["left"] if not d["gloss"].lower().startswith(exclude)]).strip()
        right = " ".join([d["gloss"] for d in item["glosses"]["right"] if not d["gloss"].lower().startswith(exclude)]).strip()
        if not translation or (not left and not right):
            continue

        sample = {"translation": translation,
                  "left": left,
                  "right": right}

        glosses = []
        if ltr: # First left, then right
            if left: glosses.append(left)
            if right: glosses.append(right)
        else:
            if right: glosses.append(right)
            if left: glosses.append(left)

        sample["glosses"] = SEPARATOR.join(glosses)

        ds.append(sample)
    return ds


def main(annotions_f: Union[str, PathLike], out_d: Union[str, PathLike]):
    all_annotations = json.loads(Path(annotions_f).read_text(encoding="utf-8"))
    train_set = [annotation for annotation in all_annotations if annotation["subset"] == "train"]
    dev_set = [annotation for annotation in all_annotations if annotation["subset"] == "validate"]
    test_set = [annotation for annotation in all_annotations if annotation["subset"] == "test"]

    proc_train = build_dataset(train_set)
    proc_dev = build_dataset(dev_set)
    proc_test = build_dataset(test_set)

    jsons_train = [json.dumps(sample) for sample in proc_train]
    jsons_dev = [json.dumps(sample) for sample in proc_dev]
    jsons_test = [json.dumps(sample) for sample in proc_test]

    print(f"{len(all_annotations)} ANNOTATED SEGMENTS FOUND!"
          f"\nTRAIN={len(jsons_train)}; DEV={len(jsons_dev)}; TEST={len(jsons_test)} (after filtering)")
    pdout = Path(out_d)
    pdout.mkdir(exist_ok=True, parents=True)
    pdout.joinpath("train.jsonl").write_text("\n".join(jsons_train) + "\n")
    pdout.joinpath("dev.jsonl").write_text("\n".join(jsons_dev) + "\n")
    pdout.joinpath("test.jsonl").write_text("\n".join(jsons_test) + "\n")




if __name__ == '__main__':
    main(r"D:\corpora\sl\Corpus VGT\annotations.json",
         r"F:\python\amr-to-sl-repr\data\corpus-vgt")
