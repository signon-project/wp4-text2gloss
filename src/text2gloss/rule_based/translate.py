from pathlib import Path

from text2gloss.utils import send_request


def generate_glosses(text: str, port: int = 5000):
    if Path(text).exists():
        pfin = Path(text)
        pfout = pfin.with_stem(f"{pfin.stem}-gloss")
        print(f"Input is a file. Output will be written to {pfout}")
        sentences = pfin.read_text(encoding="utf-8").splitlines()
        # Maud's code sometimes adds "+++" and "sweep", but I am not sure what these are. They seem like
        # sign instructions or pluralization. We do not include those here
        # Also only include upper-case tokens. Sometimes Maud's code fails, e.g. `eentje` -> `ééntj`
        sent_glosses = [
            " ".join(
                [
                    gloss.replace("wg+++", "").replace("sweep", "").strip()
                    for gloss in send_request("rb_text2gloss", port=port, params={"text": text})["glosses"]
                    if gloss.isupper()
                ]
            )
            for text in sentences
        ]
        pfout.write_text("\n".join(sent_glosses) + "\n", encoding="utf-8")
    else:
        output = send_request("rb_text2gloss", port=port, params={"text": text})
        print(output)


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        description="'Translates' a given Dutch 'text' or file to VGT glosses. Assumes that an inference server is"
        " running on the given 'port'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument(
        "text",
        help="Dutch text to translate to VGT glosses. If this is an existing file, the whole file will be translated"
        " and the output written to a file that is named the same but whose stem ends in -gloss.",
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
