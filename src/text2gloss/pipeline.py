from text2gloss.utils import send_request


def run_pipeline(text: str, port: int = 5000, verbose: bool = True):
    glosses = send_request("text2gloss", port=port, params={"text": text})
    if verbose and glosses:
        print("text", text)
        print(glosses)

    return glosses


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        description="Convert a given sentence into glosses. The pipeline consists of gathering the AMR representation,"
        " extracting concepts from AMR, and then looking up those English concepts in the dictionary for"
        " a corresponding gloss. Some disambiguation may be done through FastText embeddings if one concept"
        " has multiple gloss candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument(
        "text",
        help="Text to translate to glosses",
    )
    cparser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Local port that the inference server is running on",
    )

    run_pipeline(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
