from text2gloss.utils import send_request


def generate_glosses(text, port: int = 5000):
    output = send_request("rb_text2gloss", port=port, params={"text": text})
    print(output)


def main():
    import argparse

    cparser = argparse.ArgumentParser(
        description="'Translates' a given Dutch 'text' to VGT glosses. Assumes that an inference server is running"
        " on the given 'port' to retrieve spaCy parses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cparser.add_argument(
        "text",
        help="Dutch text to translate to VGT glosses",
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
