"""
PREPARATIONS:
pip install spacy
spacy download "nl_core_news_lg"
pip install pandas
pip install telwoord
"""
import json
import os
from re import search, sub

from pandas import DataFrame, read_csv
from text2gloss.rule_based.complex_sents import make_complex_gloss_sentence
from text2gloss.rule_based.helpers import read_file


def preprocess(input, preprocess_input):
    """
    :param input: text
    :param preprocess_input: type of preprocessing
    :return: preprocessed text
    """

    def preprocessing_chn(file):
        file_lines = file.readlines()
        text = " ".join(file_lines[1:])  # delete the caption
        text = sub(r"\([a-zA-Z ]*\) ?$", "", text)  # delete the (blog) and other at the end
        text = sub(r' +<hi rend="[a-z]*">', " ", text)  # delete the markup, e.g.: <hi rend="bold">Norbert Desaver</hi>
        text = sub(r"</hi> +", " ", text)
        return text

    def original_preprocessing(file):
        dutch_gloss_sentences = []
        data = file.read()
        data_doc = nlp(data)
        for line in data_doc.sents:
            # for line in data.split('\n'):
            line = line.text.replace("\n", " ")
            if not search(r"^ *$", line):
                # line = sub(r'[éèêë]', 'e', line)
                # glosses = sub(r'[ÊËÉÈ]', 'E', glosses)
                dutch_gloss_sentences.append(line)
        return dutch_gloss_sentences

    if preprocess_input == "original":
        return original_preprocessing(input)
    elif preprocess_input == "chn":
        return preprocessing_chn(input)
    return input


def translate_document(path_to_document, preprocess_input):
    """
    translate one document to glosses
    first split up per line because we are working with the 1 sent per line version of the Wablieft corpus
    :param path_to_document: input txt file
    :param preprocess_input: type of preprocessing
    :return: list of [[original sentence, gloss representation], []]
    """
    if path_to_document.endswith(".txt"):
        with open(path_to_document, "r", encoding="utf-8") as file:
            text = preprocess(file, preprocess_input)
            dutch_gloss_sentences = make_complex_gloss_sentence(text, nlp)
        return dutch_gloss_sentences

    elif path_to_document.endswith(".csv"):
        df = read_csv(path_to_document, encoding="utf-8-sig")
        dutch_gloss_sentences = []
        for index, row in df.iterrows():
            text = preprocess(row["annotation"], preprocess_input)
            glosses = make_complex_gloss_sentence(text, nlp)
            dutch_gloss_sentences.append([text, glosses])
        return dutch_gloss_sentences


def save_to_csv(dutch_gloss_sentences, goal_directory, csv_name):
    """
    save a list of dutch-gloss pairs to a csv
    :param dutch_gloss_sentences: list to turn into dataframe
    :param goal_directory: directory to save the csv to
    :param csv_name: the name of the csv file
    :return: None
    """
    df = DataFrame(dutch_gloss_sentences, columns=["dutch", "glosses"])
    df = df.rename(columns={"dutch": "annotation"})
    goal_directory = goal_directory + "\\" + csv_name
    df.to_csv(goal_directory, encoding="utf-8-sig")


def translate_all_files_in_directory(path_to_directory, preprocess_input):
    """
    transform all files in a folder
    :param path_to_directory: input
    :param preprocess_input: type of preprocessing
    :return: gloss representation
    """
    dutch_gloss_sentences_full = []
    for filename in os.listdir(path_to_directory):
        path = os.path.join(path_to_directory, filename)
        dutch_gloss_sentences_full += translate_document(path, preprocess_input)
    return dutch_gloss_sentences_full


def translate(path_to_directories_file, output_csv=False, preprocess_input="None"):
    """
    :param path_to_directories_file: path to a yaml file containing all necessary info
    :param output_csv: Boolean value, whether to save to an csv
    :param preprocess_input: kind of preprocessing to apply
    :return: gloss representation
    """
    directories = read_file(path_to_directories_file)
    rootdir = directories["corpus"]["root_dir"]
    input = os.path.join(rootdir, directories["corpus"]["file_input"])

    # word_list = read_file(r'word_lists_flat.json')
    # scheidbare_ww = read_file(r'scheidbare_ww.txt').split('\n')
    # scheidbare_ww_scheidbaar_deel = read_file(r'scheidbare_ww_scheidbaar_deel.txt').split('\n')
    # vgt_gloss_id = read_file(r'VGT_gloss_id.txt')
    # resource_dict = {'word_list': word_list, 'separable_verbs': scheidbare_ww,
    #                  'prounouns_in_separable_verbs': scheidbare_ww_scheidbaar_deel}

    if os.path.isdir(input):
        output = translate_all_files_in_directory(input, preprocess_input)
        # this will translate all txt and csv files in that directory
    else:
        output = translate_document(input, preprocess_input)
    return output


def generate_glosses(testsentence, nlp):
    output = make_complex_gloss_sentence(testsentence, nlp)
    print(output)
    return sub(r" +", " ", " ".join(output))


def add_synthetic_glosses_to_json(nlp, save=False, root="", annotations="", new_file_name=""):
    def choose_one_translation(text):
        # text = sub('zorg/zorgen', 'zorgen', text)
        text = sub(r"zo\'n", "zo een", text)
        text = sub("hij/zij", "hij", text)
        text = sub("hem/haar", "hem", text)
        text = sub("hem/haar/hen", "hem", text)
        text = sub("zijn/haar", "zijn", text)
        return text

    with open(os.path.join(root, annotations), "rb") as json_file:
        json_file = json.load(json_file)
        processed_file = json_file[:]
        for annotation in json_file:
            input = choose_one_translation(annotation["translation"]["text"])
            output = generate_glosses(input, nlp)
            annotation["glosses"]["synthetic"] = output
            print(output)
            # print(input)
            # print(annotation['glosses']['combined'])
    if save:
        with open(os.path.join(root, new_file_name), "w") as outfile:
            json.dump(processed_file, outfile, indent=4)
    return processed_file


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
