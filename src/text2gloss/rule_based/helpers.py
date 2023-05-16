import os

class Word:
    """save all the needed characteristics"""

    def __init__(self, text, new_form, lemma_, position, dep_, head, tag_, pos_, is_punct, is_space):
        self.text = text
        self.new_form = new_form
        self.lemma_ = lemma_
        self.position = position
        self.dep_ = dep_
        self.head = head
        self.tag_ = tag_
        self.pos_ = pos_
        self.is_punct = is_punct
        self.is_space = is_space
        self.new_position_assigned = False
        self.gloss_id = ''


class Clause:
    """sentence: list constructed from objects of class Word
    + some indices referring to that list
    + the order the glosses need to be in"""

    def __init__(self, subj_index, iobj_index, obj_indices, clause_list):
        self.subj_index = subj_index
        self.iobj_index = iobj_index
        self.obj_indices = obj_indices
        self.clause_list = clause_list
        self.verb_indices = tuple()
        self.temporal_indices = tuple()
        self.question_word_indices = tuple()
        self.sorted_glosses_indices = []
        self.is_question = False


def make_sentence_list(sentence_doc_list):
    """take a spacy doc and turn it into a list of objects of Word"""
    sentence_list = []
    for index, token in enumerate(sentence_doc_list):
        current_word = Word(token.text, token.lemma_.upper(), token.lemma_.lower(), index, token.dep_,
                            token.head.text, token.tag_, token.pos_, token.is_punct, token.is_space)
        sentence_list.append(current_word)
    return sentence_list


def delete_item_sometimes(sentence, index, item, percentage_del=50, option_true='', option_false='item.new_form'):
    from random import choices

    option_false = item.new_form if option_false == 'item.new_form' else option_false
    weights = [percentage_del, 100 - percentage_del]
    choice = choices([True, False], weights=weights)
    item.new_form = option_true if choice == [True] else option_false
    return sentence


def read_file(input):
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def read(file):
        if input.endswith('.json'):
            import json
            output = json.load(file)
        elif input.endswith('.csv'):
            import csv
            output = csv.reader(file, delimiter=',')
        elif input.endswith('.yaml'):
            import yaml
            output = yaml.safe_load(file)
        else:
            output = file.read()
        return output
    try:
        file = open(os.path.join(__location__, input), 'r', encoding="utf-8")
        output = read(file)
    except:
        file = open(os.path.join(__location__, input), 'r')
        output = read(file)
    file.close()
    return output
