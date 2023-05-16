"""function to transform conjunctions into an existing gloss"""
import copy

from word_lists import conjunctions_then_before, conjunctions_because, not_used_conjunctions, conjunctions_if, \
    conjunctions_but, conjunctions_while
from helpers import delete_item_sometimes, Word
import random


def transform_while(sentence, index, number_couple, splits):
    splits_new = copy.deepcopy(splits)
    item = sentence[number_couple[0]]
    choice1 = random.choices([True, False])
    if choice1 == [True]:
        sentence.insert(0, Word('nu', 'NU', 'nu', 0, 'cc', 'nu', 'VG|neven', 'CCONJ', False, False))
        for item2 in sentence[1:]:
            item2.position += 1
        splits_new = []
        if index > 1:
            for couple in splits[:index - 2]:
                splits_new.append(couple)
        if index > 0:
            splits_new.append([splits[index-1][0], splits[index-1][1] + 1])
        for couple in splits[index:]:
            splits_new.append([couple[0] + 1, couple[1] + 1])
        choice = random.choices([True, False])
        if choice == [True]:
            item.new_form = 'OOK NU'
        else:
            item.new_form = ''
    return sentence, splits_new


def transform_conj(sentence_object, index, item):
    """transform to the 'basic' conjunction"""
    sentence = sentence_object.clause_list
    if item.lemma_ in conjunctions_then_before:
        delete_item_sometimes(sentence, index, item, option_true='AF', option_false='AF DAN')
        item.lemma_ = 'dan'
    elif item.lemma_ in conjunctions_because:
        item.new_form = 'OMDAT' # can be REDEN too
        item.lemma_ = 'omdat'
    elif item.lemma_ in not_used_conjunctions:
        item.new_form = ''
    elif item.lemma_ in conjunctions_if:
        delete_item_sometimes(sentence, index, item)
        item.lemma_ = 'als'
    elif item.lemma_ in conjunctions_but:
        item.new_form = 'MAAR'
        item.lemma_ = 'maar'
    if item.lemma_ in conjunctions_then_before + conjunctions_because + conjunctions_if + conjunctions_but \
            and not (item.lemma_ == 'dan' and item.dep_ in ['mark', 'fixed']):
        item.dep_ = 'cc'
        item.tag_ = 'VG|neven'
        item.pos_ = 'CCONJ'
    return sentence_object
