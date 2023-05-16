"""handle comparisons"""

import copy
import re
from helpers import Word
from word_lists import one_options, more_less


def transform_comparative(sentence, index_in_splits, splits):
    """if 'dan' is used for comparing, transform the sentence"""
    index_dan = splits[index_in_splits][0]
    item_dan = sentence[index_dan]

    # FIRST DELETE THE 'DAN'
    item_dan.new_form = ''
    item_dan.text = ''
    item_dan.lemma_ = ''

    sentence_list_new = copy.deepcopy(sentence)
    item = sentence[splits[index_in_splits][0] - 1]
    for n in range(splits[index_in_splits][0] - 1, -1, -1):
        if re.search(r'er$', sentence[n].lemma_) and sentence[n].lemma_ != 'minder':
            item = sentence[n]
            break
    # groter dan --> ... groot // ... minder groot
    ignore = False
    if sentence[index_dan - 2].lemma_ == 'minder':
        sentence_list_new.insert(splits[index_in_splits][1], Word('meer', 'MEER', 'meer', len(sentence),
                                                                  'advmod', item.text,
                                                                  'VNW|onbep|grad|stan|vrij|zonder|comp', 'ADJ', False,
                                                                  False))
    elif sentence[index_dan - 1].lemma_ in ('minder', 'meer'):
        ignore = True
    else:
        sentence_list_new.insert(splits[index_in_splits][1], Word('minder', 'MINDER', 'minder', len(sentence),
                                                                  'advmod', item.text,
                                                                  'VNW|onbep|grad|stan|vrij|zonder|comp', 'ADJ', False,
                                                                  False))
    if sentence[index_dan - 2].lemma_ == 'veel' and not ignore:
        sentence_list_new[index_dan - 2].new_form = 'HEEL'

    splits_new = copy.deepcopy(splits)
    if not ignore:
        # adding the adjective again is actually not necessary, but makes it more clear
        sentence_list_new.insert(splits[index_in_splits][1] + 1, Word(item.text, item.new_form, item.lemma_,
                                                                      len(sentence) + 1, item.dep_, item.text,
                                                                      item.tag_, item.pos_, False, False))
        splits_new[index_in_splits][1] += 2
        for index, number_couple in enumerate(splits):
            if index > index_in_splits:
                splits_new[index][0] += 2
                splits_new[index][1] += 2
    return splits_new, sentence_list_new


def transform_meer_minder_dan(sentence_object, index, item):
    sentence = sentence_object.clause_list
    # 'iets meer/minder dan' = 'ongeveer', eg: 'op minder dan een dag', 'meer dan 500 mensen'
    item.new_form = ''                          # delete 'dan'
    if index > 0:
        if sentence[index - 1].lemma_ in more_less:
            if index > 1:
                if sentence[index - 2].lemma_ == 'iets':
                    sentence[index - 1].new_form = 'ONGEVEER'
                    sentence[index - 2].new_form = ''  # delete 'iets'
    if index < len(sentence) - 1:
        if sentence[index + 1].text in one_options:
            # keep the 'een'
            sentence[index + 1].tag_ = 'TW'
            sentence[index + 1].lemma_ = '1'
            sentence[index + 1].text = '1'
            sentence[index + 1].new_form = '1'
    return sentence_object
