"""check if there is an indication of plural, if not, add an indicator"""

from word_lists import quantifiers, options_before, options_after
from reorder_glosses_helpers import find_clause_with_head
import re
import random


def choose_plural_indicator(item_new_form):
    """choose a random option
    (add constraints coming from place of articulation + complexity of movement?)"""
    chosen_option = random.randint(0, len(options_before+options_after)-1)
    if chosen_option == 0:
        # add the same sign a second time
        item_new_form = item_new_form + ' ' + item_new_form
    elif chosen_option < len(options_before):
        item_new_form = options_before[chosen_option] + item_new_form
    else:
        item_new_form += options_after[chosen_option - len(options_before)]
    return item_new_form


def transform_plural(sentence_object, index, item):
    """check if there already is an indication of the plural, if not, add +++
    - this function does not check whether a sign can be made plural with reduplication"""
    sentence = sentence_object.clause_list
    if re.search(r'WG-3$', item.new_form):
        item.new_form = item.new_form[:-4] + 'sweep'
    elif index > 3:
        if sentence[index - 3].lemma_ == 'van' and 'TW' in sentence[index - 4].tag_:
            # eg: 2 van de mooie agenten
            sentence[index - 3].new_form = ''
    elif index > 2:
        if sentence[index - 2].lemma_ == 'van' and 'TW' in sentence[index - 3].tag_:
            # eg: 2 van de agenten
            sentence[index-2].new_form = ''
    else:
        clause_indices2 = []
        clause_indices = find_clause_with_head(sentence, index)
        for index3 in clause_indices:
            if 'TW' in sentence[index3].tag_ \
                    or (sentence[index3].pos_ == 'DET' and 'VNW' in sentence[index3].tag_) \
                    or sentence[index3].text in quantifiers:
                clause_indices2.append(index3)
        if 'case' not in [token.dep_ for token in sentence[min(clause_indices):max(clause_indices)]] and \
                (sentence[index-1].pos_ == 'NOUN' or sentence[min(clause_indices)-1].pos_ == 'NOUN'):
            clause_indices2 = sentence[min(clause_indices)-1].position
        if not clause_indices2:
            item.new_form = choose_plural_indicator(item.new_form)
    return sentence_object
