"""handle negation"""
from word_lists import may_conjugation, can_conjugation


def no_negation_in_question(sentence, sentence_new, index, item):
    """the negation is often confusing in questions and the meaning is kept without it
    eg: 'ga je niet naar het feest? - GA-NAAR NIET FEEST WG-2?' >< 'ga je naar het feest? - GA-NAAR FEEST WG-2?' """
    if item.lemma_ == 'niet' and sentence.pop().lemma_ == '?' and \
            'als' not in [item5.lemma_ for item5 in sentence] and \
            'indien' not in [item6.lemma_ for item6 in sentence]:
        sentence_new[index].new_form = ''
    return sentence_new
# >< ... of niet ?


def correct_negation_gloss_id(sentence, index, item, index_niet):
    if item.lemma_ in can_conjugation:
        item.new_form = 'KAN-NIET'
    elif item.lemma_ in may_conjugation:
        item.new_form = 'MAG-NIET'
    else:
        item.new_form += '-NIET'
    sentence[index_niet].new_form = ''
    return sentence


def concatenate_may_can_not(sentence_object, index, item):
    """can/must/... + not has a separate sign"""
    sentence = sentence_object.clause_list
    if len(sentence) > index + 1:
        if sentence[index+1].lemma_ == 'niet':
            correct_negation_gloss_id(sentence, index, item, index+1)
    if index > 0:
        if sentence[index-1].lemma_ == 'niet':
            correct_negation_gloss_id(sentence, index, item, index-1)
    if len(sentence) > index + 2:
        if sentence[index+2].lemma_ == 'niet':
            correct_negation_gloss_id(sentence, index, item, index+2)
    if len(sentence) > index + 3:
        if sentence[index+3].lemma_ == 'niet':
            correct_negation_gloss_id(sentence, index, item, index+3)
        # ik kan er vandaag niet bijzijn
    return sentence_object


# nog niet, toch niet, even niet, niet meer, nog geen (nog mag niet naar voor)
