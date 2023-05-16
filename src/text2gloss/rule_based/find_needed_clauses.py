"""find some clauses that are needed for the reordering of the glosses / transformation of the glosses"""

import re
from word_lists import temporal_expressions, question_words, not_signed_verbs, signed_if_only_verb, may_can, \
    time_unit, preposition_with_time, question_words_with_noun, elements_incorporated_in_time_sign, time_back, start_end


def find_verb_temp_ques(sentence_object):
    """apply all 3 functions"""
    find_verbs(sentence_object)
    find_question_phrase(sentence_object)
    find_temp_clause(sentence_object)           # must come after question word finding!
    return sentence_object


def find_verbs(sentence_object):
    """find the verbs and save their indices in the sentence object"""
    sentence = sentence_object.clause_list
    verbs = [it for it in sentence if 'WW' in it.tag_ and not 'prenom' in it.tag_] \
        if [it for it in sentence if 'WW' in it.tag_ and not 'prenom' in it.tag_] \
        else [it for it in sentence if it.pos_ == 'VERB' and not 'prenom' in it.tag_] \
        if [it for it in sentence if it.pos_ == 'VERB' and not 'prenom' in it.tag_] \
        else []
    for verb in [verb for verb in verbs]:       # no idea why verbs instead of [] didn't work
        if verb.lemma_ in not_signed_verbs:
            verbs.remove(verb)
    if len([verb2 for verb2 in verbs if verb2.lemma_ not in may_can]) > 1:
        for verb in [verb for verb in verbs if verb not in may_can]:
            if verb.lemma_ in signed_if_only_verb:
                verbs.remove(verb)
    verb_indices = [index for index, token in enumerate(sentence) if token in verbs]
    sentence_object.verb_indices = verb_indices
    return sentence_object


def find_question_phrase(sentence_object):
    """find the whole question word clause and save there indices in the sentence object"""
    sentence = sentence_object.clause_list
    if len(sentence) > 0:
        if sentence[0].lemma_ in question_words and 'CONJ' not in sentence[0].pos_:
            sentence_object.question_word_indices = [0]
        elif len(sentence) > 1 and 'CONJ' not in sentence[0].pos_:
            if sentence[1].lemma_ in question_words:
                # eg aan wie, naar waar
                sentence_object.question_word_indices = [0, 1]
    if sentence_object.question_word_indices:
        # adding the noun after 'welk' or 'hoeveel' if there is one. eg: 'welke schoenen doe je aan?'
        if sentence[sentence_object.question_word_indices[-1]].lemma_ in question_words_with_noun\
                and sentence[sentence_object.question_word_indices[-1]].head == \
                sentence[sentence_object.question_word_indices[-1]+1].text:
            sentence_object.question_word_indices.append(sentence_object.question_word_indices[-1]+1)
    return sentence_object


def find_temp_clause(sentence_object):
    """find all the words that make up the temporal clause, return their indices
    (because the temporal info must be at the front of a VGT sentence)
    ! must come after finding the question words"""
    sentence = sentence_object.clause_list
    temporal_clause_indices = []
    for index, item in enumerate(sentence):
        ignore = False
        if index > 2:
            if sentence[index - 2] == 'van' and sentence[index - 1] in ['het', 'de'] and item.lemma_ in time_unit:
                ignore = True
                # product van het jaar, aanbieding van de week
        if ('krijgen' in [item4.lemma_ for item4 in sentence] or 'geven' in [item4.lemma_ for item4 in sentence])\
                and item.dep_ == 'obj' and item.lemma_ in time_unit:
            ignore = True
            # 30 jaar (cel) krijgen / een uur straf geven
        if (item.text in temporal_expressions or item.lemma_ in temporal_expressions) and not ignore:
            if index > 0:
                if 'TW' in sentence[index - 1].tag_ and not sentence[index - 1].head == item.text:
                    # sometimes the number depends on the time word, sometimes the other way around:
                    # eg: om 2 uur VS 2 uur geleden
                    for index2, item2 in enumerate(sentence):
                        if item2.head == sentence[index - 1].text or item2.text == sentence[index - 1].text:
                            temporal_clause_indices.append(item2.position)
            if 0 < index < len(sentence) - 1:
                if 'TW' in sentence[index - 1].tag_ and sentence[index + 1].lemma_ == 'oud':
                    sentence[index + 1].new_form = ''
            if index > 1:
                if sentence[index - 2].lemma_ in start_end:
                    temporal_clause_indices.append(index - 2)
                if sentence[index - 1].lemma_ in start_end:
                    temporal_clause_indices.append(index - 1)
                for index3, item3 in enumerate(sentence):
                    if index3 <= index and item3.head == item.text or item3.text == item.text:
                        temporal_clause_indices.append(item3.position)
                    if index3 > index and (item3.head == item.text and item3.dep_ == 'nmod') :
                        # also add 'zondag' in 'vorige week zondag'
                        temporal_clause_indices.append(item3.position)
            if index < len(sentence) - 1:
                if sentence[index + 1].lemma_ in time_back:
                    # 'geleden' does not depend on the other time elements
                    temporal_clause_indices.append(sentence[index + 1].position)
            if temporal_clause_indices:
                temporal_clause_indices = [index4 for index4
                                           in range(min(temporal_clause_indices), max(temporal_clause_indices)+1)]
                # --> everything in between is part of the temporal clause as well:
                # eg: 'op (minder dan een) dag' (only 'op' depends on 'dag')
        if re.match(r'[0-9][0-9][/-_.][0-9][0-9][/-_.][0-9]?[0-9]?[0-9][0-9]', item.lemma_):
            # include years: '20' or '2020'
            for index2, item2 in enumerate(sentence):
                if item2.head == item.text or item2.text == item.text:
                    temporal_clause_indices.append(item2.position)
        if re.match(r'[0-9][0-9]?[uh:][0-9]?[0-9]?', item.lemma_):
            # include digital writing
            for index2, item2 in enumerate(sentence):
                if item2.head == item.text or item2.text == item.text:
                    temporal_clause_indices.append(item2.position)
        # matching years but not normal numbers ??
    if temporal_clause_indices:
        if len(sentence) > max(temporal_clause_indices) + 1:
            if sentence[max(temporal_clause_indices) + 1].lemma_ in preposition_with_time:
                # eg: 'de 5 weken rond pasen'
                try:
                    ind = [item4.text for item4 in sentence].index(sentence[max(temporal_clause_indices) + 1].head)
                    # find the index of the head of the preposition
                    temporal_clause_indices += [index5 for index5 in range(max(temporal_clause_indices) + 1, ind + 1)]
                except:
                    pass
        for index6 in temporal_clause_indices:
            if sentence[index6].lemma_ in elements_incorporated_in_time_sign:
                sentence[index6 + 1].new_form = sentence[index6].lemma_.upper() + '-' + sentence[index6 + 1].new_form
                # eg: 'deze week' >< 'no separate gloss_id for 'deze-maand', 'dit-jaar', 'vorige week'
                # put the pronoun inside the head because it will be just one sign
                sentence[index6].new_form = ''
    temporal_clause_indices_new = [index for index in list(dict.fromkeys(temporal_clause_indices))
                                   if index not in sentence_object.question_word_indices]
    temporal_clause_indices_new.sort()
    sentence_object.temporal_indices = temporal_clause_indices_new
    return sentence_object
