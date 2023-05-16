"""function that transforms a sentence list into a Sentence object that contains the correct glosses in 'new_form'
 (with dutch_to_glos) and then adds the positions of the words to the 'sorted_glosses_indices' in the Sentence object
  in the correct order"""


import re

from text2gloss.rule_based.dutch_to_gloss import dutch_to_gloss
from text2gloss.rule_based.reorder_glosses_helpers import (
    add_clause_to_sorted_glosses,
    add_verb_to_sorted_glosses,
    add_word_to_sorted_glosses,
)
from text2gloss.rule_based.word_lists import (
    adverbs_after_temporal_clause,
    adverbs_at_front,
    be_conjugation,
    conj_at_front,
    general_there,
    may_can,
)


def reorder_glosses(sentence_list):
    """put it in a workable format
    and add the glosses in the correct order to the sorted glosses list:
    CCONJ / SCONJ
    simple BW
    TEMP
    QUESTION WORD
    SUBJ if NO WG
    WW
    IOBJ
    OBJ
    SUBJ if WG
    REST SENTENCE
    QUESTION MARK (can be changed to the correct non-manual features)"""
    sentence_object = dutch_to_gloss(sentence_list)
    sentence = sentence_object.clause_list
    subj_index = sentence_object.subj_index[0] if sentence_object.subj_index else False
    iobj_index = sentence_object.iobj_index[0] if sentence_object.iobj_index else False
    obj_index = sentence_object.obj_indices[0] if sentence_object.obj_indices else False
    temporal_indices = sentence_object.temporal_indices if sentence_object.temporal_indices else []
    verb_indices = sentence_object.verb_indices if sentence_object.verb_indices else []
    question_word_indices = sentence_object.question_word_indices if sentence_object.question_word_indices else []
    # if this is a part of a complex sentence or this sentence started with a CONJ, this is the first sign
    sentence_started = False
    while not sentence_started:
        for index, item in enumerate(sentence):
            if (
                item.pos_ in ("SCONJ", "CCONJ")
                or "INTJ" in item.pos_
                and not sentence_started
                and item.dep_ != "fixed"
            ):
                add_word_to_sorted_glosses(sentence_object, index)
            else:
                break
        sentence_started = True

    # add simple BW such as 'dan, daarna'
    for index, item in enumerate(sentence):
        if (item.tag_ == "BW" and item.lemma_ in adverbs_at_front) or (
            item.tag_ == "VG|neven" and item.lemma_ in conj_at_front
        ):
            add_word_to_sorted_glosses(sentence_object, index)

    # temporal clause always at the front of the sentence
    if temporal_indices:
        for index in temporal_indices:
            add_word_to_sorted_glosses(sentence_object, index)

    # add simple BW such as 'misschien, ook, altijd'
    for index, item in enumerate(sentence):
        if item.tag_ == "BW" and item.lemma_ in adverbs_after_temporal_clause:
            add_word_to_sorted_glosses(sentence_object, index)

    # add the question clause
    if question_word_indices:
        for index in question_word_indices:
            # if sentence[index].lemma_ != 'waarom': # ?? should 'waarom' be handled differently?
            add_word_to_sorted_glosses(sentence_object, index)

    # 'HEBBEN' coming from 'er zijn SUBJ' at the front of the sentence, not after subject
    for index, item in enumerate(sentence):
        if index > 0:
            if item.lemma_ in be_conjugation and sentence[index - 1].lemma_ in general_there:
                add_word_to_sorted_glosses(sentence_object, index - 1)
                add_word_to_sorted_glosses(sentence_object, index)

    # WG-1 and new subject come at the front of the sentence, other WG at the end
    if subj_index is not False:
        if not re.match(r"WG-[23456]", sentence[subj_index].new_form) and subj_index not in question_word_indices:
            add_clause_to_sorted_glosses(sentence_object, subj_index)
            # WG-1 and all other subjects: SVO >< if subject = WG-2..6: OVS

            # the iobj: if it is a pronoun, it is added to the verb and the new_form is empty
            if iobj_index is not False:
                if iobj_index not in question_word_indices:
                    add_clause_to_sorted_glosses(sentence_object, iobj_index)

            # add verbs
            for index in verb_indices:
                if sentence[index].lemma_ not in may_can or not sentence_object.is_question:
                    # can_must dissapear in question
                    add_verb_to_sorted_glosses(sentence_object, index)

            # add the object
            if obj_index is not False:
                if obj_index not in question_word_indices:
                    add_clause_to_sorted_glosses(sentence_object, obj_index)

        else:
            # OVS
            # the iobj: if it is a pronoun, it is added to the verb and the new_form is empty
            if iobj_index is not False:
                if iobj_index not in question_word_indices:
                    add_clause_to_sorted_glosses(sentence_object, iobj_index)

            # add the object
            if obj_index is not False:
                if obj_index not in question_word_indices:
                    add_clause_to_sorted_glosses(sentence_object, obj_index)

            # add verbs
            for index in verb_indices:
                if sentence[index].lemma_ not in may_can or not sentence_object.is_question:
                    # can_must dissapear in question
                    add_verb_to_sorted_glosses(sentence_object, index)

    # add the extra information
    for index, item in enumerate(sentence):
        if (
            not item.new_position_assigned
            and index
            not in question_word_indices + [index for index in sentence_object.subj_index if subj_index is not False]
            and item.pos_ != "PUNCT"
        ):
            add_clause_to_sorted_glosses(sentence_object, index)

    # add the subject if it is a WG
    if subj_index is not False:
        if not sentence[subj_index].new_position_assigned and subj_index not in question_word_indices:
            add_word_to_sorted_glosses(sentence_object, subj_index)

    for index, item in enumerate(sentence):
        # in sentence with 'ik weet niet of ...' --> 'of' replaced with '?' (in change mistakes)
        if item.text == "?":
            add_word_to_sorted_glosses(sentence_object, index)
    return sentence_object
