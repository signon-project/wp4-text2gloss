"""go through the sentence word by word (Word objects) and transform the gloss if needed
refer to functions if the changes are complicated
(all functions get the same arguments to avoid mistakes, even if they don't use them all) """

from transform_word_specific import genoeg, two_nouns, relative_clause_with_of, was, lemma_not_infinitive, \
    transform_other_mistakes, last_processing_step
from word_lists import may_can, pers1, pers2, pers4, pers5, pers3_6, locative_preps, locative_verbs, not_signed_verbs,\
    comparison_with_separate_sign_with_e_del_e, have_conjugation, new_form_is_text,\
    must_conjugation, can_conjugation, may_conjugation, be_conjugation, worden_conjugation, \
    blijven_conjugation, blijken_conjugation, lijken_conjugation, seperated_part_of_verb, tw_to_ignore, \
    words_without_gloss, separate_negation_sign
from transform_num import transform_numbers
from find_needed_clauses import find_verb_temp_ques
from transform_pron_verb import transform_pronoun_verb, transform_verb2, \
    join_separable_verbs, join_separable_verbs2, possessive_pronouns
from transform_prep import transform_locative_prepositions, transform_prepositions
from transform_conj import transform_conj
from transform_ques import transform_question_word_clause
from transform_neg import concatenate_may_can_not
from transform_plural import transform_plural
from helpers import Clause, delete_item_sometimes
from transform_comp import transform_meer_minder_dan
from transform_with_gloss_ids import match_with_gloss_ids
import re


def dutch_to_gloss(sentence_list):
    """create a Sentence object consisting of Word objects
    - find indices of some POS
    - change glosses according to grammar rules of VGT"""

    # MAKE SPACY DOC
    sentence = [token for token in sentence_list if not token.is_space]

    # FIND INDICES NEEDED FOR SENTENCE OBJECT
    subj_index = [index for index, token in enumerate(sentence) if 'subj' in sentence[index].dep_
                  and sentence[index].pos_ in ['NOUN', 'PROPN', 'PRON', 'X', 'NUM']]
    iobj_index = [index for index, token in enumerate(sentence) if sentence[index].dep_ == 'iobj'
                  and sentence[index].pos_ in ['NOUN', 'PROPN', 'PRON', 'X', 'NUM']]
    obj_index = [index for index, word in enumerate(sentence) if sentence[index].dep_ == 'obj'
                 and sentence[index].pos_ in ['NOUN', 'PROPN', 'PRON', 'X', 'NUM']]
    if not subj_index:
        subj_index = [index for index, token in enumerate(sentence) if 'ROOT' in sentence[index].dep_
                      and ('N|' in sentence[index].tag_ or 'VNW' in sentence[index].tag_)]
    elif len(subj_index) > 1:
        subj_index = [subj_index[0]]
    if len(obj_index) > 1 and not iobj_index:
        iobj_index = [obj_index[0]]
        obj_index = obj_index[1:]
        # anticipate that spacy does not correctly recognise the indirect object

    # CREATE SENTENCE OBJECT
    sentence_object = Clause(subj_index, iobj_index, obj_index, sentence)
    if sentence_object.question_word_indices or '?' in [token.text for token in sentence_list if token.is_punct]:
        sentence_object.is_question = True

    # CHANGE SOME MISTAKES OF SPACY BEFORE TRANSFORMING INTO GLOSSES
    for index, item in enumerate(sentence_object.clause_list):
        if item.text == 'genoeg':
            genoeg(sentence_object, index, item)
            # must be changed before the start!
        if index > 0 and item.pos_ == 'NOUN':
            two_nouns(sentence_object, index, item)
            # mistake in POS labelling
        if item.text == 'was':
            was(sentence_object, index, item)
        if item.lemma_ in have_conjugation + must_conjugation + can_conjugation + may_conjugation \
                + be_conjugation + worden_conjugation + blijven_conjugation + blijken_conjugation \
                + lijken_conjugation:
            lemma_not_infinitive(sentence_object, index, item)

    # COMPLETE CHARACTERISTICS OF SENTENCE OBJECT
    find_verb_temp_ques(sentence_object)

    # (objects needed during the enumeration:)
    locative_verb = []
    all_preps = []
    all_not_signed_verbs = []

    # CHANGE GLOSS IF IT IS NOT THE LEMMA OF THE WORD
    for index2, item2 in enumerate(sentence_object.clause_list):
        # # PRONOUNS TO WG ( = 'wijsgebaar' = 'pointing sign')
        sentence[index2].new_form = 'WG-1' if item2.lemma_ in pers1 \
            else 'WG-2' if item2.lemma_ in pers2 \
            else 'WG-4' if item2.lemma_ in pers4 \
            else 'WG-5'if item2.lemma_ in pers5 \
            else 'WG-6' if item2.lemma_ in pers3_6 and 'VNW' in item2.tag_ and 'mv' in item2.tag_ \
            else 'WG-3' if item2.lemma_ in pers3_6 and 'VNW' in item2.tag_ \
            else sentence[index2].new_form

        if 'bez' in item2.tag_:
            possessive_pronouns(sentence_object, index2, item2)

        # # NUMBERS
        if 'TW' in item2.tag_ and item2.lemma_ not in tw_to_ignore:
            transform_numbers(sentence_object, index2, item2)

        # # PLURAL
        if item2.pos_ == 'NOUN' and 'mv' in item2.tag_:
            transform_plural(sentence_object, index2, item2)

        # # CONCATENATE MAY NOT / CAN NOT
        # # the negation has a seperate sign
        if item2.lemma_ in separate_negation_sign:
            concatenate_may_can_not(sentence_object, index2, item2)

        # # TRANSFORM CONJUNCTIONS
        transform_conj(sentence_object, index2, item2)              # replace with known gloss
        if item2.lemma_ == 'of' and index2 == 0:                    # replace 'of' with '?'
            relative_clause_with_of(sentence_object, index2, item2)

        # # TRANSFORM SUPERLATIVES
        # if re.search(r'ste?$', item2.text) and item2.pos_ == 'ADJ' and 'TW' not in item2.tag_:
        #     # they add a 'e' hand for marking the superlative
        #     item2.new_form = item2.text.upper()
        #     item2.new_form = re.sub(r"STE$", '^STE', item2.new_form)
        #     item2.new_form = re.sub(r"ST$", '^ST', item2.new_form)

        # # TRANSFORM EXPRESSIONS WITH 'DAN': '(IETS) MEER/MINDER DAN'
        if item2.lemma_ == 'dan' and item2.pos_ == 'SCONJ' and item2.dep_ in ['mark', 'fixed']:
            transform_meer_minder_dan(sentence_object, index2, item2)

        # # FIND PREPOSITIONS AND VERBS WITH LOCATIVE FUNCTION
        if item2.lemma_ in locative_verbs and index2 in sentence_object.verb_indices:
            locative_verb = [index2, item2]
        if item2.lemma_ in locative_preps:
            all_preps.append([index2, item2])

        # # FIND VERBS THAT ARE NOT SIGNED
        if item2.lemma_ in not_signed_verbs and 'WW' in item2.tag_:
            all_not_signed_verbs.append([index2, item2])
            item2.new_form = ''

        # # JOIN SEPARABLE VERBS
        # # add seperable part to make the infinitive as new_form
        if item2.dep_ == 'compound:prt':
            join_separable_verbs(sentence_object, index2, item2)
        if item2.lemma_ in seperated_part_of_verb:
            join_separable_verbs2(sentence_object, index2, item2)

        # # TRANSFORM PREPOSITIONS
        if 'VZ' in item2.tag_ and item2.new_form != '':
            transform_prepositions(sentence_object, index2, item2)

        # # TRANSFORM VERBS
        transform_verb2(sentence_object, index2, item2, all_not_signed_verbs)

        # # WORD SPECIFIC CHANGES
        # # # GLOSS = RAW WORD (WITH SOME ADAPTATION), NOT THE LEMMA
        if item2.text.lower() in new_form_is_text:
            # keep the '-e': 'verschillend' >< 'verschillende' / 'enkel' >< 'enkele'
            # words that have a seperate sign for the plural + words that where the gloss is not the lemma
            # 'al' >< 'alle'
            item2.new_form = item2.text.upper()
        elif item2.text in comparison_with_separate_sign_with_e_del_e:
            # 'goed' >< 'beter' (has different sign >< other comparing adjectives)
            item2.new_form = item2.text[:-1].upper()

        # # # NO SIGN
        elif 'expl' in item2.dep_ \
                or item2.lemma_ in words_without_gloss \
                or (item2.text == 'dan' and item2.pos_ == 'SCONJ' and item2.dep_ == 'mark') \
                or (item2.lemma_ == 'maar' and item2.tag_ == 'BW') \
                or item2.pos_ == 'PUNCT' \
                or 'LID' in item2.tag_ \
                or (item2.text == 'meer' and ('niet' in [item3.text for item3 in sentence]
                                              or 'geen' in [item4.text for item4 in sentence])) \
                :
            # expletives are empty words, eg: 'hij vertelt het aan haar' --> 'het' --> context
            # 'het' has no real meaning --> context
            # 'wel' and 'toch' is just to emphasize something --> context
            # 'die, dat, deze' added for now, sometimes they do have meaning, most of the time they don't
            # 'elkaar' is obvious in the context
            # 'kom jij maar naar hier' --> not MAAR
            # 'dan' in 'groter dan'
            # no determiners
            # for now we delete possisive pronouns - mostly not adding info, context is enough
            item2.new_form = ''

        elif item2.lemma_ == 'niet':
            delete_item_sometimes(sentence, index2, item2)

        # # # CORRECTION OF MISTAKES MADE BY SPACY AND OTHER WORD SPECIFIC ADAPTATIONS
        else:
            transform_other_mistakes(sentence_object, index2, item2, all_preps)

    # # APPLY SOME FUNCTIONS THAT DO NOT JUST LOOK AT ONE WORD
    transform_locative_prepositions(sentence_object, all_preps, locative_verb)
    transform_pronoun_verb(sentence_object)     # must come after changing pronouns into WG
    transform_question_word_clause(sentence_object)

    # CHANGES BASED ON GLOSS_ID LIST (after all other operations)
    match_with_gloss_ids(sentence_object)

    last_processing_step(sentence_object)

    # PRINT to see what happens and why
    print([item7.new_form for item7 in sentence])
    print(sentence_object.verb_indices)
    print([token.pos_ for token in sentence_list])
    print([token.dep_ for token in sentence_list])
    print([token.tag_ for token in sentence_list])
    print([token.head for token in sentence_list])
    print([token.lemma_ for token in sentence_list])
    return sentence_object
