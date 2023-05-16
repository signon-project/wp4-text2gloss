"""some more advanced functions that transform pronouns, verbs and prepositions while looking at the whole sentence"""
import re
from word_lists import not_signed_verbs, verb_with_congruence, schijnen_literal, separable_verbs, locative_verbs, \
    locative_preps


def possessive_pronouns(sentence_object, index, item):
    from helpers import delete_item_sometimes

    sentence = sentence_object.clause_list
    all_wg = tuple(item2.new_form for item2 in sentence if 'WG' in item2.new_form and item2 != item)
    if item.new_form in all_wg:
        item.new_form = ''
    else:
        delete_item_sometimes(sentence, index, item, percentage_del=80)
    return sentence_object


def transform_pronoun_verb(sentence_object):
    """change pronouns into number and append to verb if subj and/or (i)obj's make congruence"""
    sentence = sentence_object.clause_list
    subj_num = ''
    obj_num = ''
    verb_with_congruence_index = False
    for index in sentence_object.verb_indices:
        if sentence[index].lemma_ in verb_with_congruence:
            verb_with_congruence_index = index
        if sentence[index].lemma_ in locative_verbs:
            for i in sentence:
                if i.lemma_ in locative_preps:
                    sentence[index].new_form = ''
                    break
    if sentence_object.subj_index:
        subj_index = sentence_object.subj_index[0]
        if 'WG' in sentence[subj_index].new_form and verb_with_congruence_index is not False:
            # if 'VNW|pers' in sentence[subj_index].tag_ and sentence[subj_index].text == 'het':
            #     sentence[subj_index].new_form = ''
            #     # eg het regent
            # --> the 'remove element' : expl takes care of this
            if sentence[verb_with_congruence_index].lemma_ == 'geven':
                if sentence[subj_index].new_form[3] == '2':
                    subj_num = sentence[subj_index].new_form[3] + '-'
                    # sentence[subj_index].new_form = ''
        # if no congruence: WG1 behaves as a noun "WG1 verb ..."
        # >< other WG come at the end of the sentence "verb obj WG3"
    if sentence_object.iobj_index:
        # is there ever an indirect object but no congruence possible?
        iobj_index = sentence_object.iobj_index[0]
        if 'WG' in sentence[iobj_index].new_form:
            iobj_num = '-' + sentence[iobj_index].new_form[3]
            sentence[iobj_index].new_form = ''
        else:
            iobj_num = '-' + '6' if 'mv' in sentence[iobj_index].tag_ else '-' + '3'
        if verb_with_congruence_index is not False:
            if sentence[verb_with_congruence_index].lemma_ == 'geven' and iobj_num in ('-2', '-1'):
                sentence[verb_with_congruence_index].new_form = subj_num \
                                                                + sentence[verb_with_congruence_index].new_form \
                                                                + iobj_num
        # if you have a subject and an indirect object, the congruence of the verb is with these 2
    if sentence_object.obj_indices:
        obj_index = sentence_object.obj_indices[0]
        if 'WG' in sentence[obj_index].new_form:
            if verb_with_congruence_index is not False and not sentence_object.iobj_index:
                obj_num = '-' + sentence[obj_index].new_form[3]  # only keep number of WG
                sentence[obj_index].new_form = ''
            # if it is referring to something/someone mentioned before, it is not repeated
            # eg ik heb een groot cadeau gemaakt. Ik geef het aan de leerkracht. = 1-GEVEN LEERKRACHT
    if verb_with_congruence_index is not False and not sentence_object.iobj_index:
        sentence[verb_with_congruence_index].new_form = subj_num \
                                                          + sentence[verb_with_congruence_index].new_form \
                                                          + obj_num
        # this will be executed if you only have a subject of you have a subject & object
    return sentence_object


def redefine_subj_obj_for_passive(sentence_object):
    sentence_object.obj_indices = sentence_object.subj_index.copy()
    subj_text = False
    for index, item in enumerate(sentence_object.clause_list):
        if item.lemma_ == 'door':
            subj_text = item.head
            item.new_form = ''
        if item.text == subj_text:
            sentence_object.subj_index = [index]
    if sentence_object.subj_index and sentence_object.obj_indices:
        if sentence_object.subj_index[0] == sentence_object.obj_indices[0]:
            sentence_object.subj_index = []
    return sentence_object


def transform_verb2(sentence_object, index, item, all_not_signed_verbs):
    sentence = sentence_object.clause_list
    if item.lemma_ in not_signed_verbs and 'WW' in item.tag_:
        all_not_signed_verbs.append([index, item])
        item.new_form = ''
    if item.lemma_ == 'schijnen':
        literal = False
        for word in schijnen_literal:
            for index3, item3 in enumerate(sentence):
                if word in item3.lemma_:
                    literal = True
        if not literal:
            item.new_form = 'BLIJKBAAR'
    if 'WW' in item.tag_ and index not in sentence_object.verb_indices and not 'prenom' in item.tag_:
        # remove all aux verbs
        item.new_form = ''
    if item.lemma_ == 'lijken':
        if len(sentence) < 3:
            item.new_form = 'WG-1 DENKEN'
        else:
            for item2 in sentence:
                if item2.lemma_ == 'op':
                    item2.new_form = 'GELIJKEN-OP' # this is the gloss_id
                    item2.text = 'gelijken-op'
                    item2.lemma_ = 'gelijken-op'

    if all_not_signed_verbs:
        for one_verb in all_not_signed_verbs:
            if 'pass' in sentence[one_verb[0]].dep_:
                redefine_subj_obj_for_passive(sentence_object)                  # refactor passives
    if index < len(sentence) - 1:
        # delete 'OM TE'
        if item.lemma_ == 'te' and 'WW' in sentence[index+1].tag_:
            item.new_form = ''
            for index3, item3 in enumerate(sentence):
                if item3.lemma_ == 'om':
                    item3.new_form = ''
    if index < len(sentence) - 2:
        # delete 'AAN HET ...'
        if item.lemma_ == 'aan' and sentence[index+1].lemma_ == 'het' and 'WW' in sentence[index+2].tag_:
            item.new_form = ''
            sentence[index+1].new_form = ''
    return sentence_object


def join_separable_verbs(sentence_object, index, item):
    sentence = sentence_object.clause_list
    if not (item.lemma_ == 'af' and 'van' in [item5.lemma_ for item5 in sentence]):
        # !! valt van het gebouw af --> NOT afvallen
        if sentence_object.verb_indices:
            last_verb = sentence[sentence_object.verb_indices[-1]]
            new_form_option1 = (item.new_form + last_verb.new_form).lower()
            new_form_option2 = (item.new_form + ' ' + last_verb.new_form).lower()

            if new_form_option1 in separable_verbs:
                last_verb.new_form = item.new_form + last_verb.new_form
                last_verb.new_form = re.sub(r' ', '^', last_verb.new_form)
                item.new_form = ''
            elif new_form_option2 in separable_verbs:
                last_verb.new_form = item.new_form + '^' + last_verb.new_form
                last_verb.new_form = re.sub(r' ', '^', last_verb.new_form)
                item.new_form = ''
            elif 'er' in [item2.lemma_ for item2 in sentence]:
                new_form_option3 = 'er' + new_form_option1
                new_form_option4 = 'er' + new_form_option2
                if new_form_option3 in separable_verbs:
                    pass
    # put the compound preposition in front of the verb:
    # eg zegt af --> AFZEGGEN
    return sentence_object


def join_separable_verbs2(sentence_object, index, item):
    """seperate formula for when spacy does not recognize the verb
    eg flauwvallen"""
    sentence = sentence_object.clause_list
    options = []
    if not (item.lemma_ == 'af' and 'van' in [item2.lemma_ for item2 in sentence]):
        if sentence_object.verb_indices:
            last_verb = sentence[sentence_object.verb_indices[-1]]
            if last_verb.position < len(sentence):
                if item.head == sentence[last_verb.position].text:
                    options.append((item.lemma_ + ' ' + last_verb.new_form).lower())
                    options.append((item.lemma_ + last_verb.new_form).lower())
                    for option in options:
                        if option in separable_verbs:
                            last_verb.lemma_ = option
                            last_verb.new_form = option.upper()
                            last_verb.new_form = re.sub(r' ', '-', last_verb.new_form)
                            for item4 in sentence:
                                if item4.lemma_ in option and item4 != last_verb:
                                    item4.new_form = ''
    return sentence_object

