import re
from word_lists import verb_with_prep


def transform_prepositions(sentence_object, index, item):
    sentence = sentence_object.clause_list
    if item.head in [i.text for i in sentence]:
        i_head = [i.text for i in sentence].index(item.head)
        if 'WW' in sentence[i_head].tag_:
            item.new_form = ''
    if 'fin' in item.tag_:
        item.new_form = ''
    if item.lemma_ in verb_with_prep:
        verbs_with_fixed_prep = verb_with_prep[item.lemma_]
        verb_lemmas = [sentence[i].lemma_ for i in sentence_object.verb_indices]
        verb_texts = [sentence[i].text for i in sentence_object.verb_indices]
        for v in verb_lemmas + verb_texts:
            if v in verbs_with_fixed_prep:
                item.new_form = ''
    return sentence_object


def transform_locative_prepositions(sentence_object, all_preps, locative_verb):
    sentence = sentence_object.clause_list
    if all_preps:
        for one_prep in all_preps:
            item = one_prep[1]
            if item.new_form != '':
                if locative_verb:
                    if 'naar' in item.lemma_:
                        item.new_form = 'GA-NAAR' # if locative_verb[1].lemma_ == 'gaan' else 'NAAR'
                        # --> 'NAAR' is not in gloss ids
                        # 'naartoe', 'ernaar' etc
                    sentence[locative_verb[0]].new_form = ''
                    # usually not signed, but could be in some situations
                if 'op' in item.text and item.new_form != '':
                    item.new_form = 'OP'
                    # 'bovenop' etc
                item.new_form = re.sub(r'^ER', '', item.new_form)
                if len(all_preps) > 1:
                    # if there are more than one prepositions:
                    # eg NOT: 'doe de deur toe' >< REMOVE: 'ik loop naar het volk toe' --> WG-1 LOPEN NAAR VOLK
                    item.new_form = re.sub(r'TOE', '', item.new_form)
                if 'af' in item.lemma_:
                    # '(er)van (...) af' --> make verb construction
                    for one_prep2 in all_preps:
                        item2 = one_prep2[1]
                        if 'van' in item2.lemma_:
                            if sentence_object.verb_indices:
                                last_verb = sentence[sentence_object.verb_indices[-1]]
                                if item.position < item2.position: # ik val af van te sporten.
                                    last_verb.new_form = 'AF' + last_verb.new_form
                                    item.new_form = ''
                                    item2.new_form = ''
                                else: # ik val van het gebouw af.
                                    item.new_form = 'ervanaf' + '^' + last_verb.new_form.lower()
                                    last_verb.new_form = ''
                                    item2.new_form = ''
                                    # small because this is productive lexicon
    return sentence_object
