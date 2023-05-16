"""transform with the list of gloss id's
"""
from text2gloss.rule_based.word_lists import gloss_ids_json


def match_with_one_word(sentence_object, index, item, gloss_ids):
    """search for a match, add the gloss_id as a feature of the word object if a match is found"""
    match_found = False
    if item.new_form.lower() in gloss_ids and item.new_form != "":
        item.new_form = gloss_ids[item.new_form.lower()]
        match_found = True

    return sentence_object, match_found


def match_with_multiple_words(sentence_object, index, item, gloss_ids_json):
    """handle glosses that are a translation of multiple words in Dutch"""
    import re

    sentence = sentence_object.clause_list
    gloss_ids_long = gloss_ids_json["ids_long"]
    gloss_ids_long_last_words = gloss_ids_json["ids_long_last_words"]
    gloss_ids_long["vijftien minuten"] = "KWARTIER"  # only '15' in list, not 'vijftien'
    # this can be changed to a hand-crafted list as well
    match_found = []
    if item.lemma_ in gloss_ids_long_last_words and not match_found:
        # faster to first check for the last word only
        for translation, gloss_id in gloss_ids_long.items():
            if (item.lemma_ in translation or item.text.lower() in translation) and not match_found:
                for word in re.split(r"[\s+\^-]", translation):
                    if word in [item2.lemma_ for item2 in sentence]:
                        position = [item3.position for item3 in sentence if item3.lemma_ == word]
                        match_found.append(position[0])
                    elif word in [item2.text.lower() for item2 in sentence]:
                        position = [item3.position for item3 in sentence if item3.text.lower() == word]
                        match_found.append(position[0])
                    else:
                        match_found = []
                        break  # if not all the words of the translation are
                        # present in our current sentence
            if match_found:
                for position in match_found:
                    sentence[position].new_form = ""
                    sentence[position].gloss_id = gloss_id  # put in each part of the translation the whole gloss_id
                sentence[match_found[-1]].new_form = gloss_id  # we want the gloss_id to appear once in the translation
                break  # don't search further
    return sentence_object, match_found


def match_with_gloss_ids(sentence_object):
    """look for gloss_id + translation in the list
    if no translation is found, append it to the 'unknown glosses' list"""
    for index, item in enumerate(sentence_object.clause_list):
        if item.gloss_id == "":  # if it was not added in 'match_with_multiple_words'
            sentence_object, match_found = match_with_multiple_words(sentence_object, index, item, gloss_ids_json)
            if not match_found:
                sentence_object, match_found = match_with_one_word(
                    sentence_object, index, item, gloss_ids_json["ids_all"]
                )
    return sentence_object
