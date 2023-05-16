"""Word specific changes"""
import re

from text2gloss.rule_based.word_lists import (
    be_conjugation,
    blijken_conjugation,
    blijven_conjugation,
    can_conjugation,
    demonstrative,
    have_conjugation,
    lijken_conjugation,
    may_conjugation,
    must_conjugation,
    worden_conjugation,
)


def genoeg(sentence_object, index, item):
    """genoeg is not handled correctly by spacy, the sign is an ADJ
    !! must be changed before the start"""
    sentence = sentence_object.clause_list

    # CHECK FOR NEGATION AND EXTRA ADJECTIVE
    if index > 0:
        if sentence[index - 1].text == "niet":
            item.new_form = "NIET GENOEG"
            sentence[index - 1].new_form = ""
        elif sentence[index - 1].pos_ == "ADJ":
            # eg: 'groot genoeg'
            sentence[index - 1].new_form = item.new_form + " " + sentence[index - 1].new_form
            item.new_form = ""
            if index > 1:
                # eg: 'niet groot genoeg'
                if sentence[index - 2].text == "niet":
                    sentence[index - 1].new_form = "NIET " + sentence[index - 1].new_form
                    sentence[index - 2].new_form = ""
        else:
            item.new_form = "GENOEG"
    else:
        item.new_form = "GENOEG"

    # CHANGE OBJECT CHARACTERISTICS
    if len(sentence) > index + 1:
        if sentence[index + 1].pos_ == "NOUN":
            item.head = sentence[index + 1].text
            item.pos_ = "ADJ"
            item.tag_ = "ADJ|prenom|basis|met-e|stan"
            item.dep_ = "amod"
    return sentence_object


def two_nouns(sentence_object, index, item):
    """one noun being the quantifier of the other --> make sure the dependency is set correctly"""
    # eg: 'een hoop boeken'
    sentence = sentence_object.clause_list
    if index > 1:
        if (
            sentence[index - 1].pos_ == "NOUN"
            and not (sentence[index - 1].dep_ == "iobj" and item.dep_ == "obj")
            and item.tag_ == "N|soort|mv|basis"
        ):
            item.head = sentence[index - 1].text
        elif (
            sentence[index - 2].pos_ == "NOUN"
            and sentence[index - 1].head == item.text
            and not (sentence[index - 2].dep_ == "iobj" and item.dep_ == "obj")
            and item.tag_ == "N|soort|mv|basis"
        ):
            item.head = sentence[index - 2].text
    return sentence_object


def was(sentence_object, index, item):
    """Spacy does not correctly classify 'de was' as a noun instead of a verb"""
    sentence = sentence_object.clause_list
    # 'de was' >< 'ik was' >< not: 'hij was de president'
    if sentence[index - 1].text == "de" or (sentence[index - 1].text in demonstrative and "det" in item.tag_):
        sentence[index - 1].pos_ = "DET"
        sentence[index - 1].dep_ = "det"
        sentence[index - 1].tag_ = "LID|bep|stan|rest"
        sentence[index - 1].head = "was"
        item.new_form = "WAS"
        item.pos_ = "NOUN"
        item.dep_ = "nsubj" if "nsubj" not in [item2.dep_ for item2 in sentence] else "obj"
        item.lemma_ = "was"
        item.text = "was"
        item.tag_ = "N|soort|ev|basis|zijd|stan"
    else:
        item.new_form = ""
        item.lemma_ = "zijn"
        item.tag_ = "WW|pv|verl|ev"
        item.pos_ = "AUX"
        item.dep_ = "cop"
    return sentence_object


def lemma_not_infinitive(sentence_object, index, item):
    """some verbs that are not always handled correctly by Spacy"""
    if "WW" in item.tag_:
        item.lemma_ = (
            "kunnen"
            if item.lemma_ in can_conjugation
            else "hebben"
            if item.lemma_ in have_conjugation
            else "moeten"
            if item.lemma_ in must_conjugation
            else "mogen"
            if item.lemma_ in may_conjugation
            else "zijn"
            if item.lemma_ in be_conjugation
            else "worden"
            if item.lemma_ in worden_conjugation
            else "lijken"
            if item.lemma_ in lijken_conjugation
            else "blijven"
            if item.lemma_ in blijven_conjugation
            else "blijken"
            if item.lemma_ in blijken_conjugation
            else item.lemma_
        )
        item.new_form = (
            "KUNNEN"
            if item.lemma_ in can_conjugation
            else "HEBBEN"
            if item.lemma_ in have_conjugation
            else "MOETEN"
            if item.lemma_ in must_conjugation
            else "MOGEN"
            if item.lemma_ in may_conjugation
            else item.new_form
        )
    return sentence_object


def relative_clause_with_of(sentence_object, index, item):
    """'of' is not signed, but it needs the non manual marking of a question"""
    # eg: 'ze willen weten of het mag'
    item.new_form = "?"
    item.pos_ = "PUNCT"
    item.dep_ = "punct"
    item.tag_ = "LET"
    item.lemma_ = "?"
    item.text = "?"
    item.is_punct = True
    sentence_object.is_question = True
    return sentence_object


def er(sentence_object, index, item, all_preps):
    sentence = sentence_object.clause_list
    if item.position in sentence_object.subj_index:
        # 'er zijn' --> 'HEBBEN'
        if not all_preps:
            for index6, item6 in enumerate(sentence):
                if item6.lemma_ == "zijn" and "WW" in item6.tag_:
                    sentence[index6].new_form = "HEBBEN"
                    sentence_object.obj_indices = sentence_object.subj_index.copy()
                    sentence_object.subj_index = []
                    # eg: 'Daar is een bos. er zijn bloemen.' --> 'DAAR BOS // HEBBEN BLOEMEN'
        # if there is a prep, it is also possible, but then the sentence is reversed, so don't do this for now
        # eg: 'er zijn bloemen in het bos' --> 'BOS HEBBEN BLOEMEN'
    item.new_form = ""  # 'er' is not signed anyway
    return sentence_object


def richting(sentence, index, item):
    """if 'richting' is used as 'naar', spacy makes mistakes"""
    if index > 0 and len(sentence) > index + 1:
        if sentence[index - 1].pos_ != "DET" or (
            sentence[index - 2].lemma_ == "in" and sentence[index + 1].lemma_ == "van"
        ):
            item.new_form = "NAAR"
            item.text = "naar"
            item.lemma_ = "naar"
            item.head = sentence[index + 1].head if sentence[index + 1].pos_ == "DET" else sentence[index + 1].text
            # if it depends on a determiner, the model later does not remember which one
            # eg: het paard draaft richting het houten hek --> model chooses dependency on first 'het'
            item.tag_ = "VZ|init"
            item.pos_ = "ADP"
            item.dep_ = "case"
    if index > 1 and len(sentence) > index + 1:
        if sentence[index - 2].lemma_ == "in" and sentence[index + 1].lemma_ == "van":
            sentence[index - 2].new_form = ""
            sentence[index + 1].new_form = ""
    return sentence


def transform_other_mistakes(sentence_object, index, item, all_preps):
    sentence = sentence_object.clause_list

    # ER
    if item.lemma_ == "er":
        er(sentence_object, index, item, all_preps)

    # MISTAKES OF SPACY
    elif item.lemma_ == "uitgaaf" and item.text.lower() == "uitgaven":
        item.new_form = "UITGAVEN"

    elif item.lemma_ == "groter":
        item.lemma_ = "groot"
        item.new_form = "GROOT"

    # % = PROCENT
    elif item.lemma_ == "%":
        item.new_form = "PROCENT"
    elif "%" in item.new_form:
        item.new_form = re.sub(r"%", " PROCENT", item.new_form)

    # samen met
    elif item.lemma_ == "met":
        if index > 0:
            if sentence[index - 1].lemma_ == "samen":
                item.new_form = ""

    # JAAR OUD = JAAR
    elif item.lemma_ == "oud":
        if index > 0:
            if sentence[index - 1].lemma_ == "jaar":
                item.new_form = ""

    # VOLGENDE
    elif item.text == "volgend" or item.text == "volgende" and "WW" not in item.tag_:
        item.new_form = "VOLGENDE"

    # AF
    elif item.text == "het" and index > 1:
        if sentence[index - 2].text == "dat" and sentence[index - 1].text in ("was", "is"):
            sentence[index - 1].new_form = "AF"
    elif item.text.lower() == "voila":
        item.new_form = "AF"

    # WORD IS NOT CLEAR
    elif item.text.lower() == "morgens":
        # change ''s morgens' to 'OCHTEND' >< 'MORGEN' has different meaning
        item.new_form = "OCHTEND"
    elif item.lemma_ == "richting":
        richting(sentence, index, item)

    # NAAMGEBAR(EN)
    elif item.text.lower() == "naamgebaren":
        item.new_form = "NAAMGEBAAR"

    # van mij
    elif item.lemma_ in ("mij", "mezelf") and index > 0:
        if sentence[index - 1].lemma_ == "van":
            item.new_form = "VAN-1"
            sentence[index - 1].new_form = ""

    return sentence_object


def last_processing_step(sentence_object):
    sentence = sentence_object.clause_list
    for index, item in enumerate(sentence):
        # kijken
        if item.lemma_ in ("kijken", "zien", "uitkijk"):
            item.new_form = "KIJKEN"
    return sentence_object
