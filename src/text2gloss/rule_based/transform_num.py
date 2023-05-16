"""transform numbers to multiple glosses in the correct order (order same as Dutch)"""

import re
from random import randint

import telwoord
from text2gloss.rule_based.word_lists import (
    big_units,
    big_units_numbers,
    bigger_units,
    double_numbers,
    double_signs,
    number_in_numbers,
    number_in_words,
    one_options,
    small_units,
)


def to_numbers(m, number_written_out):
    # # CHANGE NUMBER IN WORDS INTO SIGNS WITH SPACES
    matched = False
    for index2, double in enumerate(double_numbers):
        if double in m:
            # these ones are signed differently: eg: 22 = 2 2 not 2 20
            matched = True
            m = re.sub(double, double_signs[index2], m)
    if not matched:
        for unit in bigger_units:
            m = re.sub(unit, unit + " ", m)
            m = re.sub(r" [eë]n", " ", m)
        for unit2 in small_units:
            m = (
                re.sub(unit2, unit2 + " ", m)
                if not re.search(rf"{unit2}t", m) and not re.search(rf"{unit2}i", m)
                else m
            )  # tacht-ig & negen-tig
            m = re.sub(r" [eë]n", " ", m)
    for unit3 in big_units:
        m = re.sub(unit3, unit3 + " ", m)
    for one in one_options:
        if one in number_written_out:
            m = re.sub(one, "één", m)  # this is the representation in the 'VGT woordenboek'

    if re.search(r"[0-9]", m):
        m = m.upper()
    else:
        m_new = ""
        for part in re.split(r"[ ^]", m):
            if part in big_units:
                m_new += big_units_numbers[big_units.index(part)] + " "
            elif part == "en":
                m_new += "+++"
            elif part in one_options:
                m_new += "1 "
            else:
                try:
                    m_new += str(number_in_numbers[number_in_words.index(part)]) + " "
                except Exception:
                    m_new += part
        m = m_new

    return m


def transform_numbers(sentence_object, index, item):
    """split up number into each sign"""
    # sentence = sentence_object.sentence_list
    number = item.text.lower()  # sometimes the lemma is wrong (eg: een --> e)
    to_add = ""

    # ORDINAL NUMBERS WITH THERE OWN SIGN
    if re.search(r"^[1-4]s?t?d?[eE]", number) or number.lower() in ["eerst", "eerste", "tweede", "derde", "vierde"]:
        number = re.sub(r"1s?t?e", "EERSTE", number)
        number = re.sub(r"2d?e", "TWEEDE", number)
        number = re.sub(r"3d?e", "DERDE", number)
        number = re.sub(r"4d?e", "VIERDE", number)
        item.new_form = number.upper()
        return sentence_object  # don't do anything else

    # OTHER ORDINAL NUMBERS
    elif re.search(r"[^ei]e$", number):
        add_de_or_ste = randint(0, 1)
        if re.search(r"de$", number):
            number = re.sub(r"de$", "", number)
            if add_de_or_ste == 1:
                to_add = ""  # '^DE' (= vuist met duim omhoog)
        elif re.search(r"ste$", number):
            number = re.sub(r"ste$", "", number)
            if add_de_or_ste == 1:
                to_add = ""  # '^STE' (= e-hand)
        elif re.search(r"e$", number):
            number = re.sub(r"e$", "", number)
            if add_de_or_ste == 1:
                to_add = ""  # '^E' eg: 25e = STE / 5e = DE
        # don't add these, because they are not used by all signers, is something personal

    # SPLIT UP PER SIGN
    # (work with written out numbers because that is easier to spot patterns eg: 22 000 >< 2 200)
    number_list = (
        [re.sub(r"\.", "", number)]
        if re.search(r"\.[0-9][0-9][0-9]", number)
        else number.split(".")
        if re.search(r"[.]", number)
        else [number]
    )
    # delete . for thousand, not if it is a comma
    # --> will make mistakes when there are 3 numbers after the comma
    number_new = []
    for n in number_list:
        number_new += n.split(",")
    full_number = []
    for index_m, m in enumerate(number_new):
        # # PUT NUMBERS INTO WORDS
        try:
            integer = int(m)
            number_written_out = telwoord.cardinal(integer, friendly=False)
            m = telwoord.cardinal(integer, friendly=False)
        except Exception:
            number_written_out = item.text

        if " " in m:
            part_of_number = ""
            for o in m.split(" "):
                m = to_numbers(o, number_written_out)
                part_of_number += m
            full_number.append(part_of_number)
        else:
            m = to_numbers(m, number_written_out)
            m = re.sub(r" +", "^", m)
            if to_add:
                m = m[:-1]
            m = re.sub(r"^$", "", m)
            full_number.append(m)

    item.new_form = " KOMMA ".join(full_number)
    item.new_form += to_add
    if re.search(r"[^e]en$", item.new_form):
        item.new_form = re.sub(r"en$", "+++", item.new_form)
    item.new_form = re.sub("\^", " ", item.new_form)
    return sentence_object
