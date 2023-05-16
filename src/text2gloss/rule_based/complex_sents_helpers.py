from re import sub

from text2gloss.rule_based.helpers import Word
from text2gloss.rule_based.reorder_glosses_helpers import find_relative_clause
from text2gloss.rule_based.word_lists import and_or, more_less, that, what, who


def is_true_split(sentence, index, item):
    """check whether the conjunction is really marking a split.
    eg: NOT 'ik koop bananen en peren.' // 'ik zoek en vindt mijn schoenen.'"""

    if item.lemma_ in and_or:
        if item.tag_ == "VG|onder":
            return True
        for item2 in sentence[:index]:
            if "WW" in item2.tag_:
                for item3 in sentence[index:]:
                    if "WW" in item3.tag_:
                        if 0 < index < len(sentence) - 3:
                            if (
                                (
                                    sentence[index - 1].pos_ == sentence[index + 1].pos_
                                    and sentence[index - 1].tag_ == sentence[index + 1].tag_
                                )
                                or (
                                    sentence[index - 1].pos_ == sentence[index + 2].pos_
                                    and sentence[index - 1].tag_ == sentence[index + 2].tag_
                                )
                                or (
                                    sentence[index - 1].pos_ in ("NOUN")
                                    and sentence[index + 1].pos_ in ("NOUN", "PROPN", "PRON")
                                )
                                or (
                                    sentence[index - 1].pos_ in ("PROPN", "PRON")
                                    and sentence[index + 1].pos_ in ("PROPN", "PRON")
                                )
                                or (
                                    sentence[index - 1].pos_ in ("NOUN", "PROPN", "PRON")
                                    and sentence[index + 2].pos_ in ("NOUN", "PROPN", "PRON")
                                    and sentence[index + 1].head == sentence[index + 2].text
                                )
                            ):
                                return False
                        elif 0 < index < len(sentence) - 2:
                            if (
                                (
                                    sentence[index - 1].pos_ == sentence[index + 1].pos_
                                    and sentence[index - 1].tag_ == sentence[index + 1].tag_
                                )
                                or (
                                    sentence[index - 1].pos_ in ("NOUN")
                                    and sentence[index + 1].pos_ in ("NOUN", "PROPN", "PRON")
                                )
                                or (
                                    sentence[index - 1].pos_ in ("PROPN", "PRON")
                                    and sentence[index + 1].pos_ in ("PROPN", "PRON")
                                )
                            ):
                                return False

                        # eg: 'hij zoekt en vindt ...' / 'de man en de vrouw ...' / 'Yves en zijn vrouw ...'
                        return True
    elif item.lemma_ == ",":
        if index < len(sentence) - 1:
            if sentence[index + 1].dep_ == "conj" or (item.dep_ == "flat" and sentence[index + 1].dep_ == "flat"):
                return False
                # eg: 'een nieuw, arm land'
        if 0 < index < len(sentence) - 1:
            if sentence[index - 1].pos_ == "ADJ" and sentence[index + 1].pos_ == "ADJ":
                return False
        return True
    else:
        return True
    return False


def is_true_comparison_split(sentence, index, item):
    """check if 'dan' is truly marking a split
    eg NOT: 'meer dan 100 inwoners' - 'meer dan wat we gedacht hadden'"""

    if len(sentence) > index + 1:
        if index > 0:
            if sentence[index - 1].lemma_ in more_less:  # and 'TW' in sentence[index + 1].tag_:
                return False
                # 'meer dan 100' is no comparison - 'meer' and 'minder' have their own sign
        if sentence[index + 1].lemma_ in what:
            return False
            # '... dan wat ...' is no comparison
    if (
        item.pos_ == "SCONJ"
        and item.dep_ == "advmod"
        and item.tag_ == "VG|onder"
        and "WW" not in [item2.tag_[0:2] for item2 in sentence[index:]]
    ):
        return False
        # if there is no verb in the part after 'dan', this is no separate clause
    return True


def split_relative_clause(sentence, index, item, sentence_new, take_outs):
    """find the beginning and end of the relative clause and add the word positions to 'take_outs'"""
    rel_clause_indices = find_relative_clause(sentence, index)
    if rel_clause_indices:
        rel_clause_indices = [num for num in range(min(rel_clause_indices) + 1, max(rel_clause_indices) + 1)]
        # min + 1 because the rel pronoun will first be put at the front of the rel clause
        head_index = [index - 1]
        for m in range(index - 1, -1, -1):
            if sentence[m].pos_ in ("NOUN", "PRON", "PROPN"):
                head_index = [m]
                # eg: ignore 'te helpen' in  'om de man te helpen die de straat over wil steken.'
                break

        # ADD WORDS MODIFYING THE HEAD
        for index2 in range(head_index[0] - 1, 0, -1):
            # eg: "We kunnen in de huidige situatie waar het virus wijdverspreid is, niet toelaten dat ..."
            if sentence[index2].head == sentence[head_index[0]].text and "WW" not in sentence[index2].tag_:
                head_index.append(index2)
            else:
                break

        if item.lemma_ in who:
            # eg: 'de man voor wie ik het gekocht heb ...'
            head_index += [index - 1]
            sentence_new[index - 2].dep_ = "obl"  # can never be the subject
            sentence_new[index - 1].head = sentence_new[index - 2].text
            sentence_new[index - 2].new_form += " WG-3"  # place in space
            sentence_new.insert(index - 2, sentence_new.pop(index - 1))  # put preposition before head
            sentence_new[index].new_form = ""
        else:
            if index > 1:
                if sentence[index - 1].dep_ == "conj":
                    start = [item2.position for item2 in sentence if item2.text == sentence[index - 1].head]
                    if start:
                        start = start[0]
                        start = start - 1 if sentence[start - 1].dep_ == "case" else start
                        head_index += [n for n in range(start, index)]
            if item.lemma_ not in that:
                sentence_new[head_index[0]].dep_ = "obj"
                # het plan waarmee wij willen winnen --> plan = object in the relative clause
            else:
                sentence_new[head_index[0]].dep_ = (
                    "obj"
                    if item.dep_ == "obj"
                    else "iobj"
                    if item.dep_ == "iobj"
                    else "nsubj"
                    if "subj" in item.dep_
                    else sentence_new[index - 1].dep_
                )

            # if 'waar-': change to prep (eg: waarop = op) + 'mee' in 'waarmee' --> MET
            sentence_new[index].new_form = (sub(r"waar", "", item.text)).upper() if "waar" in item.text else ""
            sentence_new[index].new_form = sub(r"MEE", "MET", sentence_new[index].new_form)

            sentence_new[index].head = sentence[min(head_index)].text
            sentence_new[index].pos_ = "ADP"
            sentence_new[index].dep_ = "case"
            sentence_new[index].tag_ = "VZ|init"
            sentence_new[index].text = sub(r"waar", "", item.text) if "waar" in item.text else ""
            sentence_new[index].lemma_ = sub(r"waar", "", item.text) if "waar" in item.text else ""
            sentence_new[head_index[0]].new_form += " WG-3"  # place in space

            # put it at the front
            for n in range(index, min(head_index) - 1, -1):
                sentence_new[n].position += 1
                if n in head_index:
                    head_index[head_index.index(n)] += 1
            sentence_new[index].position = min(head_index) - 1
            sentence_new.insert(min(head_index) - 1, sentence_new.pop(index))

        rel_clause_indices = [min(head_index) - 1] + head_index + rel_clause_indices
        take_outs.append(rel_clause_indices)
        sentence_new.insert(
            max(rel_clause_indices) + 1,
            Word(
                "WG-3",
                "WG-3",
                "WG-3",
                max(rel_clause_indices) + 1,
                "subj",
                sentence[index - 1].head,
                item.tag_,
                item.pos_,
                item.is_punct,
                item.is_space,
            ),
        )
        for item8 in sentence_new[max(rel_clause_indices) + 2 :]:
            item8.position += 1
        for n in range(min(head_index) - 2, -1, -1):
            # put dependencies to new word
            if sentence_new[n].head == sentence[index - 1].text:
                sentence_new[n].head = "WG-3"
    return sentence, sentence_new, take_outs


def split_te_verb(sentence, index, item, splits, start_of_next_clause):
    """split if there is a 'te + infinitive'"""
    if len(sentence) > index + 1:
        if "WW" in sentence[index + 1].tag_ and "om" not in [item2.lemma_ for item2 in sentence]:
            for n in range(index, -1, -1):
                if "WW" in sentence[n].tag_:
                    splits.append([start_of_next_clause, n + 1])
                    start_of_next_clause = n + 1
                    break
    return splits, start_of_next_clause


def plural_with_apostrophe(sentence):
    """spacy recognises a ''s' as a new word, so just append without ''' to the previous word"""
    from re import sub

    sentence_new = ""
    for word in sentence.split(" "):
        sentence_new += sub(r"\'s", "s", word) + " " if word != "'s" else word + " "
    return sentence
