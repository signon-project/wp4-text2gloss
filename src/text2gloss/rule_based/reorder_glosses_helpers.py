"""functions that are needed to reorder the glosses"""

from text2gloss.rule_based.word_lists import not_


def add_word_to_sorted_glosses(sentence_object, index):
    """add the index to the sorted list"""
    if not sentence_object.clause_list[index].new_position_assigned:
        sentence_object.sorted_glosses_indices.append(index)
    sentence_object.clause_list[index].new_position_assigned = True
    return sentence_object


def find_dependencies(sentence_list, head_index):
    """find words that depend on the given head,
    return a list of indices of those words"""
    clause_indices = []
    if head_index < len(sentence_list):
        word = sentence_list[head_index].text.lower()
        for item in sentence_list:
            if (
                item.head.lower() == word
                and (
                    "prenom" in item.tag_
                    or not ("WW" in item.tag_ and sentence_list[head_index].pos_ in ["NOUN", "PRON", "PROPN"])
                )
                and not item.is_punct
                and item.position < len(sentence_list)
            ):
                clause_indices.append(item.position)
    return clause_indices


def find_clause_with_head(sentence_list, head_index):
    """find all the words that depend on the head (recursively) + sort indices"""
    clause_indices = [head_index]
    done = False
    while not done:
        for index in clause_indices:
            new_dependencies = find_dependencies(sentence_list, index)
            done = True
            for index2 in new_dependencies:
                if index2 not in clause_indices:
                    done = False
            clause_indices += [index for index in new_dependencies if index not in clause_indices]
    clause_indices = list(dict.fromkeys(clause_indices))
    clause_indices.sort()
    return clause_indices


def move_plural_indicator(sentence_object, clause_indices):
    """if there is a noun indicating the plural, it usually comes after the plural noun
    eg: een hoop stenen --> STEEN HOOP"""
    clause_indices_new = clause_indices.copy()
    sentence = sentence_object.clause_list
    for index in clause_indices:
        if sentence[index].pos_ == "NOUN" and "mv" in sentence[index].tag_:
            plural_noun_clause_indices = find_dependencies(sentence, index) + [index]
            for index2 in clause_indices:
                if (
                    index != index2
                    and sentence[index2].pos_ == "NOUN"
                    and "case"
                    not in [
                        token.dep_
                        for token in sentence[min(plural_noun_clause_indices) : max(plural_noun_clause_indices)]
                    ]
                    and (sentence[index - 1].pos_ == "NOUN" or sentence[min(clause_indices) - 1].pos_ == "NOUN")
                ):
                    # if there is a noun that our plural noun depends on, eg: een hoop stenen
                    if index2 in plural_noun_clause_indices:
                        plural_noun_clause_indices.remove(index2)
                    clause_indices_new = plural_noun_clause_indices + [
                        index4 for index4 in clause_indices if index4 not in plural_noun_clause_indices
                    ]
    return clause_indices_new


def add_clause_to_sorted_glosses(sentence_object, head_index):
    """add full clause to the list of sorted glosses"""
    clause_indices = find_clause_with_head(sentence_object.clause_list, head_index)
    clause_indices = move_plural_indicator(sentence_object, clause_indices)
    # clause_indices = move_adj(sentence_object, clause_indices)
    for index in clause_indices:
        add_word_to_sorted_glosses(sentence_object, index)
    return sentence_object


def add_verb_to_sorted_glosses(sentence_object, head_index):
    """make a seperate function,
    because we only want to add NIET or adverbs but not other words that depend on the verb"""
    sentence = sentence_object.clause_list
    clause_indices = find_dependencies(sentence_object.clause_list, head_index)
    adv_indices = [
        index
        for index in clause_indices
        if sentence[index].lemma_ in not_
        or (sentence[index].dep_ in ["advmod", "xcomp"] and sentence[index].pos_ in ["ADJ", "ADV"])
        and index not in sentence_object.question_word_indices
    ]
    clause_indices_new = adv_indices.copy()

    for index in adv_indices:
        # eg heel laag
        dependencies = find_clause_with_head(sentence, index)
        clause_indices_new += [index3 for index3 in dependencies if sentence[index3].dep_ in ["advmod", "xcomp"]]
    clause_indices_new.sort()
    clause_indices_new += [head_index]
    for index in clause_indices_new:
        add_word_to_sorted_glosses(sentence_object, index)
    return sentence_object


##############################


def find_relative_clause(sentence_list, rel_pronoun_index):
    """find all the words that depend on the head (recursively) + sort indices"""
    try:
        index_of_head_of_rel_pron = [token.text for token in sentence_list].index(
            sentence_list[rel_pronoun_index].head
        )
        clause_indices = find_clause_with_head(sentence_list, index_of_head_of_rel_pron)
        clause_indices.sort()
        return clause_indices
    except Exception:
        return []


def head_of_sub_clause_before(sentence_list, conj_index):
    """check whether the head of a clause is before or after the clause
    eg: 'voordat hij naar buiten ging, ...' >< '..., voordat hij naar buiten ging'"""
    try:
        index_of_head_of_conj = [token.text for token in sentence_list].index(sentence_list[conj_index].head)
        index_of_head_clause = [token.text for token in sentence_list].index(sentence_list[index_of_head_of_conj].head)
        if index_of_head_clause < conj_index:
            return True
        else:
            return False
    except Exception:
        return False
