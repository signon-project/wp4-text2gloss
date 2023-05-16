import re
from reorder_glosses_helpers import find_relative_clause, head_of_sub_clause_before
from helpers import make_sentence_list, Word
from reorder_glosses import reorder_glosses
import copy
from transform_comp import transform_comparative
from transform_neg import no_negation_in_question
from word_lists import question_words, and_or, more_less, what, who, that, conjunctions_then_after, \
    conjunctions_then_before, conjunctions_if, conjunctions_while
from transform_conj import transform_while
from re import sub


def is_true_split(sentence, index, item):
    """check whether the conjunction is really marking a split.
    eg: NOT 'ik koop bananen en peren.' // 'ik zoek en vindt mijn schoenen.'"""

    if item.lemma_ in and_or:
        if item.tag_ == 'VG|onder':
            return True
        for item2 in sentence[:index]:
            if 'WW' in item2.tag_:
                for item3 in sentence[index:]:
                    if 'WW' in item3.tag_:
                        if 0 < index < len(sentence) - 3:
                            if (sentence[index - 1].pos_ == sentence[index + 1].pos_
                                and sentence[index - 1].tag_ == sentence[index + 1].tag_) \
                                    or (sentence[index - 1].pos_ == sentence[index + 2].pos_
                                        and sentence[index - 1].tag_ == sentence[index + 2].tag_) \
                                    or (sentence[index - 1].pos_ in ('NOUN')
                                        and sentence[index + 1].pos_ in ('NOUN', 'PROPN', 'PRON')) \
                                    or (sentence[index - 1].pos_ in ('PROPN', 'PRON')
                                        and sentence[index + 1].pos_ in ('PROPN', 'PRON')) \
                                    or (sentence[index - 1].pos_ in ('NOUN', 'PROPN', 'PRON')
                                        and sentence[index + 2].pos_ in ('NOUN', 'PROPN', 'PRON')
                                        and sentence[index + 1].head == sentence[index + 2].text):
                                return False
                        elif 0 < index < len(sentence) - 2:
                            if (sentence[index - 1].pos_ == sentence[index + 1].pos_
                                and sentence[index - 1].tag_ == sentence[index + 1].tag_) \
                                    or (sentence[index - 1].pos_ in ('NOUN')
                                        and sentence[index + 1].pos_ in ('NOUN', 'PROPN', 'PRON')) \
                                    or (sentence[index - 1].pos_ in ('PROPN', 'PRON')
                                        and sentence[index + 1].pos_ in ('PROPN', 'PRON')):
                                return False

                        # eg: 'hij zoekt en vindt ...' / 'de man en de vrouw ...' / 'Yves en zijn vrouw ...'
                        return True
    elif item.lemma_ == ',':
        if index < len(sentence) - 1:
            if sentence[index + 1].dep_ == 'conj' or (item.dep_ == 'flat' and sentence[index + 1].dep_ == 'flat'):
                return False
                # eg: 'een nieuw, arm land'
        if 0 < index < len(sentence) - 1:
            if sentence[index - 1].pos_ == 'ADJ' and sentence[index + 1].pos_ == 'ADJ':
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
            if sentence[index - 1].lemma_ in more_less: # and 'TW' in sentence[index + 1].tag_:
                return False
                # 'meer dan 100' is no comparison - 'meer' and 'minder' have their own sign
        if sentence[index + 1].lemma_ in what:
            return False
            # '... dan wat ...' is no comparison
    if item.pos_ == 'SCONJ' and item.dep_ == 'advmod' and item.tag_ == 'VG|onder' \
        and 'WW' not in [item2.tag_[0:2] for item2 in sentence[index:]]:
        return False
        # if there is no verb in the part after 'dan', this is no separate clause
    return True


def split_relative_clause(sentence, index, item, sentence_new, take_outs):
    """find the beginning and end of the relative clause and add the word positions to 'take_outs' """
    rel_clause_indices = find_relative_clause(sentence, index)
    if rel_clause_indices:
        rel_clause_indices = [num for num in range(min(rel_clause_indices) + 1, max(rel_clause_indices) + 1)]
        # min + 1 because the rel pronoun will first be put at the front of the rel clause
        head_index = [index - 1]
        for m in range(index - 1, -1, -1):
            if sentence[m].pos_ in ('NOUN', 'PRON', 'PROPN'):
                head_index = [m]
                # eg: ignore 'te helpen' in  'om de man te helpen die de straat over wil steken.'
                break

        # ADD WORDS MODIFYING THE HEAD
        for index2 in range(head_index[0] - 1, 0, -1):
            # eg: "We kunnen in de huidige situatie waar het virus wijdverspreid is, niet toelaten dat ..."
            if sentence[index2].head == sentence[head_index[0]].text and 'WW' not in sentence[index2].tag_:
                head_index.append(index2)
            else:
                break

        if item.lemma_ in who:
            # eg: 'de man voor wie ik het gekocht heb ...'
            head_index += [index - 1]
            sentence_new[index - 2].dep_ = 'obl'                            # can never be the subject
            sentence_new[index - 1].head = sentence_new[index - 2].text
            sentence_new[index - 2].new_form += ' WG-3'                     # place in space
            sentence_new.insert(index - 2, sentence_new.pop(index - 1))     # put preposition before head
            sentence_new[index].new_form = ''
        else:
            if index > 1:
                if sentence[index - 1].dep_ == 'conj':
                    start = [item2.position for item2 in sentence
                             if item2.text == sentence[index - 1].head]
                    if start:
                        start = start[0]
                        start = start - 1 if sentence[start - 1].dep_ == 'case' else start
                        head_index += [n for n in range(start, index)]
            if item.lemma_ not in that:
                sentence_new[head_index[0]].dep_ = 'obj'
                # het plan waarmee wij willen winnen --> plan = object in the relative clause
            else:
                sentence_new[head_index[0]].dep_ = 'obj' if item.dep_ == 'obj' \
                    else 'iobj' if item.dep_ == 'iobj' else 'nsubj' \
                    if 'subj' in item.dep_ else sentence_new[index - 1].dep_

            # if 'waar-': change to prep (eg: waarop = op) + 'mee' in 'waarmee' --> MET
            sentence_new[index].new_form = (sub(r'waar', '', item.text)).upper() if 'waar' in item.text else ''
            sentence_new[index].new_form = sub(r'MEE', 'MET', sentence_new[index].new_form)

            sentence_new[index].head = sentence[min(head_index)].text
            sentence_new[index].pos_ = 'ADP'
            sentence_new[index].dep_ = 'case'
            sentence_new[index].tag_ = 'VZ|init'
            sentence_new[index].text = sub(r'waar', '', item.text) if 'waar' in item.text else ''
            sentence_new[index].lemma_ = sub(r'waar', '', item.text) if 'waar' in item.text else ''
            sentence_new[head_index[0]].new_form += ' WG-3'                     # place in space

            # put it at the front
            for n in range(index, min(head_index) - 1, -1):
                sentence_new[n].position += 1
                if n in head_index:
                    head_index[head_index.index(n)] += 1
            sentence_new[index].position = min(head_index) - 1
            sentence_new.insert(min(head_index) - 1, sentence_new.pop(index))

        rel_clause_indices = [min(head_index) - 1] + head_index + rel_clause_indices
        take_outs.append(rel_clause_indices)
        sentence_new.insert(max(rel_clause_indices) + 1, Word('WG-3', 'WG-3', 'WG-3',
                                                              max(rel_clause_indices) + 1,
                                                              'subj',
                                                              sentence[index - 1].head, item.tag_,
                                                              item.pos_, item.is_punct, item.is_space))
        for item8 in sentence_new[max(rel_clause_indices) + 2:]:
            item8.position += 1
        for n in range(min(head_index) - 2, -1, -1):
            # put dependencies to new word
            if sentence_new[n].head == sentence[index - 1].text:
                sentence_new[n].head = 'WG-3'
    return sentence, sentence_new, take_outs


def split_te_verb(sentence, index, item, splits, start_of_next_clause):
    """split if there is a 'te + infinitive' """
    if len(sentence) > index + 1:
        if 'WW' in sentence[index + 1].tag_ and not 'om' in [item2.lemma_ for item2 in sentence]:
            for n in range(index, -1, -1):
                if 'WW' in sentence[n].tag_:
                    splits.append([start_of_next_clause, n + 1])
                    start_of_next_clause = n + 1
                    break
    return splits, start_of_next_clause


# MAIN FUNCTION
def split_based_on_conjunction_words(raw_sentence, nlp, print_output=False):
    """main function that splits complex sentences into separate clauses
    that each can be transformed into glosses separately"""

    # MAKE SPACY DOC
    sentence_doc = nlp(raw_sentence)
    sentence_doc_list = tuple(token for token in sentence_doc if not token.is_space)
    sentence = make_sentence_list(sentence_doc_list)
    sentence_new = copy.deepcopy(sentence)

    # PRINT
    if print_output:
        print([token.pos_ for token in sentence])
        print([token.dep_ for token in sentence])
        print([token.tag_ for token in sentence])
        print([token.head for token in sentence])
        print([token.lemma_ for token in sentence])

    # NEEDED VARIABLES
    splits = []
    separated_clauses = []
    take_outs_rel = []
    take_outs_brackets = []
    start_of_next_clause = 0

    for index, item in enumerate(sentence):
        if index > 0:

            # SPLIT RELATIVE CLAUSES
            # change relative clauses to an extra clause at the front,
            # add a WG to the head of the relative clause and add the same WG in the next clause to make the connection
            if 'betr' in item.tag_ or (('waar' in item.lemma_ or 'wie' in item.lemma_)
                                       and 'acl:relcl' in [item7.dep_ for item7 in sentence]):
                # eg: 'de man die ...' / 'de man voor wie ...' / 'de kast waarin ...'
                sentence, sentence_new, take_outs_rel = \
                    split_relative_clause(sentence, index, item, sentence_new, take_outs_rel)

            # SPLIT QUESTION WORD IN THE MIDDLE OF THE SENTENCE
            # eg: 'weet jij waarom ze dat doen?'
            if item.lemma_ in question_words and index > 1 \
                    and not (sentence[index-1].lemma_ == 'heel' and item.lemma_ == 'wat'):
                # if it is one of the first 2 words, it is just a question
                splits.append([start_of_next_clause, index])
                start_of_next_clause = index

            # SPLIT BASED ON CONJUNCTION WORDS
            if item.pos_ in ('CCONJ', 'SCONJ') and index - 1 not in [item2 for sublist in splits for item2 in sublist]:
                if item.lemma_ == 'dan':
                    if is_true_comparison_split(sentence, index, item):
                        splits.append([start_of_next_clause, index])
                        start_of_next_clause = index

                elif is_true_split(sentence, index, item):
                    splits.append([start_of_next_clause, index])
                    start_of_next_clause = index

            # SPLIT BASED ON PUNCTUATION
            if item.lemma_ in (',', '.', ';', ':', '!', '?') \
                    and index not in tuple(item2 for sublist in splits for item2 in sublist):
                # not all PUNCT
                if is_true_split(sentence, index, item) \
                        and index - 1 not in tuple(item3 for sublist in take_outs_rel for item3 in sublist):
                    splits.append([start_of_next_clause, index + 1])
                    start_of_next_clause = index + 1

            if '(' in item.lemma_:
                take_out_brackets = []
                for n in range(index, len(sentence)):
                    if ')' not in sentence[n].lemma_:
                        take_out_brackets.append(n)
                    else:
                        take_out_brackets.append(n)
                        break
                take_outs_brackets.append(take_out_brackets)

            # SPLIT BASED ON SPECIFIC CONSTRUCTION
            # 'om te' + 'te + infinitive'
            if item.lemma_ == 'om' and item.dep_ == 'mark' \
                    and index - 1 not in tuple(item for sublist in take_outs_rel for item in sublist):
                splits.append([start_of_next_clause, index])
                start_of_next_clause = index + 1
            if item.lemma_ == 'te' and not take_outs_rel:
                splits, start_of_next_clause = split_te_verb(sentence, index, item, splits, start_of_next_clause)

        # CHANGES THAT DEPEND ON THE WHOLE COMPLEX SENTENCE
        no_negation_in_question(sentence, sentence_new, index, item)

    # if the last clause is not yet added to the splits list:
    if start_of_next_clause < len(sentence_new):
        splits.append([start_of_next_clause, len(sentence_new)])

    # PUT RELATIVE CLAUSES AT THE FRONT
    for clause_list in take_outs_rel:
        clause = [word for word in sentence_new if word.position in clause_list]
        separated_clauses.append(clause)

    splits_new = copy.deepcopy(splits)
    for index2, number_couple in enumerate(splits):
        # CHANGE COMPARING CLAUSES
        if sentence_new[number_couple[0]].text == 'dan' and sentence_new[number_couple[0]].pos_ == 'SCONJ' \
                and sentence_new[number_couple[0]].dep_ in ('mark', 'fixed'):
            splits_new, sentence_new = transform_comparative(sentence_new, index2, splits)

        # CHANGE WHILE CLAUSE
        if sentence_new[number_couple[0]].text in conjunctions_while:
            sentence_new, splits_new = transform_while(sentence_new, index2, number_couple, splits_new)

        # IF TEMPORAL CLAUSE: CHECK ORDER
        # chronological order is preferred in VGT
        if sentence_new[number_couple[0]].lemma_ in conjunctions_then_after:
            addition = Word('dan', 'DAN', 'dan', splits[index2 - 1][0], 'advmod', 'dan', 'BW', 'ADV', False,
                            False)  # head not correct # AF DAN
            # check whether clause head is before or after
            if head_of_sub_clause_before(sentence, number_couple[0]) or len(splits) == index2 + 1:
                sentence_new.insert(splits[index2 - 1][0], addition)
                sentence_new.pop(number_couple[0] + 1)
                splits_new[index2 - 1][1] += 1
                splits_new[index2][0] += 1
                splits_new.insert(index2 - 1, splits_new.pop(index2))
            else:
                if index2 < len(splits) - 1:
                    sentence_new.insert(splits[index2 + 1][0], addition)
                    sentence_new.pop(number_couple[0])
                    splits_new[index2][1] -= 1
                    splits_new[index2 + 1][0] -= 1
        if sentence_new[number_couple[0]].lemma_ in conjunctions_then_before:
            # check whether clause head is before or after
            if not head_of_sub_clause_before(sentence, number_couple[0]):
                splits_new.insert(index2 + 1, splits_new.pop(index2))

        # IF CONDITIONAL CLAUSE: CHECK ORDER
        # Condition must come before the main clause
        print(splits_new)
        print(index2)
        if sentence_new[number_couple[0]].lemma_ in conjunctions_if and 0 < index2 < len(splits_new):
            splits_new.insert(index2 - 1, splits_new.pop(index2))

        if sentence_new[number_couple[0]].text == 'dan' and sentence_new[number_couple[0]].pos_ == 'SCONJ' \
                and sentence_new[number_couple[0]].dep_ in ['mark', 'fixed']:
            splits_new, sentence_new = transform_comparative(sentence_new, index2, splits)

    # APPLY SPLITS
    for index3, number_couple in enumerate(splits_new):
        clause = [word for word in sentence_new[number_couple[0]:number_couple[1]]
                  if word.position not in [item for sublist in take_outs_rel for item in sublist]
                  and word.position not in [item for sublist in take_outs_brackets for item in sublist]]
        if clause:
            separated_clauses.append(clause)

    # ADD BRACKET CLAUSE AS THE LAST CLAUSE
    for clause_list in take_outs_brackets:
        clause = [word for word in sentence_new if word.position in clause_list]
        separated_clauses.append(clause)

    # CHANGE POSITIONS
    # every clause starts with position 0
    for clause3 in separated_clauses:
        index4 = 0
        for item3 in clause3:
            item3.position = index4
            index4 += 1
    return separated_clauses


def plural_with_apostrophe(sentence):
    """spacy recognises a ''s' as a new word, so just append without ''' to the previous word """
    from re import sub

    sentence_new = ''
    for word in sentence.split(' '):
        sentence_new += sub(r'\'s', 's', word) + ' ' if word != '\'s' else word + ' '
    return sentence