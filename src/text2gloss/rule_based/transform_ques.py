"""some transformations for questions with question word"""
# probleem als complexe zin --> enkel laatste deel heeft ? eg: Was hij niet op de plaats van de misdaad als getuige?

from word_lists import question_words_with_noun
from reorder_glosses_helpers import find_clause_with_head


def transform_question_word_clause(sentence_object):
    """question words (QW) signs at the front of the sentence.
    - 'hoe laat' = 1 sign"""
    sentence = sentence_object.clause_list

    if sentence_object.question_word_indices:
        question_word = sentence[sentence_object.question_word_indices[-1]]
        if question_word.position < len(sentence) - 1:
            next_word = sentence[sentence_object.question_word_indices[-1] + 1]
            if question_word.lemma_ == 'hoe' and next_word.text in ['laat', 'vroeg']:
                sentence_object.question_word_indices.append(next_word.position)
                next_word.new_form = ''
                # spacy changes 'laat' in a verb
                question_word.new_form = 'HOELAAT'
            elif question_word.text in question_words_with_noun:
                noun_head = question_word.head
                index_head = [index2 for index2, item2 in enumerate(sentence) if item2.text == noun_head]
                if index_head:
                    index_head = index_head[0]
                    clause_indices = find_clause_with_head(sentence, index_head)
                    clause_indices_new = [index for index in clause_indices
                                          if index not in sentence_object.question_word_indices]
                    for index3 in clause_indices_new:
                        sentence_object.question_word_indices.append(index3)
    return sentence_object
