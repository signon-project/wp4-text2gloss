"""split up complex sentences into phrases"""
from text2gloss.rule_based.complex_sents_helpers import plural_with_apostrophe, split_based_on_conjunction_words
from text2gloss.rule_based.reorder_glosses import reorder_glosses


def make_complex_gloss_sentence(complex_sentence, nlp):
    """split into clausses and transform each clause with 'reorder_glosses'
    then concatenate all clauses with '//'"""
    if isinstance(complex_sentence, str):
        complex_sentence = plural_with_apostrophe(complex_sentence)
        clauses = split_based_on_conjunction_words(complex_sentence, nlp)
        complex_gloss_sentence_list = []
        for clause in clauses:
            sentence_object = reorder_glosses(clause)
            glosses = [
                str(sentence_object.clause_list[index].new_form)
                for index in sentence_object.sorted_glosses_indices
                if sentence_object.clause_list[index].new_form != ""
            ]
            complex_gloss_sentence_list += glosses
            # complex_gloss_sentence_list.append('//')
        # complex_gloss_sentence_list = complex_gloss_sentence_list[:-1] # delete the last '//'
        return complex_gloss_sentence_list
    else:
        return [str(complex_sentence)]
