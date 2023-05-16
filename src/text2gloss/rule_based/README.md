# Rule-based system translating Dutch to VGT gloss-annotations

By Maud Goddefroy. Modified by Bram Vanroy

## Description
This project contains the scripts for translating Dutch to VGT gloss-annotations using a rule-based system.

## Usage
You can put an example sentence or input a file in 'translate.py'.

## Summary
	- First splits up the sentence into simple sentences if it is a complex sentence
		○ Some extra changes are made
			§ To split up a relative clauses into a separate but linked sentence
			§ To represent comparing expressions correctly
			§ Ordering events chronologically
			§ Delete negations in questions
	- Then for each simple sentence:
		○ An object is created that can store all the necessary info on the sentence level = Sentence
			§ The indices of
				□ The subject                                           = subj_index
				□ The indirect object                                   = iobj_index
				□ The other objects                                     = obj_index
				□ All the verbs                                         = verb_indices
				□ The words in the temporal clause                      = temporal_indices
				□ The words in the question word clause                 = question_word_indices
				□ The sorted glosses                                    = sorted_glosses_indices
			§ Boolean value indicating whether the sentence is a question = is_question
			§ A list of Word objects:
				□ Each word is transformed into an object as well, which is a modification of the word-object of spacy. For each word, we store:
					® The original form                          = text
					® The position in the sentence               = position
					® The gloss representation                   = new_form, which is initialized to be the lemma
					® From spacy: the dep_, head, tag_, pos_, is_punct and is_space, lemma_
					® Boolean value indicating whether the word got assigned a position in the gloss sentence new_position_assigned
		○ The system looks for several phrases and remembers the positions of those words
			§ any temporal phrases 
			§ All verbs that can be signed. Auxiliary verbs are thus not included
			§ Any question words with their whole phrase
		○ The system changes the gloss annotations if necessary:
			§ Conjunctions
			§ Preposition
			§ Some word specific changes
			§ Some mistakes made by Spacy, eg lemma of some verbs
			§ Can/must/know/… + not = one sign
			§ Split up numbers in signs
			§ Add plural indicator
				□ This is chosen randomly, no indication is also an option
				□ It first checks if there is an indication already present, eg a number
			§ Add subject / object indication onto the verb, if the verb needs that (e.g., 1-GEVEN-2)
			§ Change pronouns to WG (wijsgebaar = pointing sign)
			§ Reformulate passive sentences
			§ Schijnen = BLIJKBAAR (apparently) / lijken = WG-1 DENKEN (seem = I think) or GELIJKEN-OP (look like)
			§ Rejoin verbs with their separable part 
			§ if the lemma_ / a combination of multiple lemma's is in the gloss_id list, the gloss_id is used
				□ The -A indicating multiple options for one sign (regional variation etc) is not added
			§ Er zijn = HEBBEN
			§ Multiple reasons a word is not translated at all
				□ No determiners
				□ No punctuation marks
				□ ...
		○ Then, the system reorders these glosses
			§ Conjunction words or simple adverbs first
			§ Then temporal clause
			§ Then temporal clause
			§ Then subject if it is no WG (except WG-1)
			§ Then verb
			§ Then indirect object
			§ Then other objects
			§ Then subject if WG
			§ Then rest of sentence
