import copy
import logging
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import penman
import spacy
import torch
from databases import Database
from fastapi import FastAPI, HTTPException, Query
from mbart_amr.data.linearization import linearized2penmanstr
from pydantic import BaseSettings
from sentence_transformers import SentenceTransformer, util
from spacy.tokens import Doc
from text2gloss.rule_based.complex_sents_helpers import (
    is_true_comparison_split,
    is_true_split,
    plural_with_apostrophe,
    split_relative_clause,
    split_te_verb,
)
from text2gloss.rule_based.helpers import Word, make_sentence_list
from text2gloss.rule_based.reorder_glosses import reorder_glosses
from text2gloss.rule_based.reorder_glosses_helpers import head_of_sub_clause_before
from text2gloss.rule_based.transform_comp import transform_comparative
from text2gloss.rule_based.transform_conj import transform_while
from text2gloss.rule_based.transform_neg import no_negation_in_question
from text2gloss.rule_based.word_lists import (
    conjunctions_if,
    conjunctions_then_after,
    conjunctions_then_before,
    conjunctions_while,
    question_words,
)
from text2gloss.text2amr import get_resources, translate
from text2gloss.utils import standardize_gloss
from transformers import LogitsProcessorList
from typing_extensions import Annotated


logging.getLogger("penman").setLevel(logging.WARNING)


class Settings(BaseSettings):
    no_db: bool = False
    db_path: str = "glosses.db"

    no_sbert: bool = False
    sbert_model_name: str = "sentence-transformers/LaBSE"
    sbert_device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

    no_amr: bool = False
    mbart_input_lang: Literal["English", "Dutch"] = "English"
    mbart_device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    mbart_quantize: bool = True
    mbart_num_beams: int = 3

    no_spacy_nl: bool = False

    logging_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] = "INFO"

settings = Settings()
resources = {}


# see https://fastapi.tiangolo.com/advanced/events/
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=settings.logging_level,
    )
    if not settings.no_sbert:
        resources["stransformer"] = SentenceTransformer(settings.sbert_model_name, device=settings.sbert_device)
        logging.info(f"Using {resources['stransformer']._target_device} for Sentence Transformers")
    else:
        logging.info("Sentence Transformers disabled.")

    if not settings.no_amr:
        amr_model, amr_tokenizer, amr_logitsprocessor = get_resources(
            multilingual=settings.mbart_input_lang != "English",
            quantize=settings.mbart_quantize,
            no_cuda=settings.mbart_device == "cpu",
        )
        amr_gen_kwargs = {
            "max_length": amr_model.config.max_length,
            "num_beams": settings.mbart_num_beams if settings.mbart_num_beams else amr_model.config.num_beams,
            "logits_processor": LogitsProcessorList([amr_logitsprocessor]),
        }
        logging.info(f"Using {amr_model.device} for AMR")
        resources["amr_model"] = amr_model
        resources["amr_tokenizer"] = amr_tokenizer
        resources["amr_gen_kwargs"] = amr_gen_kwargs
    else:
        logging.info("AMR disabled.")

    if not settings.no_db:
        db_path = str(Path(settings.db_path).resolve().expanduser())
        logging.info(f"Using database file at {db_path}")
        resources["database"] = Database(f"sqlite:///{db_path}")
        await resources["database"].connect()
    else:
        logging.info("Database disabled.")

    if not settings.no_spacy_nl:
        resources["spacy_nl"] = spacy.load("nl_core_news_lg")
    else:
        logging.info("spaCy Dutch disabled.")

    yield

    if "database" in resources:
        await resources["database"].disconnect()

    # Clean up the ML models and release the resources
    resources.clear()


app = FastAPI(lifespan=lifespan)


def resource_exists(resource: str) -> bool:
    return resource in resources and resources[resource]


#########################
# SENTENCE TRANSFORMERS #
#########################
def encode_texts(tokens: Tuple[str, ...]):
    if not resource_exists("stransformer"):
        raise HTTPException(status_code=404, detail="The Sentence Transformer model was not loaded.")
    return resources["stransformer"].encode(tokens, device=settings.sbert_device, show_progress_bar=False)


@app.get("/centroid/")
def build_tokens_centroid(
    tokens: Annotated[
        List[str],
        Query(
            title="Tokens",
            description="List of tokens whose vectors to retrieve and average (centroid)",
        ),
    ]
):
    vectors = get_tokens_vectors(tokens)
    return np.mean(vectors, axis=0).tolist()


@app.get("/vectors/")
def get_tokens_vectors(
    tokens: Annotated[
        List[str],
        Query(
            title="Tokens",
            description="List of tokens whose vectors to retrieve",
        ),
    ]
):
    return encode_texts(tuple(tokens)).tolist()


@app.get("/similarity/")
def get_tokens_similarity(
    left_token: Annotated[
        str,
        Query(
            title="First token",
        ),
    ],
    right_token: Annotated[
        str,
        Query(
            title="Second token",
        ),
    ],
):
    vecs = get_tokens_vectors([left_token, right_token])
    return util.cos_sim(vecs[0], vecs[1]).squeeze(dim=0).item()


@app.get("/closest/")
def find_closest(
    text: Annotated[
        str,
        Query(
            title="Anchor token or sentence",
        ),
    ],
    candidates: Annotated[
        List[str],
        Query(
            title="Token candidates",
        ),
    ],
) -> str:
    """If one English concept is linked to multiple VGT glosses, we select the "right" gloss by selecting the one
    whose semantic similarity is closest to the English concept _or_ closer to a given sentence, like the whole input
     sentences. To do so, the gloss is preprocessed though (removing regional variety identifiers -A, -B and removing
      extra information in brackets, lowercased) so that, e.g.,

      'ABU-DHABI(UAR)-A' becomes 'abu-dhabi'. If the gloss does not exist in FastText it is not considered as an option
    (unless it is the only option).

    :param text: concept extracted from AMR or sentence
    :param candidates: list of possible gloss options
    :return: the 'best' gloss
    """
    if len(candidates) == 1:
        return candidates[0]

    candidates_std = [standardize_gloss(gloss).lower() for gloss in candidates]
    vecs = get_tokens_vectors([text] + candidates_std)
    sims = util.cos_sim(vecs[0], vecs[1:]).squeeze(dim=0)
    logging.debug(f"TEXT: {text}")
    logging.debug(f"CANDIDATES: {candidates}")
    logging.debug(f"CANDIDATES (STD): {candidates_std}")
    logging.debug(f"SIMILARITIES: {sims}")
    best_cand_idx = sims.argmax(axis=0).item()

    return candidates[best_cand_idx]


##################################
# GLOSS EXTRACTION FROM DATABASE #
##################################
async def get_gloss_candidates(en_token: str, sign_lang: Literal["vgt", "ngt"]) -> List[str]:
    if not resource_exists("database"):
        raise HTTPException(status_code=404, detail="The SQLite Database was not loaded.")

    query = f"SELECT gloss FROM {sign_lang}_en2gloss_tbl WHERE en='{en_token}'"
    results = await resources["database"].fetch_all(query=query)
    # Flatten. Output above is list of singleton-tuples.
    results = [r for res in results for r in res]
    logging.debug(f"DB RESULTS FOR {en_token} in {sign_lang}: {results}")

    return results


#########################################
# AMR GENERATION AND CONCEPT EXTRACTION #
#########################################
def extract_concepts_from_invalid_penman(penman_str):
    # TODO: probably a regex/string-based extraction?
    return []


def extract_concepts(penman_str: str) -> List[str]:
    """Extract concepts from a given penman string
    :param penman_str: AMR in penman format
    :return: a list of concepts (str)
    """
    try:
        graph = penman.decode(penman_str)
    except Exception:
        return extract_concepts_from_invalid_penman(penman_str)
    else:
        logging.debug(f"AMR GRAPH: {graph}")
        tokens = []
        for source, role, target in graph.triples:
            if source is None or target is None:
                continue

            if role == ":location":
                tokens.append("in")
            elif role == ":instance":
                if target == "amr-unknown":
                    # Questions
                    continue
                target = re.sub(r"(\w)-\d+$", "\\1", target)  # Remove frame/concept sense ID
                tokens.append(target)
            elif role == ":polarity" and target == "-":
                tokens.append("NIET")
            elif role == ":quant":
                # :quant can sometimes occur as precursor to other quant, e.g.:
                #   ('c', ':quant', 'v'): [Push(v)],
                #     ('v', ':instance', 'volume-quantity'): [],
                #     ('v', ':quant', '2'): [],
                # So we want to ignore the first quant
                if not (len(target) == 1 and target.isalpha()):
                    tokens.append(target)

        logging.debug(f"Extracted AMR concepts: {tokens}")
        return tokens


async def concepts2glosses(tokens: List[str], src_sentence: str, sign_lang: Literal["vgt", "ngt"]) -> List[str]:
    """Convert a list of tokens/concepts (extracted from AMR) to glosses by using
    mappings that were collected from the VGT dictionary.

    :param tokens: list of AMR tokens/concepts
    :param src_sentence: full input sentence
    :param sign_lang: which sign language to generate glosses for
    :return: a list of glosses
    """
    glosses = []
    skip_extra = 0

    for token_idx in range(len(tokens)):
        if skip_extra:
            skip_extra -= 1
            continue

        token = tokens[token_idx]
        # We can ignore the special "quantity" identifier
        if token.endswith(("-quantity",)):
            continue
        elif token == "cause":
            glosses.append("[PU]")
        elif token in "i":
            glosses.append("WG-1")
        elif token == "you":
            glosses.append("WG-2")
        elif token in ("he", "she", "they"):
            glosses.append("WG-3")
        elif token == "we":
            glosses.append("WG-4")
        elif token.isdigit():  # Copy digits
            glosses.append(token)
        elif token.startswith('"') and token.endswith('"'):  # Copy literal items but remove quotes
            glosses.append(token[1:-1])
        else:  # Conditions that require info about the next token
            next_token = tokens[token_idx + 1] if token_idx < len(tokens) - 1 else None
            # If this token is "city" and the next token is the city name, we can ignore "city"
            if token in ("city", "station") and next_token and next_token[0].isupper():
                continue
            elif token == "person" and next_token and next_token == "have-rel-role":
                skip_extra = 1
                continue
            else:
                candidates = await get_gloss_candidates(token, sign_lang=sign_lang)
                if not candidates:  # English amr token not found in database (skip token)
                    continue
                else:
                    best_match = find_closest(text=src_sentence, candidates=candidates)
                    glosses.append(best_match)
                    logging.info(f"Best gloss for {token} (out of {candidates}): {best_match}")

    logging.debug(f"Glosses: {glosses}")

    return glosses


#################################
# AMR-BASED TEXT2GLOSS PIPELINE #
#################################
@app.get("/text2gloss/")
async def run_pipeline(
    text: Annotated[
        str,
        Query(
            title="Text to convert to a penman representation",
        ),
    ],
    sign_lang: Annotated[Literal["vgt", "ngt"], Query(title="Which sign language to generate glosses for")] = "vgt",
) -> Dict[str, Any]:
    if (
        not resource_exists("amr_model")
        or not resource_exists("amr_tokenizer")
        or not resource_exists("amr_gen_kwargs")
    ):
        raise HTTPException(status_code=404, detail="The AMR model was not loaded.")

    linearizeds = translate(
        [text],
        settings.mbart_input_lang,
        resources["amr_model"],
        resources["amr_tokenizer"],
        **resources["amr_gen_kwargs"],
    )
    penman_str = linearized2penmanstr(linearizeds[0])
    amr_concepts = extract_concepts(penman_str)
    glosses = await concepts2glosses(amr_concepts, src_sentence=text, sign_lang=sign_lang)

    return {"glosses": glosses, "meta": {"amr_concepts": amr_concepts}}


###############
# SPACY DUTCH #
###############
@app.get("/parse_spacy_nl/")
def get_spacy_nl_doc(
    text: Annotated[
        str,
        Query(
            title="Text to convert to a penman representation",
        ),
    ],
):
    if not resource_exists("spacy_nl"):
        raise HTTPException(status_code=404, detail="The spaCy NL model was not loaded.")

    return resources["spacy_nl"](text).to_json()


###########################################
# RULE-BASED COMPONENT TO SPLIT SENTENCES #
###########################################
@app.get("/split_sentence/")
def split_based_on_conjunction_words(
    text: Annotated[
        str,
        Query(
            title="Sentence to split",
        ),
    ]
):
    """main function that splits complex sentences into separate clauses
    that each can be transformed into glosses separately"""

    # MAKE SPACY DOC
    sentence_json = get_spacy_nl_doc(text)
    sentence_doc = Doc(resources["spacy_nl"].vocab).from_json(sentence_json)
    sentence_doc_list = tuple(token for token in sentence_doc if not token.is_space)
    sentence = make_sentence_list(sentence_doc_list)
    sentence_new = copy.deepcopy(sentence)

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
            if "betr" in item.tag_ or (
                ("waar" in item.lemma_ or "wie" in item.lemma_) and "acl:relcl" in [item7.dep_ for item7 in sentence]
            ):
                # eg: 'de man die ...' / 'de man voor wie ...' / 'de kast waarin ...'
                sentence, sentence_new, take_outs_rel = split_relative_clause(
                    sentence, index, item, sentence_new, take_outs_rel
                )

            # SPLIT QUESTION WORD IN THE MIDDLE OF THE SENTENCE
            # eg: 'weet jij waarom ze dat doen?'
            if (
                item.lemma_ in question_words
                and index > 1
                and not (sentence[index - 1].lemma_ == "heel" and item.lemma_ == "wat")
            ):
                # if it is one of the first 2 words, it is just a question
                splits.append([start_of_next_clause, index])
                start_of_next_clause = index

            # SPLIT BASED ON CONJUNCTION WORDS
            if item.pos_ in ("CCONJ", "SCONJ") and index - 1 not in [item2 for sublist in splits for item2 in sublist]:
                if item.lemma_ == "dan":
                    if is_true_comparison_split(sentence, index, item):
                        splits.append([start_of_next_clause, index])
                        start_of_next_clause = index

                elif is_true_split(sentence, index, item):
                    splits.append([start_of_next_clause, index])
                    start_of_next_clause = index

            # SPLIT BASED ON PUNCTUATION
            if item.lemma_ in (",", ".", ";", ":", "!", "?") and index not in tuple(
                item2 for sublist in splits for item2 in sublist
            ):
                # not all PUNCT
                if is_true_split(sentence, index, item) and index - 1 not in tuple(
                    item3 for sublist in take_outs_rel for item3 in sublist
                ):
                    splits.append([start_of_next_clause, index + 1])
                    start_of_next_clause = index + 1

            if "(" in item.lemma_:
                take_out_brackets = []
                for n in range(index, len(sentence)):
                    if ")" not in sentence[n].lemma_:
                        take_out_brackets.append(n)
                    else:
                        take_out_brackets.append(n)
                        break
                take_outs_brackets.append(take_out_brackets)

            # SPLIT BASED ON SPECIFIC CONSTRUCTION
            # 'om te' + 'te + infinitive'
            if (
                item.lemma_ == "om"
                and item.dep_ == "mark"
                and index - 1 not in tuple(item for sublist in take_outs_rel for item in sublist)
            ):
                splits.append([start_of_next_clause, index])
                start_of_next_clause = index + 1
            if item.lemma_ == "te" and not take_outs_rel:
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
        if (
            sentence_new[number_couple[0]].text == "dan"
            and sentence_new[number_couple[0]].pos_ == "SCONJ"
            and sentence_new[number_couple[0]].dep_ in ("mark", "fixed")
        ):
            splits_new, sentence_new = transform_comparative(sentence_new, index2, splits)

        # CHANGE WHILE CLAUSE
        if sentence_new[number_couple[0]].text in conjunctions_while:
            sentence_new, splits_new = transform_while(sentence_new, index2, number_couple, splits_new)

        # IF TEMPORAL CLAUSE: CHECK ORDER
        # chronological order is preferred in VGT
        if sentence_new[number_couple[0]].lemma_ in conjunctions_then_after:
            addition = Word(
                "dan", "DAN", "dan", splits[index2 - 1][0], "advmod", "dan", "BW", "ADV", False, False
            )  # head not correct # AF DAN
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
        if sentence_new[number_couple[0]].lemma_ in conjunctions_if and 0 < index2 < len(splits_new):
            splits_new.insert(index2 - 1, splits_new.pop(index2))

        if (
            sentence_new[number_couple[0]].text == "dan"
            and sentence_new[number_couple[0]].pos_ == "SCONJ"
            and sentence_new[number_couple[0]].dep_ in ["mark", "fixed"]
        ):
            splits_new, sentence_new = transform_comparative(sentence_new, index2, splits)

    # APPLY SPLITS
    for index3, number_couple in enumerate(splits_new):
        clause = [
            word
            for word in sentence_new[number_couple[0] : number_couple[1]]
            if word.position not in [item for sublist in take_outs_rel for item in sublist]
            and word.position not in [item for sublist in take_outs_brackets for item in sublist]
        ]
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


##################################
# RULE-BASED TEXT2GLOSS PIPELINE #
##################################
@app.get("/rb_text2gloss/")
def run_rb_pipeline(
    text: Annotated[
        str,
        Query(
            title="Complex sentence to turn into glosses",
        ),
    ]
):
    """split into clausses and transform each clause with 'reorder_glosses'
    then concatenate all clauses with '//'"""
    if isinstance(text, str):
        text = plural_with_apostrophe(text)
        clauses = split_based_on_conjunction_words(text)
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
        # complex_gloss_sentence_list = complex_gloss_sentence_list[:-1] # delete the last '//'from
        glosses = complex_gloss_sentence_list
    else:
        glosses = [str(text)]

    return {"glosses": glosses}
