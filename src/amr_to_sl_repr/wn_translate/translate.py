import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
import re
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import penman
import torch
from bert_score import model2layers, get_model, score, get_bert_embedding, get_tokenizer
from sentence_transformers import util, SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizer


def get_en2vgtgloss_dict(fin: str) -> Dict[str, List[str]]:
    df = pd.read_csv(fin, sep="\t", encoding="utf-8")
    transl2gloss = defaultdict(list)
    # iterate over col1 and col2 simultaneously
    for index, row in df.iterrows():
        gloss = row["GLOSS"]
        transls = row["translations"]

        if pd.isna(transls):
            continue

        for transl in transls.split(", "):
            transl2gloss[transl].append(gloss)

    return transl2gloss


def get_amr_concepts(graph: penman.Graph) -> List[str]:
    concepts = []
    # TODO: also return numbers, names, negation
    for _, rel, target in graph.triples:
        if rel == ":instance":
            target = re.sub(r"-\d+$", "", target)
            concepts.append(target)

    return concepts


def concept2gloss(concept: str, dictionary: Dict[str, List[str]]):
    try:
        return dictionary[concept]
    except (KeyError, IndexError):
        return None


def sbert_similarity_selection(src_embedding: torch.Tensor, glosses: List[str], embedder):
    glosses = [cleanup_gloss(g) for g in glosses]

    embeddings = embedder.encode(glosses)
    scores = [util.cos_sim(src_embedding, embed).item() for embed in embeddings]
    best_score = np.argmax(scores)
    most_sim_glos = glosses[best_score].upper().replace(" ", "-")

    return most_sim_glos


def init_bertscore(model_type: str = "microsoft/deberta-xlarge-mnli", device: Optional[str] = None):
    num_layers = model2layers[model_type]
    embedder = get_model(model_type=model_type, num_layers=num_layers)
    tokenizer = get_tokenizer(model_type, use_fast=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        embedder.to(device)
    except RuntimeError:
        # Memory error if model does not fit on GPU
        device = "cpu"

    return embedder, tokenizer, device

def get_bertscore_sim(src_sent: str, tgt_words: List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: Optional[str] = None, batch_size: int = 16):
    """Modified from bert_score: https://github.com/Tiiiger/bert_score/blob/master/bert_score/score.py#L21"""
    batch_hyps = tgt_words
    batch_refs = [src_sent] * len(batch_hyps)

    sentences = batch_refs + batch_hyps

    idf_dict = defaultdict(lambda: 1.0)
    # set idf for [SEP] and [CLS] to 0
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    stats_dict = dict()

    iter_range = range(0, len(sentences), batch_size)
    for batch_start in iter_range:
        sen_batch = sentences[batch_start : batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            sen_batch, model, tokenizer, idf_dict, device=device, all_layers=False
        )
        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()
        for i, sen in enumerate(sen_batch):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            stats_dict[sen] = (emb, idf)

    def pad_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        emb = [e.to(device) for e in emb]
        idf = [i.to(device) for i in idf]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask, idf_pad

    ref_embedding, ref_masks, ref_idf = pad_batch_stats(batch_refs, stats_dict, device)
    hyp_embedding, hyp_masks, hyp_idf = pad_batch_stats(batch_hyps, stats_dict, device)

    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))

    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)

    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
    ref_zero_mask = ref_masks.sum(dim=1).eq(2)

    if torch.any(hyp_zero_mask):
        print(
            "Warning: Empty candidate sentence detected; setting raw BERTscores to 0.",
            file=sys.stderr,
        )
        P = P.masked_fill(hyp_zero_mask, 0.0)
        R = R.masked_fill(hyp_zero_mask, 0.0)

    if torch.any(ref_zero_mask):
        print(
            "Warning: Empty reference sentence detected; setting raw BERTScores to 0.",
            file=sys.stderr,
        )
        P = P.masked_fill(ref_zero_mask, 0.0)
        R = R.masked_fill(ref_zero_mask, 0.0)

    F = F.masked_fill(torch.isnan(F), 0.0)

    return P.cpu(), R.cpu(), F.cpu()


def cleanup_gloss(gloss: str):
    gloss = re.sub(r"\([^)]+\)", "", gloss)
    gloss = re.sub(r"-[A-Z]$", "", gloss)
    gloss = gloss.replace("-", " ")

    return gloss.lower()


def main(dict_path: str, amr_path: str, use_sbert: bool = False, best_metric: str = "recall", device: Optional[str] = None):
    dictionary = get_en2vgtgloss_dict(dict_path)
    tokenizer = None
    if use_sbert:
        embedder = SentenceTransformer("distiluse-base-multilingual-cased-v1", device=device)
    else:
        embedder, tokenizer, device = init_bertscore(device=device)

    with open(amr_path, encoding="utf-8") as fhin:
        for graph in penman.iterdecode(fhin):
            src_sent = graph.metadata["snt"]
            print(graph.metadata["snt"])
            print("=" * len(graph.metadata["snt"]))

            concepts = get_amr_concepts(graph)
            print(f"AMR CONCEPTS: {concepts}")
            gloss_cands = {concept: glosses for concept in concepts if (glosses := concept2gloss(concept, dictionary))}
            gloss_cands_prep = {concept: list(set([cleanup_gloss(g) for g in glosses])) for concept, glosses in gloss_cands.items()}

            selected_glosses = []
            if use_sbert:
                src_embedding = embedder.encode(src_sent)

                for concept in concepts:
                    try:
                        gloss_candidates = gloss_cands_prep[concept]
                    except KeyError:
                        continue

                    reglossed = [g.upper().replace(" ", "-") for g in gloss_candidates]
                    print(f"\t- OPTIONS FOR {concept}: {reglossed}")

                    most_sim_cand = sbert_similarity_selection(src_embedding, gloss_candidates, embedder=embedder)
                    print(f"\t- BEST OPTION FOR {concept}: {most_sim_cand}\n")
                    selected_glosses.append(most_sim_cand)
            else:
                for concept in concepts:
                    try:
                        gloss_candidates = gloss_cands_prep[concept]
                    except KeyError:
                        continue

                    reglossed = [g.upper().replace(" ", "-") for g in gloss_candidates]
                    print(f"\t- OPTIONS FOR {concept}: {reglossed}")

                    precision, recall, fscore = get_bertscore_sim(src_sent, gloss_candidates, embedder, tokenizer,
                                                                  device=device)
                    print("PRECISION", precision)
                    best_score = np.argmax(precision)
                    most_sim_cand = reglossed[best_score]
                    print(f"\t- BEST PRECISION OPTION FOR {concept}: {most_sim_cand}\n")

                    print("RECALL", recall)
                    best_score = np.argmax(recall)
                    most_sim_cand = reglossed[best_score]
                    print(f"\t- BEST RECALL OPTION FOR {concept}: {most_sim_cand}\n")

                    print("FSCORE", fscore)
                    best_score = np.argmax(fscore)
                    most_sim_cand = reglossed[best_score]
                    print(f"\t- BEST FSCORE OPTION FOR {concept}: {most_sim_cand}\n")
                    exit()

                    selected_glosses.append(most_sim_cand)

            print(">>> FINAL GLOSSES:", " ".join(selected_glosses))
            print()


if __name__ == '__main__':
    import argparse

    cparser = argparse.ArgumentParser()
    cparser.add_argument("dict_path", help="Path to VGT dictionary (tsv)")
    cparser.add_argument("amr_path", help="Path to sentences in AMR format with ::snt attribute")

    main(**vars(cparser.parse_args()))
    # TODO: recheck syns + transl
    # Probably stick with sbert?
    