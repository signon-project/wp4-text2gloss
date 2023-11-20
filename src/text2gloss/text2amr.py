from typing import List, Literal, Tuple

import penman
import torch
from multi_amr.data.tokenization import AMRTokenizerWrapper
from multi_amr.evaluate_amr import batch_translate
from optimum.bettertransformer import BetterTransformer
from torch import nn, qint8
from torch.ao.quantization import quantize_dynamic
from transformers import MBartForConditionalGeneration, PreTrainedModel


LANGUAGES = {
    "English": "en_XX",
    "Dutch": "nl_XX",
    "Spanish": "es_XX",
}


def get_resources(
    multilingual: bool, quantize: bool = True, no_cuda: bool = False
) -> Tuple[MBartForConditionalGeneration, AMRTokenizerWrapper]:
    """Get the relevant model, tokenizer and logits_processor. The loaded model depends on whether the multilingual
    model is requested, or not. If not, an English-only model is loaded. The model can be optionally quantized
    for better performance.

    :param multilingual: whether to load the multilingual model. If not, loads the English-only model
    :param quantize: whether to quantize the model with PyTorch's 'quantize_dynamic'
    :param no_cuda: whether to disable CUDA, even if it is available
    :return: the loaded model, tokenizer, and logits processor
    """
    if multilingual:
        # Tokenizer src_lang is reset during translation to the right language
        tokenizer = AMRTokenizerWrapper.from_pretrained(
            "BramVanroy/mbart-large-cc25-ft-amr30-en_es_nl", src_lang="nl_XX"
        )
        model = MBartForConditionalGeneration.from_pretrained("BramVanroy/mbart-large-cc25-ft-amr30-en_es_nl")
    else:
        tokenizer = AMRTokenizerWrapper.from_pretrained("BramVanroy/mbart-large-cc25-ft-amr30-en", src_lang="en_XX")
        model = MBartForConditionalGeneration.from_pretrained("BramVanroy/mbart-large-cc25-ft-amr30-en")

    model = BetterTransformer.transform(model, keep_original_model=False)

    if torch.cuda.is_available() and not no_cuda:
        model = model.to("cuda")
    elif quantize:  # Quantization not supported on CUDA
        model = quantize_dynamic(model, {nn.Linear, nn.Dropout, nn.LayerNorm}, dtype=qint8)

    model.eval()
    return model, tokenizer


def translate(
    texts: List[str],
    src_lang: Literal["English", "Dutch", "Spanish"],
    model: PreTrainedModel,
    tok_wrapper: AMRTokenizerWrapper,
    **gen_kwargs,
) -> List[str]:
    """Translates a given text of a given source language with a given model and tokenizer. The generation is guided by
    potential keyword-arguments, which can include arguments such as max length, logits processors, etc.

    :param texts: batch of texts to translate (must be in same language)
    :param src_lang: source language
    :param model: AMR finetuned model
    :param tok_wrapper: tokenizer wrapper
    :param gen_kwargs: potential keyword arguments for the generation process
    :return: a list of penman graphs
    """
    try:
        src_lang = LANGUAGES[src_lang]
    except KeyError:
        raise KeyError("'src_lang' must be one of 'English', 'Spanish', or 'Dutch'")

    output = batch_translate(texts, src_lang=src_lang, model=model, tok_wrapper=tok_wrapper, **gen_kwargs)
    graphs = [penman.encode(g) for g in output["graph"]]
    return graphs
