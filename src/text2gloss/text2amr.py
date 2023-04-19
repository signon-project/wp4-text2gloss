from functools import lru_cache
from typing import List, Literal, Tuple

import torch
from mbart_amr.constraints.constraints import AMRLogitsProcessor
from mbart_amr.data.linearization import linearized2penmanstr
from mbart_amr.data.tokenization import AMRMBartTokenizer
from optimum.bettertransformer import BetterTransformer
from torch import nn, qint8
from torch.ao.quantization import quantize_dynamic
from transformers import LogitsProcessorList, MBartForConditionalGeneration


LANGUAGES = {
    "English": "en_XX",
    "Dutch": "nl_XX",
    "Spanish": "es_XX",
}


@lru_cache
def get_resources(
    multilingual: bool, quantize: bool = True, no_cuda: bool = False
) -> Tuple[MBartForConditionalGeneration, AMRMBartTokenizer, AMRLogitsProcessor]:
    """Get the relevant model, tokenizer and logits_processor. The loaded model depends on whether the multilingual
    model is requested, or not. If not, an English-only model is loaded. The model can be optionally quantized
    for better performance.
    :param multilingual: whether or not to load the multilingual model. If not, loads the English-only model
    :param quantize: whether to quantize the model with PyTorch's 'quantize_dynamic'
    :param no_cuda: whether to disable CUDA, even if it is available
    :return: the loaded model, tokenizer, and logits processor
    """
    if multilingual:
        # Tokenizer src_lang is reset during translation to the right language
        tokenizer = AMRMBartTokenizer.from_pretrained("BramVanroy/mbart-en-es-nl-to-amr", src_lang="nl_XX")
        model = MBartForConditionalGeneration.from_pretrained("BramVanroy/mbart-en-es-nl-to-amr")
    else:
        tokenizer = AMRMBartTokenizer.from_pretrained("BramVanroy/mbart-en-to-amr", src_lang="en_XX")
        model = MBartForConditionalGeneration.from_pretrained("BramVanroy/mbart-en-to-amr")

    model = BetterTransformer.transform(model, keep_original_model=False)
    model.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available() and not no_cuda:
        model = model.to("cuda")
    elif quantize:  # Quantization not supported on CUDA
        model = quantize_dynamic(model, {nn.Linear, nn.Dropout, nn.LayerNorm}, dtype=qint8)

    logits_processor = AMRLogitsProcessor(tokenizer, model.config.max_length)

    return model, tokenizer, logits_processor


def translate(
    text: List[str],
    src_lang: Literal["English", "Dutch", "Spanish"],
    model: MBartForConditionalGeneration,
    tokenizer: AMRMBartTokenizer,
    **gen_kwargs,
) -> List[str]:
    """Translates a given text of a given source language with a given model and tokenizer. The generation is guided by
    potential keyword-arguments, which can include arguments such as max length, logits processors, etc.
    :param text: source text to translate
    :param src_lang: source language
    :param model: MBART model
    :param tokenizer: MBART tokenizer
    :param gen_kwargs: potential keyword arguments for the generation process
    :return: the translation (linearized AMR graph)
    """
    tokenizer.src_lang = LANGUAGES[src_lang]
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    generated = model.generate(**encoded, **gen_kwargs).cpu()
    return tokenizer.decode_and_fix(generated)


def text2penman(
    text: List[str], src_lang: Literal["English", "Dutch", "Spanish"], quantize: bool = True, no_cuda: bool = False
) -> List[str]:
    """Converts a given list of sentences into a list of penman representations. Note that these are not
    validated so it is possible that the returned penman strings are not valid!
    """
    multilingual = src_lang != "English"
    model, tokenizer, logitsprocessor = get_resources(multilingual, quantize=True, no_cuda=False)
    gen_kwargs = {
        "max_length": model.config.max_length,
        "num_beams": model.config.num_beams,
        "logits_processor": LogitsProcessorList([logitsprocessor]),
    }

    linearizeds = translate(text, src_lang, model, tokenizer, **gen_kwargs)
    penman_strs = [linearized2penmanstr(lin) for lin in linearizeds]

    return penman_strs