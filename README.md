# Text-to-gloss for VGT

## Installation

Simply pip install this repository. This will automatically download dependencies and also register shortcut commands.

```shell
python -m pip install .
```

For developers: you can automatically install some goodies like black, isort, mypy with the `dev` extra.

```shell
python -m pip install .[dev]
```


## How to use

### 0. Download VGT videos

Download the corresponding videos from the URLs in the VGT dictionary.

```shell
download-vgt-videos data/vgt-woordenboek-27_03_2023.tsv data/videos error.log -j 8
```


### 1. Pre-process the VGT dictionary

The VGT dictionary contains glosses with a lot of information per gloss. For this repository especially the possible
Dutch "translations" and the videos.

#### 1.0. (Optional -- paid) Add OpenAI GPT-x translations

An optional first step is to add English translations for each gloss. This is done by translating the "possible Dutch
translations" column. Using the OpenAI API allows us to be descriptive in our prompt. Rather than just translating the
individual Dutch words, we can prompt the model by indicating that the given list of Dutch words are synonyms and that
English translations of this whole "synset" should be given, rather than individual, lexical translations.

Translations will be written to the "en" column.

Note that to use this script, you need to have your [OpenAI API key](https://platform.openai.com/account/api-keys) as
an enviroment variable `OPENAI_API_KEY`. Also note that using the OpenAI API is not free!

Required inputs are the initial dictionary TSV, and the output path to write the resulting data to.

```shell
translate-openai data/vgt-woordenboek-27_03_2023.tsv data/vgt-woordenboek-27_03_2023+openai.tsv
```

#### 1.1. Add multilingual WordNet synset "translations" and disambiguate

**Before running this script** make sure that the inference server is running (see 
[FastAPI inference server](#fastapi-inference-server))

In addition to the optional previous step, we can also find broad translations through open multilingual WordNet. By
looking up a Dutch word's synset, we can find the English words that correspond with the English synset that is aligned
with the Dutch one. This gives us a set of potential, very broad, translations.

Therefore, we use disambiguation to filter out English translations that are too "far" from the Dutch translations in
terms of semantics. This script will disambiguate all the translation candidates in the "en" column. That includes the
potential OpenAI translations as well as the WordNet translations. This is done by means of vector similarities through
fastText. The fastText models will be downloaded automatically to `models/fasttext`.

Required inputs is the dictionary in TSV format. Output will be written to a TSV file and a JSON file that
start with the same name/path but that end in `+wn_transls.*`.

```shell
translate-wn data/vgt-woordenboek-27_03_2023+openai.tsv
```

The output of this is step is an updated TSV file (same directory as input), as well as a JSON file that contains the 
following keys:

- `gloss2en`: a dictionary (str->list) of gloss to potential English translations
- `gloss2nl`: a dictionary (str->list) of gloss to potential Dutch translations
- `en2gloss`: a dictionary (str->list) of English translation to glosses
- `nl2gloss`: a dictionary (str->list) of Dutch translation to glosses


### 2. Full text2gloss pipeline

**Before running this script** make sure that the inference server is running (see 
[FastAPI inference server](#fastapi-inference-server))

The fulle pipeline allows you to input a sentence and get back a sequence of glosses. Under the hood, this will
make use of text2amr neural models, then the English PropBank concepts will be extracted from that AMR,
and finally the processed JSON-version of the VGT dictionary (cf. 
[step 1.1](#11-add-multilingual-wordnet-synset-translations-and-disambiguate)) will be used to find glosses that
correspond with the extracted English concepts. If multiple gloss options are available, we use FastText to calculate
the similarity. The gloss that is closest to the English query word is then selected as the final gloss.

The required input is the sentence to covert, the source language ('Dutch' or 'English'), and the path to the JSON file
that was generated in the previous step.

```shell
text2gloss "De postbode blaft naar de hond" Dutch data/vgt-woordenboek-27_03_2023+openai+wn_transls.json
```


## FastAPI inference server

Because loading the fastText vectors takes a LONG time, I included an `inference_server.py` that can run in the background.
It runs a FastAPI endpoint that can be queried for fastText vectors but also for text-to-AMR.

Start the server by going into `src/text2gloss/api` and running:

```shell
uvicorn inference_server:app --port 5000
```

This server needs to be started before running the `pipeline.py` and `vgt_preprocessing.py` scripts.

## Data

I recommend to keep a directory `data` in the highest position (next to `src` and `models`). I recommend to keep the
VGT dictionary (as TSV/JSON) in here for easy access.

## LICENSE

Distributed under a GPLv3 [license](LICENSE).
