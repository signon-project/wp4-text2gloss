# Text-to-gloss for VGT

## Installation

Simply pip install this repository. This will automatically download dependencies and also register shortcut commands.

```shell
python -m pip install .
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

#### 0. (Optional -- paid) Add OpenAI GPT-x translations

An optional first step is to add English translations for each gloss. This is done by translating the "possible Dutch
translations" column. Using the OpenAI API allows us to be descriptive in our prompt. Rather than just translating the
individual Dutch words, we can prompt the model by indicating that the given list of Dutch words are synonyms and that
English translations of this whole "synset" should be given, rather than individual, lexical translations.

Translations will be written to the "en" column.

Note that to use this script, you need to have your [OpenAI API key](https://platform.openai.com/account/api-keys) as
an enviroment variable `OPENAI_API_KEY`. Also note that using the OpenAI API is not free!

```shell
translate-openai data/vgt-woordenboek-27_03_2023.tsv data/vgt-woordenboek-27_03_2023+openai.tsv
```

#### 1. Add multilingual WordNet synset "translations" and disambiguate

This script will disambiguate all the translation candidates in the "en" column, too. That includes the OpenAI 
translations as well as the WordNet translations. This is done by means of vector similarities through fastText.
The fastText models will be downloaded automatically to `models/fasttext`.

**Before running this script** make sure that the inference server is running (see 
[FastAPI inference server](#fastapi-inference-server))

TODO

## FastAPI inference server

Because loading the fastText vectors takes a LONG time, I included an `inference_server.py` that can run in the background.
It runs a FastAPI endpoint that can be queried for fastText vectors but also for text-to-AMR.

Start the server by doing into the deepest directory in `src/text2gloss/api` and running:

```shell
uvicorn inference_server:app --port 5000
```

This server needs to be started before running the `pipeline.py` and `vgt_preprocessing.py` scripts.

## Data

I recommend to keep a directory `data` in the highest position (next to `src` and `models`). I recommend to keep the
VGT dictionary (as TSV/JSON) in here for easy access.

## LICENSE

Distributed under a GPLv3 [license](LICENSE).
