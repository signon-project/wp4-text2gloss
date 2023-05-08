# Text-to-gloss through semantic abstraction

## Installation

Simply pip install this repository. This will automatically download dependencies and also register shortcut commands.

```shell
python -m pip install .
```

For developers: you can automatically install some goodies like black, isort, flake8 with the `dev` extra.

```shell
python -m pip install .[dev]
```


## How to use

### 1. Reformat dictionary

Because dictionaries of different languages have a different format, we streamline their structure for the following
steps. We are only interested in the gloss, the meaning in the related language (e.g. 'nl' for NGT and VGT), and
optionally the video URLs.

Run the following script. Specify the dictionary input file, and use the `-l` flag to indicate which sign language the
dictionary describes. Note that the input format can also be a TSV file (e.g. for VGT).

```shell
reformat-dictionary .\data\ngt-dictionary.csv -l ngt
```

#### 2. (Optional -- paid) Add OpenAI GPT-x translations

An optional first step is to add English translations for each gloss. Using the OpenAI API allows us to be descriptive
in our prompt. Rather than just translating the individual words of the explanation column (e.g. 'nl'), we can prompt
the model by indicating that the given list of words are synonyms and that English translations of this whole "synset"
should be given, rather than individual, lexical translations.

Translations will be written to the "en" column.

Note that to use this script, you need to have your [OpenAI API key](https://platform.openai.com/account/api-keys) as
an enviroment variable `OPENAI_API_KEY` or pass it as `--api_key`. Also note that using the OpenAI API is not free!

Required inputs are the initial dictionary TSV (after running the reformat step) and a description of the language
that is used in the descriptions, which will be used in the prompt.

```shell
translate-openai data/ngt-dictionary-reformat.tsv 'Dutch, the variant of Dutch spoken in the Netherlands'
```

#### 3. Add multilingual WordNet synset "translations" and disambiguate

**Before running this script** make sure that the inference server is running (see 
[FastAPI inference server](#fastapi-inference-server))

In addition to the optional previous step, we can also find broad translations through open multilingual WordNet. By
looking up a Dutch word's synset, we can find the English words that correspond with the English synset that is aligned
with the Dutch one. This gives us a set of potential, very broad, translations.

Therefore, we use disambiguation to filter out English translations that are too "far" from the Dutch translations in
terms of semantics. This script will disambiguate all the translation candidates in the "en" column. That includes the
potential OpenAI translations as well as the WordNet translations. This is done by means of vector similarities through
[LABSE](https://huggingface.co/sentence-transformers/LaBSE).

Required inputs is the dictionary in TSV format. Output will be written to a TSV file and a JSON file that
start with the same name/path but that end in `+wn_transls.*`.

```shell
translate-wn data/vgt-woordenboek-27_03_2023+openai.tsv
```

The output of this is step is an updated TSV file (same directory as input), as well as a JSON file that contains the 
following keys:

- `gloss2en`: a dictionary (str->list) of gloss to potential English translations
- `gloss2nl`: a dictionary (str->list) of gloss to potential Dutch translations
- `en2gloss`: a dictionary (str->list) of English translation to glosses (most important)
- `nl2gloss`: a dictionary (str->list) of Dutch translation to glosses


### 4. Full text2gloss pipeline

**Before running this script** make sure that the inference server is running (see 
[FastAPI inference server](#fastapi-inference-server))

The fulle pipeline allows you to input a sentence and get back a sequence of glosses. Under the hood, this will
make use of text2amr neural models, then the English PropBank concepts will be extracted from that AMR,
and finally the processed JSON-version of the dictionary (cf. 
[step 1.1](#3-add-multilingual-wordnet-synset-translations-and-disambiguate)) will be used to find glosses that
correspond with the extracted English concepts. If multiple gloss options are available, we use LABSE to calculate
the similarity. The gloss that is closest to the English query word is then selected as the final gloss.

The required input is the sentence to covert, the source language ('Dutch' or 'English'), and the path to the JSON file
that was generated in the previous step.

```shell
text2gloss "I want to eat my grandma's cookies"
```



## FastAPI inference server

An inference server is included to serve the MBART AMR pipeline as well as LABSE vector creation.

Start the server by going into `src/text2gloss/api` and running:

```shell
uvicorn main:app --port 5000
```

Alternatively, you can use the Dockerfile to set up the server.

```shell
docker build -t text2gloss-vgt-img .
docker run --rm -d --name text2gloss -p 5000:5000 text2gloss-vgt-img
```

Some configuration is possible. Below you find them (but you should add them as uppercase) with their type and default
value. If for instance you want to make sure that the models are NOT using a GPU, you can use
`SBERT_DEVICE="cpu" MBART_DEVICE="cpu"`. To set these environment variables in Docker, use the --env option with
`docker run`.

```python
json_vgt_dictionary: str = r"vgt-woordenboek-27_03_2023+openai+wn_transls.json"

sbert_model_name: str = "sentence-transformers/LaBSE"
sbert_device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

mbart_input_lang: Literal["English", "Dutch"] = "English"
mbart_device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
mbart_quantize: bool = True
mbart_num_beams: int = 3
```

This server needs to be started before running the `text2gloss` and `translate-wn` scripts.

### Downloading VGT videos

Download the corresponding videos from the URLs in the VGT dictionary.

```shell
download-vgt-videos data/vgt-woordenboek-27_03_2023.tsv data/videos error.log -j 8
```

## LICENSE

Distributed under a GPLv3 [license](LICENSE).
