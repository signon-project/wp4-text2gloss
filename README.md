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
looking up a word's synset, we can find the English words that correspond with the English synset that is aligned
with the initial (e.g. Dutch) one. This gives us a set of potential, very broad, translations.

Therefore, we use disambiguation to filter out English translations that are too "far" from the translations in
terms of semantics. This script will disambiguate all the translation candidates in the "en" column. That includes the
potential OpenAI translations as well as the WordNet translations. This is done by means of vector similarities through
[LABSE](https://huggingface.co/sentence-transformers/LaBSE).

Required inputs is the reformatted dictionary in TSV format, a path to write the output database to, and the column
that specifies the "explanation" of a gloss. Output will be written to an updated TSV file, and, importantly, a sqlite
database. This database will later be used in the pipeline.


```shell
preprocess-gloss data/vgt-woordenboek-27_03_2023-reformat-openai.tsv data/glosses.db nl_vgt --port 5001
```

Repeat this process for all languages that you need. The database will be modified in-place. Specifically,
we add separate tables for the separate languages.


### 4. Full text2gloss pipeline

**Before running this script** make sure that the inference server is running (see 
[FastAPI inference server](#fastapi-inference-server))

The fulle pipeline allows you to input a sentence and get back a sequence of glosses. Under the hood, this will
make use of text2amr neural model, then the English PropBank concepts will be extracted from that AMR,
and finally the processed SQLite version of the dictionary (cf. 
[step 1.1](#3-add-multilingual-wordnet-synset-translations-and-disambiguate)) will be used to find glosses that
correspond with the extracted English concepts. If multiple gloss options are available, we use LABSE to calculate
the similarity. The gloss that is closest to the input sentence is then selected as the final gloss.

The required input is the sentence to covert, the target language (e.g. "vgt" or "ngt").

```shell
text2gloss "I want to eat my grandma's cookies" vgt
```

The output printed to the console will look something like this:

```
VGT {'glosses': ['WENSEN', 'WG-1', 'ETEN', 'KOEK', 'GROOTMOEDER'], 'meta': {'amr_concepts': ['want', 'i', 'eat', 'cookie', 'person', 'have-rel-role', 'grandmother']}}
```


### 5. Full text2gloss pipeline (rule-based)

The text2gloss pipeline was initially created with the idea that, because we start from AMR, we could work with any 
language: either through multilingual AMR generation or first via MT from X-to-EN and then EN to AMR. In reality, we
are still limited by other parts of the pipeline, namely the signbank ("dictionary"). If a language does not have
a dictionary with glosses, that can be linked to videos, then that pipeline has little benefit over a simpler pipeline
that can produce pseudo-glosses.

In earlier iterations of this project, Maud Goddefroy (with help of other partners) worked on rule-based generation
of (pseudo-)glosses for Dutch->VGT. The initial code has been adapted slightly by Bram Vanroy and integrated in this
package as well. It can be used as such:

```shell
rb-text2gloss "Ik wil graag koekjes eten" --port 5001
```

The `--port` argument is only required if it differs from the default (`5000`).

To run the code, the inference server must be running (see below). You can disable many of the other components, as we
only need spaCy for this pipeline. (Of course the command below only works if you have built the image first.)

```shell
docker run --env NO_SBERT=true --env NO_AMR=true --env NO_DB=true --rm -d --name text2gloss -p 5000:5000 text2gloss-img
```

## FastAPI inference server

An inference server is included to serve the MBART AMR pipeline as well as LABSE vector creation.

Start the server by going into `src/text2gloss/api` and running:

```shell
uvicorn main:app --port 5000
```

Alternatively, you can use the Dockerfile to set up the server.

```shell
docker build -t text2gloss-img .
docker run --rm -d --name text2gloss -p 5000:5000 text2gloss-img
```

Some configuration is possible. Below you find them (but you should add them as uppercase) with their type and default
value. If for instance you want to make sure that the models are NOT using a GPU, you can use
`SBERT_DEVICE="cpu" MBART_DEVICE="cpu"`. To set these environment variables in Docker, use the --env option with
`docker run`, for instance:

```shell
docker run --env NO_SBERT=true --env NO_AMR=true --env NO_DB=true --rm -d --name text2gloss -p 5000:5000 text2gloss-img
```

These are all the options that are available and their defaults.

```python
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
```


### Downloading videos

Download the corresponding videos from the URLs in a dictionary. Use this on a reformatted dictionary (a 'video'
column must be present). The required input is the reformatted dictionary, the location to save the videos, and a 
path to an error log file.

```shell
download-videos data/vgt-woordenboek-27_03_2023-reformat.tsv data/videos error.log
```

## LICENSE

Distributed under a GPLv3 [license](LICENSE).
