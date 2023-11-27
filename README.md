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


## How to preprocess

### 1. Reformat dictionary

Because dictionaries of different languages have a different format, we streamline their structure for the following
steps. We are only interested in the gloss, the meaning in the related language (e.g. 'nl' for NGT and VGT), and
optionally the video URLs.

Run the following script. Specify the dictionary input file, and use the `-l` flag to indicate which sign language the
dictionary describes. Note that the input format can also be a TSV file (e.g. for VGT).

```shell
reformat-dictionary .\data\ngt-dictionary.csv -l ngt
```

### 2. (Optional -- paid) Add OpenAI GPT-x translations

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

### 3. Add multilingual WordNet synset "translations" and disambiguate

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


## Full text2gloss pipeline

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
text2gloss "I want to eat my grandma's cookies" English vgt
```

The output printed to the console will look something like this, where META includes
whether the utterance was a question (`unknown`), the mode (e.g. `imperative`), and the AMR string in PENMAN representation.

```
TEXT: I want to eat my grandma's cookies
VGT: WENSEN WG-1 ETEN KOEK IEMAND GROOTMOEDER
META {'is_unknown': False, 'mode': None, 'penman_str': '(w / want-01\n   :ARG0 (i / i)\n   :ARG1 (e / eat-01\n            :ARG0 i\n
:ARG1 (c / cookie\n                     :poss (p / person\n                              :ARG0-of (h / have-rel-role-91\n
                    :ARG1 i\n                                          :ARG2 (g / grandma))))))', 'text': "I want to eat my grandma's cookies"}
```

Under the hood this is making use of an API endpoint that is running through the inference server (see below), which is running on `http://127.0.0.1:{port}/text2gloss/`. For the available parameters, see [Swagger](#swagger).


## Full text2gloss pipeline (rule-based)

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

Example output:

```
TEXT: Ik wil graag koekjes eten
VGT: WG-1 WIL GRAAG ETEN KOEK
```

The `--port` argument is only required if it differs from the default (`5000`). Under the hood this is making use of an API endpoint that is running through the inference server (see below), which is running on `http://127.0.0.1:{port}/rb_text2gloss/`.  For the available parameters, see [Swagger](#swagger).

To run the code, the inference server must be running (see below). You can disable many of the other components, as we
only need spaCy for this pipeline. (Of course the command below only works if you have built the image first.) This is
discussed in more detail in the section [below](#fastapi-inference-server).

```shell
docker run --env NO_SBERT=true --env NO_AMR=true --env NO_DB=true --rm -d --name text2gloss -p 5000:5000 text2gloss-img
```

## Generating and evaluating glosses from files

Two utility scripts have been creatied to facilitate batch generation of glosses as well as evaluation of the generated
glosses. They are installed automatically after you install this library, but you will have to install it with its
`evaluate` option:

```sh
python -m pip install .[evaluate]
```

This will install two scripts `generate-gloss` and `evaluate-gloss`. The first one will generate glosses for a given
file. The second one will evaluate the generated glosses against a reference file. Both scripts have a `--help` option
so you can use that to get more information about possible arguments.

Example usage:

```sh
# Generate glosses. This expect the inference server to be running!
generate-gloss sentences.txt ref_glosses.txt output_glosses.txt --src-lang Dutch --sign-lang ngt

# Evaluate generated glosses. 
# - Writes corpus-level scores to ngt-only_supported_glosses-scores.json
# - Writes sentence-level scores to ngt-only_supported_glosses-sent_scores.csv for supported metrics
evaluate-gloss ngt-only_supported_glosses-predictions.txt gold_glosses.txt gold_sents.txt ngt-only_supported_glosses-scores.json ngt-only_supported_glosses-sent_scores.csv --batch-size 64 --no-lower-case
```


## FastAPI inference server

An inference server is included to serve the MBART AMR pipeline as well as LABSE vector creation.

Start the server by running:

```shell
cd src/text2gloss/api
uvicorn main:app --port 5000
```

Alternatively, you can use the Dockerfile to set up the server.

```shell
docker build -t text2gloss-img .
docker run --rm -d --name text2gloss -p 5000:5000 text2gloss-img
```

Some configuration is possible. Below you find them (but you should add them as uppercase) with their type and default
value. If for instance you want to make sure that the models are NOT using a GPU, you can use
`SBERT_DEVICE="cpu" MBART_DEVICE="cpu"`. If you are running the command from the command-line, you can use

Windows (PowerShell):

```shell
$env:NO_SBERT="true"; $env:NO_AMR="true"; uvicorn main:app
```

Bash:

```shell
NO_SBERT="true" NO_AMR="true" uvicorn main:app
```

To set these environment variables in Docker, use the --env option with
`docker run`, for instance:

```shell
docker run --env NO_SBERT=true --env NO_AMR=true --rm -d --name text2gloss -p 5000:5000 text2gloss-img
```

These are all the options that are available and their defaults.

```python
no_db: bool = False
db_path: str = "glosses.db"

no_sbert: bool = False
sbert_model_name: str = "sentence-transformers/LaBSE"
sbert_device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

no_amr: bool = False
mbart_device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
mbart_quantize: bool = True
mbart_num_beams: int = 3

no_spacy_nl: bool = False

logging_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] = "WARNING"
```

If you are using spaCy, enabled by default, make sure to download the spaCy model.

```shell
python -m spacy download nl_core_news_lg
```

### Swagger

To view the available end points and their required parameters, see the swagger documentation which is available after launching the server
on `http://127.0.0.1:{port}/docs`.

## Downloading videos

Download the corresponding videos from the URLs in a dictionary. Use this on a reformatted dictionary (a 'video'
column must be present). The required input is the reformatted dictionary, the location to save the videos, and a 
path to an error log file.

```shell
download-videos data/vgt-woordenboek-27_03_2023-reformat.tsv data/videos error.log
```

## Note on offensive content

Consider that neither the input nor the output of the translation has been checked or cleaned. The whole process is
automatic. It is therefore possible that offensive terms are present in the output. When debugging the pipeline, I found
that offensive terms (such as the n-word in the NGT signbank) exist in the signbank(s) and in WordNet. OpenAI
translation seems to avoid such terms.

While the proposed pipeline only generates glosses that are present in the signbanks, you may find these offensive
terms in intermediate steps in the files in the "meaning" or "translations" columns. Again, these are not manually
added by us nor will they be generated for the end-user, but they are automatically included due to the input data
or due to automatic translation (at least via WordNet).

## LICENSE

Distributed under a GPLv3 [license](LICENSE).
