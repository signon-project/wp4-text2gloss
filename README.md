# Text-to-gloss for VGT
 
# FastText models

We use fastText aligned models for disambiguation purposes. The models will be downloaded automatically to `models/fasttext`.

# FastAPI inference server

Because loading the fastText vectors takes a LONG time, I included an `inference_server.py` that can run in the background.
It runs a FastAPI endpoint that can be queried for fastText vectors but also for text-to-AMR.

Start the server by doing into the deepest directory in `src/...` and running:

```shell
uvicorn inference_server:app --port 5000
```

This server needs to be started before running the `pipeline.py` script.


## LICENSE

Distributed under a GPLv3 [license](LICENSE).
