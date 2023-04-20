# TODO

- Use LABSE instead of fasttext
- Incorporate Euwan/Maud's code (maybe as fallback)? (translation -> rule-based -> (pseudo-gloss)

```python
import torch
from sentence_transformers import SentenceTransformer, util


nl = "aanslag"
nl_words = nl.split(", ")
num_nl_words = len(nl_words)
en = "assault, attack, bombing, tax assessment"
en_words = en.split(", ")

model = SentenceTransformer('sentence-transformers/LaBSE')
embeddings = model.encode(nl_words+en_words)

nl_centroid = torch.tensor(embeddings[:num_nl_words].mean(axis=0))
cos_sims = util.cos_sim(nl_centroid, embeddings[num_nl_words:]).squeeze(dim=0).tolist()

for en_word, cos_sim in zip(en_words, cos_sims):
    print(en_word, cos_sim)
```
