from dataclasses import dataclass, field
from functools import cached_property
import re
from typing import Dict, List

import penman

from amr_to_sl_repr.dictionary import Dictionary


@dataclass  # Might be better suited in a functional approach, not as class
class Linearizer:
    dictionary: Dictionary
    tree: penman.Tree = field(default=None)
    graph: penman.Graph = field(default=None)
    varmaps: Dict[str, str] = field(default_factory=dict, init=False)
    metadata: Dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if not self.tree and not self.graph:
            raise ValueError("At least either 'tree' or 'graph' must be given")
        elif not self.graph:
            self.graph = penman.interpret(self.tree)
        elif not self.tree:
            self.tree = penman.configure(self.graph)

        self.metadata = self.graph.metadata

    @cached_property
    def amr_linearized_tokens(self) -> List[str]:
        tokens: List[str] = []
        for source, role, target in self.graph.triples:
            if role == ":instance":
                target = re.sub(r"(\w)-\d+$", "\\1", target)  # Remove frame/concept sense ID
                tokens.append(target)

        return tokens

    @cached_property
    def gloss_repr(self):
        print("SEQ TRANSLATION", self.dictionary.hf_translator(" ".join(self.amr_linearized_tokens))[0])
        translations = []

        for token_idx in range(len(self.amr_linearized_tokens)):
            token = self.amr_linearized_tokens[token_idx]
            # We can ignore the special "quantity" identifier
            if token.endswith(("-quantity", )):
                continue
            elif token == "cause":
                translations.append("[PU]")
            elif token.isdigit():  # Copy digits
                translations.append(token)
            elif token.startswith('"') and token.endswith('"'):  # Copy literal items but remove quotes
                translations.append(token[1:-1])
            else:  # Conditions that require info about the next tion
                next_token = self.amr_linearized_tokens[token_idx+1] if token_idx < len(self.amr_linearized_tokens) - 1 else None
                # If this token is "city" and the next token is the city name, we can ignore "city"
                if token in ("city", "station") and next_token and next_token[0].isupper():
                    continue
                else:
                    translations.append(self.dictionary.word2gloss(token))

        return " ".join(translations)



example_vgt = """# ::snt There are thieves on the train so please mind your belongings
# ::vgt-glossen (TREIN) HEEFT DIEF // [PU] OPPASSEN (BEWAKEN MATERIAAL)
(c / cause-01
   :ARG0 (t / thief
            :location (t2 / train))
   :ARG1 (m / mind-15
            :mode imperative
            :polite +
            :ARG0 (y / you)
            :ARG1 (t3 / thing
                      :ARG1-of (b / belong-01
                                  :ARG0 y))))
"""



if __name__ == '__main__':
    dictionary = Dictionary(r"D:\corpora\sl\Corpus VGT\woordenboek.tsv")
    ln = Linearizer(dictionary, tree=penman.parse(example_vgt))
    print(ln.gloss_repr)


