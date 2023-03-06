from dataclasses import dataclass, field
from typing import Dict

import penman


@dataclass(repr=False)
class Converter:
    amr: penman.Tree
    amr_graph: penman.Graph = field(default_factory=penman.Graph, init=False)
    varmaps: Dict[str, str] = field(default_factory=dict, init=False)
    metadata: Dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.amr_graph = penman.interpret(self.amr)
        self.metadata = self.amr_graph.metadata
        self._build_var_maps()

    def _build_var_maps(self):
        for triplet in self.amr_graph.triples:
            if triplet[1] == ":instance":
                self.varmaps[triplet[0]] = triplet[2]


if __name__ == "__main__":
    with open(r"C:\Python\projects\amr-to-sl-repr\data\vgt-amr-sentences.txt", encoding="utf-8") as fhin:
        for tree in penman.iterparse(fhin):
            meta = Converter(tree)
            print(meta.varmaps)
            break