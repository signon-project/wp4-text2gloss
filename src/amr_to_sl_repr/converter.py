from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import penman


@dataclass(repr=False)
class Converter:
    tree: penman.Tree = field(default=None)
    graph: penman.Graph = field(default=None)
    split_graphs: List[penman.Graph] = field(default_factory=list)
    varmaps: Dict[str, str] = field(default_factory=dict, init=False)
    metadata: Dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if not self.tree and not self.graph:
            raise ValueError("At least either 'amr' (tree) or 'amr_graph' must be given")
        elif not self.graph:
            self.graph = penman.interpret(self.tree)
        elif not self.tree:
            self.tree = penman.configure(self.graph)

        self.metadata = self.graph.metadata
        self._build_var_maps()
        self._split_arg_graphs()

    def _build_var_maps(self):
        for triplet in self.graph.triples:
            if triplet[1] == ":instance":
                self.varmaps[triplet[0]] = triplet[2]

    def _split_arg_graphs(self):
        """Split the main graph in cases where we have one parent node with multiple :ARGX children.
        Subgraphs are saved in self.split_graphs
        """

        """Collect which main structures are splitable. That means, structures that have more than two
        :ARG roles on the same level, e.g.
            (a / arrive-01
               :ARG1 (t / train
                        :destination (s / station
                                        :name (n / name
                                                 :op1 "Anterwerpen"
                                                 :op2 "Centraal")))
               :ARG4 (t2 / track
                         :mod 5))

        should lead to {'a': ['t', 't2']}, which is a mapping from the parent varnode to the children varnodes (:ARGs)
        """
        splitable = defaultdict(list)
        for triplet in self.graph.triples:
            if triplet[1].startswith(":ARG") and not triplet[1].endswith("-of"):
                splitable[triplet[0]].append(triplet[2])
        splitable = dict(splitable)

        # Now that we know the split candidates, we can iterate over them and create graphs that
        # only include one of the options. So from the example above, we will generate two new graphs,
        # one that only contains the t subgraph and one with only t2
        for rootvar, splitvars in splitable.items():
            for splitvar in splitvars:
                new_triples = []
                to_ignore = set()
                for source, role, target in self.graph.triples:
                    if source == rootvar and role.startswith(":ARG") and not role.endswith("-of"):
                        # we only want to include splitvar for this rootvar
                        if target != splitvar:
                            to_ignore.add(target)
                            continue
                    elif source in to_ignore:
                        # For deeper triples: if this source is in to_ignore, continue
                        # and also add the current target to to_ignore so that we recursively ignore this
                        # whole subgraph
                        if role != ":instance":
                            to_ignore.add(target)
                        continue

                    new_triples.append((source, role, target))
                self.split_graphs.append(penman.Graph(new_triples))


if __name__ == "__main__":
    with open(r"F:\python\amr-to-sl-repr\data\vgt-amr-sentences.txt", encoding="utf-8") as fhin:
        for tree in penman.iterparse(fhin):
            meta = Converter(tree=tree)
            print(f"MAIN GRAPH\n==========")
            print(penman.encode(meta.graph))
            for graph_idx, subgraph in enumerate(meta.split_graphs, 1):
                print(f"SUB GRAPH #{graph_idx} \n==========")
                print(penman.encode(subgraph))
                print()
