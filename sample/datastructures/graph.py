# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

from typing import Dict, Any, Set

from sample.datastructures.edge import Edge

Node = int


class Graph:
    """ Undirected graph. Access the nodes and neighbours through {graph.structure}."""

    def __init__(self, name):
        # The graph structure links each Node to its neighbours and its incident edges.
        self.structure: Dict[Node, Dict[Node, Edge]] = {}
        self.node_mapping = {}
        self.name = name

    def set_node_attribute(self, node, attribute, value):
        self.node_mapping[node][attribute] = value

    def add_node(self, node_id, node_attributes: Dict[str, Any]):
        self.node_mapping[node_id] = node_attributes

    def add_edge(self, n1: Node, n2: Node, length: float = 0) -> Edge:
        e = Edge(n1, n2, length)
        self._add_edge(e, e.source, e.target)
        self._add_edge(e, e.target, e.source)

        return e

    def _add_edge(self, e: Edge, source: Node, target: Node):
        if source not in self.structure:
            self.structure[source] = {}
        self.structure[source][target] = e

    def get_neighbor_dict(self, node: Node) -> Dict[Node, Edge]:
        """ Return the neighbours of a node in a dict: neighbour -> incident edge"""
        if node not in self.structure:
            print(node)
        return self.structure[node]

    def get_edges(self) -> Set[Edge]:
        """ Not efficient (takes O(|E|) time)"""
        return {e for v in self.structure.values() for e in v.values()}


if __name__ == "__main__":
    g = Graph("graph")
    print(g)
