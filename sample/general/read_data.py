# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

"""
Read stp instances from steinlib.
"""

import os
from typing import Tuple, List, Dict

from definitions import RESOURCE_PATH
from sample.datastructures.graph import Graph
from sample.general.general_functions import euclid_distance

Terminals = List[int]


def generate_steinlib_graphs(dir_name) -> Tuple[Graph, Terminals, float]:
    sol_dict = _read_solutions_steinlib(dir_name)
    for name in _generate_steinlib_names(dir_name):
        g, t = read_stp(os.path.join(dir_name, name))
        graph_name = name.split('.')[0]
        yield g, t, sol_dict[graph_name], graph_name


def _generate_steinlib_names(dir_name) -> List[str]:
    """ Collect the names of all the steinerlib files in the directory {dir_name}. Names in the format *.stp """
    names = sorted(os.listdir(f'{RESOURCE_PATH}/{dir_name}'))
    names.remove('solutions.txt')
    names = [name for name in names if name.split(".")[-1] == 'stp']
    return names


def _read_solutions_steinlib(dir_name) -> Dict[str, int]:
    """ Solutions from the file 'name' in a dict format: Key = filename (no extension), Value = optimal cost """
    name_cost_dict = {}
    with open(os.path.join(RESOURCE_PATH, dir_name, 'solutions.txt')) as f:
        for line in f:
            line = line.strip()
            sol = line.split(' ')
            name_cost_dict[sol[0]] = int(sol[-1])
    return name_cost_dict


if __name__ == '__main__':
    for n in _generate_steinlib_names('vienna'):
        print(n)


def read_stp(name) -> Tuple[Graph, Terminals]:
    """
    Read an stp file and return the networkx graph with positions and edge weights. And return terminals.
    File format description: http://steinlib.zib.de/format.php
    """

    path = os.path.join(RESOURCE_PATH, name)
    name_without_extension_nor_directory = os.path.split(name)[1].split(".")[0]
    g = Graph(name_without_extension_nor_directory)
    terminals = []
    with open(path, 'r') as f:
        for line in f:
            words = line.split(' ')
            if words[0] == 'section' and words[1] == 'presolve':
                break

            var = words[0]
            if var == 'DD':
                g.add_node(int(words[1]), {"pos": (int(words[2]), int(words[3]))})
            if var == 'E':
                u, v, weight = map(int, words[1:])
                e = g.add_edge(u, v)
                e.weight = weight
            if var == 'T':
                terminals.append(int(words[1]))

        for node in g.structure.keys():
            for neigh, e in g.get_neighbor_dict(node).items():
                if node in g.node_mapping.keys() and "pos" in g.node_mapping[node].keys():
                    e.length = euclid_distance(g.node_mapping[node]["pos"], g.node_mapping[neigh]["pos"])
                else:
                    e.length = e.weight

    return g, terminals
