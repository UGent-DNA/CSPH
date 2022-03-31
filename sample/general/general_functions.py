# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

import math
from typing import Tuple, List

from sample.algorithm.csph import Node
from sample.datastructures.edge import Edge
from sample.datastructures.graph import Graph


def euclid_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def get_edge_path(g: Graph, node_path: List[Node]) -> List[Edge]:
    """ In O(|{node_path}|) return all the edges on the path, in order """
    return [g.structure[node_path[i - 1]][node_path[i]] for i in range(1, len(node_path))]
