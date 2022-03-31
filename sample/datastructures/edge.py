# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

class Edge:
    def __init__(self, source: int, target: int, length: float):
        if source < target:
            self.source = source
            self.target = target
        else:
            self.target = source
            self.source = target
        self.length = length
        self.weight = 1
        self.capacity = 1000

    def get_other_node(self, node: int):
        return self.source if node == self.target else self.target


if __name__ == "__main__":
    e = Edge(1, 2, 3)

    print(e)
