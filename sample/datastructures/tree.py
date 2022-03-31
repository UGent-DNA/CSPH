# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

from typing import Set, List, Union, Dict

from sample.datastructures.linkedlist import Linkedlist

Node = int


class Tree:
    """ A graph tree. Does not extend Graph because the neighbour relation is stored differently."""

    def __init__(self, root):
        self.root = root
        self.parents: Dict[Node, Union[Node, None]] = {root: None}
        self.children: Dict[Union[Node, None], Set[Node]] = {None: {root}}

    def add_relation(self, n: Node, parent: Node):
        if parent not in self.children:
            self.children[parent] = set()
        if n in self.parents:
            self.children[self.parents[n]].remove(n)
        self.children[parent].add(n)
        self.parents[n] = parent

    def get_all_descendants(self, node: Node, keep_root: bool) -> List[Node]:
        """
        Returns all nodes that can for which the root-to-node path contains {node}.
        If {keep_root} also returns {node}.
        """
        counter = 0
        ordered_children: List[Node] = []
        node_queue = Linkedlist([node])
        while not node_queue.is_empty():
            counter += 1
            n = node_queue.pop_tail()
            ordered_children.append(n)
            for child in self.children.get(n, {}):
                node_queue.add_head(child)
        if not keep_root:
            ordered_children.remove(node)
        return ordered_children

    def get_nodes(self) -> Set[Node]:
        return self.parents.keys()

    def remove_leaf(self, n: Node):
        """ Remove {n} from the tree if {n} is a leaf. Otherwise throws an error. """
        if self.children.get(n, set()):
            raise ValueError("n is not a leaf")
        parent: Node = self.parents.get(n)
        if parent is not None:
            if parent not in self.children:
                raise ValueError("This should not happen")
            self.children.get(parent).remove(n)
        del self.parents[n]
        if n in self.children:
            del self.children[n]

    def remove_subtree(self, subroot: Node, keep_root: bool) -> List[Node]:
        """
        Remove the subtree starting from {subroot}; returns the removed nodes.
        If {keep_root} then do not remove it.
        """
        descendants = self.get_all_descendants(subroot, not keep_root)
        descendants.reverse()
        for n in descendants:
            self.remove_leaf(n)

        return descendants

    def get_path_from_root_to_node(self, n: Node) -> List[Node]:
        """
        Return the path from the root to the node (including both endpoints).
        This method contains loop-detection: include when testing, remove if it is a computational bottleneck.
        """
        if n not in self.parents.keys():
            raise RuntimeError(f"Node {n} not in tree")
        visited: Set[Node] = {n}
        path: List[Node] = [n]
        prev_n = n
        while n != self.root:
            n = self.parents.get(prev_n)
            if n in visited:
                raise RuntimeError("There should be no loops in a tree!")
            visited.add(n)
            path.append(n)
            prev_n = n
        path.reverse()
        return path

    def get_leaves(self) -> Set[Node]:
        """
        In O(|V|), return all leaves
        """
        return {n for n in self.parents.keys() if not self.children.get(n, set())}

    def prune(self, nodes_to_remain) -> Set[Node]:
        """
        Prune the graph (remove as many as possible) so that all nodes in {nodes_to_remain} are still connected to the
        root. Can be done in O(|V|), but this implementation is slower.
        """
        repeat = True
        pruned_nodes = set()
        while repeat:
            repeat = False
            for leaf in self.get_leaves():
                if leaf not in nodes_to_remain:
                    path_node = leaf
                    while not self.children.get(path_node, set()) and path_node not in nodes_to_remain:
                        parent = self.parents[path_node]
                        self.remove_leaf(path_node)
                        path_node = parent
                        pruned_nodes.add(path_node)
                        repeat = True
        return pruned_nodes

    def get_farthest_from_root(self, n1, n2):
        """ Restricted to nodes that have a parent-child relationship. """
        if self.parents[n1] == n2:
            return n1
        elif self.parents[n2] == n1:
            return n2
        return None

    def contains_edge(self, n1, n2):
        """ Return True if the tree contains the relation n1->n2 or n2->n1. """
        return self.parents.get(n1, None) == n2 or self.parents.get(n2, None) == n1

    def check_tree(self):
        """
        Verify that the tree is sound (it is tree and data structures are not compromised).
        Raises an error otherwise.
        """
        visited = set()
        to_process = Linkedlist([self.root])
        while not to_process.is_empty():
            node = to_process.pop_head()
            if node in visited:
                raise RuntimeError("LOOP")
            else:
                for child in self.children.get(node, set()):
                    to_process.add_tail(child)
                visited.add(node)

        if len(visited) != len(self.parents.keys()):
            print("Not all nodes can be reached from the root")

        for n, parent in self.parents.items():
            if n not in self.children[parent]:
                raise RuntimeError("Parent-child relationship is broken")
        return True
