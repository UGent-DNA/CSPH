# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

from typing import Iterable, Any


class Element:
    """ LinkedList Element, keeps track of its left and right neighbour and its data packet """

    def __init__(self, data, left, right):
        self.data = data
        self.left = left
        self.right = right


class Linkedlist:
    """ Simple linkedList implementation with pop and add methods."""

    def __init__(self, collection: Iterable[Any]):
        self.head = None
        self.tail = None
        collection_iterator = iter(collection)
        self.head = self.tail = Element(next(collection_iterator), None, None)

        for val in collection_iterator:
            self.add_tail(val)

    def add_tail(self, val):
        old_tail = self.tail
        self.tail = Element(val, old_tail, None)
        if old_tail is not None:
            old_tail.right = self.tail
        if self.head is None:
            self.head = self.tail

    def add_head(self, val):
        old_head = self.head
        self.head = Element(val, None, self.head)
        if old_head is not None:
            old_head.left = self.head
        if self.tail is None:
            self.tail = self.head

    def pop_head(self):
        old_head = self.head
        new_head = old_head.right
        if new_head is not None:
            new_head.left = None
        else:
            self.tail = None
        self.head = new_head
        return old_head.data

    def pop_tail(self):
        old_tail = self.tail
        new_tail = old_tail.left
        if new_tail is not None:
            new_tail.right = None
        else:
            self.head = None
        self.tail = new_tail

        return old_tail.data

    def is_empty(self):
        return self.head is None
