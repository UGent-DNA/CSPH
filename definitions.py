# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2022 IDLab/UGent - Simon Van den Eynde

import os
import random

# Fix directory names so the programs work independent of the current working directory.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is the Project Root
RESOURCE_PATH = os.path.join(ROOT_DIR, 'resources')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output')

random.seed(149675132)


def change_seed(seed: int):
    random.seed(seed)
