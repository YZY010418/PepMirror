#!/usr/bin/python
# -*- coding:utf-8 -*-
from .base import BaseTemplate, ComplexDesc
from .pep import LinearPeptide

try:
    from .mol import Molecule
except ModuleNotFoundError:
    Molecule = None

try:
    from .antibody import Antibody
except ModuleNotFoundError:
    Antibody = None
