#!/usr/bin/python
# -*- coding:utf-8 -*-
import utils.register as R
from data.bioparse.hierarchy import Atom, Block

from .base import BaseTemplate


@R.register('LinearPeptide')
class LinearPeptide(BaseTemplate):

    def dummy_lig_block_bonds(self, cplx_desc, size):
        blocks = [Block(
            name='GLY', # use glycine instead of UNK so that is_aa can be identified as True
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(i + 1, '')
        ) for i in range(size)]
        bonds = []
        return blocks, bonds