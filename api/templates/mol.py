#!/usr/bin/python
# -*- coding:utf-8 -*-
from data.bioparse.utils import recur_index
from data.resample import SizeSamplerByPocketSpace
import utils.register as R

from .base import BaseTemplate


@R.register('Molecule')
class Molecule(BaseTemplate):

    def __init__(self, size_min=None, size_max=None):
        super().__init__(size_min, size_max)
        self.size_sampler = SizeSamplerByPocketSpace(size_min, size_max)
    
    def sample_size(self, cplx_desc):
        pocket_pos = []
        for _id in cplx_desc.pocket_block_ids:
            block = recur_index(cplx_desc.cplx, _id)
            for atom in block: pocket_pos.append(atom.coordinate)
        
        num_blocks = self.size_sampler(1, pocket_pos)[0]

        return num_blocks

    def set_control(self, data):
        data['is_aa'][data['generate_mask']] = False
        data['position_ids'][data['generate_mask']] = 0
        return data
    
    def validate(self, cplx_desc):
        return True
