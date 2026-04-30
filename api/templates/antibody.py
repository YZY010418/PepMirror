#!/usr/bin/python
# -*- coding:utf-8 -*-
import utils.register as R

from data.bioparse.utils import recur_index
from data.bioparse.hierarchy import Atom, Block
from scripts.data_process.antibody.sabdab import Chothia, _get_model_id_mask

from .base import BaseTemplate


@R.register('Antibody')
class Antibody(BaseTemplate):

    def __init__(self, cdr_type='HCDR3', fr_len=3, size_min=None, size_max=None):
        super().__init__(size_min, size_max)
        self.cdr_type = cdr_type # H/LCDR1/2/3
        self.fr_len = fr_len

    def sample_size(self, cplx_desc):
        raise ValueError(f'Do not sample size for CDR loop')

    def remove_ref_lig(self, cplx_desc):
        # do not remove the antibody, because we need the framework
        return cplx_desc.cplx
    
    def add_dummy_lig(self, cplx_desc):
        cplx = cplx_desc.cplx

        # identify heavy chain and light chain
        if len(cplx_desc.lig_chains) == 2:  # assume heavy chain goes first
            hc, lc = cplx_desc.lig_chains
        elif len(cplx_desc.lig_chains) == 1:
            if self.cdr_type.startswith('H'):
                hc, lc = cplx_desc.lig_chains[0], None
            else:
                hc, lc = None, cplx_desc.lig_chains[0]
        else:
            raise ValueError(f'Number of antibody chains not correct: got {len(cplx_desc.lig_chains)}, but expect 1 or 2')
        
        # mark residues in the variable domain
        # heavy chain
        if hc is not None:
            blocks = [block for block in cplx[hc] \
                      if block.id[0] >= Chothia.HFR1[0] and block.id[0] <= Chothia.HFR4[-1]
                      ]
            ids = [block.id for block in blocks]
            heavy_cdr = Chothia.mark_heavy_seq([_id[0] for _id in ids])
            heavy_block_ids = [(hc, _id) for _id in ids]
            # modeling blocks (CDR +fr_len blocks and -fr_len blocks)
            heavy_model_block_id, heavy_model_mark = _get_model_id_mask(heavy_block_ids, heavy_cdr, self.fr_len)
        else:
            heavy_model_block_id, heavy_model_mark = [], []
    
        # light chain
        if lc is not None:
            blocks = [block for block in cplx[lc] \
                      if block.id[0] >= Chothia.LFR1[0] and block.id[0] <= Chothia.LFR4[-1]
                      ]
            ids = [block.id for block in blocks]
            light_cdr = Chothia.mark_light_seq([_id[0] for _id in ids])
            light_block_ids = [(lc, _id) for _id in ids]
            light_model_block_id, light_model_mark = _get_model_id_mask(light_block_ids, light_cdr, self.fr_len)
        else:
            light_model_block_id, light_model_mark = [], []

        # get generate mask
        generate_mask = [0 for _ in cplx_desc.pocket_block_ids]
        cdr = self.cdr_type
        for m in heavy_model_mark:
            if cdr.startswith('H') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)
        for m in light_model_mark:
            if cdr.startswith('L') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)

        # centering at the medium of two ends
        center_mask = [0 for _ in generate_mask]
        for i in range(len(center_mask)):
            if i + 1 < len(generate_mask) and generate_mask[i + 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1 # left end
            elif i - 1 > 0 and generate_mask[i - 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1

        # remove ground truth cdr
        for m, block_id in zip(generate_mask, cplx_desc.pocket_block_ids + heavy_model_block_id + light_model_block_id):
            if m == 0: continue
            block: Block = recur_index(cplx, block_id)
            block.name = 'GLY'
            block.atoms = [Atom(    # we should retain the atom ids to ensure correct chemical bonding
                name='C',
                coordinate=[0, 0, 0],
                element='C',
                id=atom.id
            ) for atom in block.atoms]

        # set desc here
        cplx_desc.generate_mask = generate_mask
        cplx_desc.center_mask = center_mask
        cplx_desc.lig_block_ids = heavy_model_block_id + light_model_block_id
        
        return cplx
    
    def set_desc_attr(self, cplx_desc):
        # already set above
        return cplx_desc