#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from typing import List

import torch

from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.interface import compute_pocket
from data.base import transform_data

from .templates import BaseTemplate, ComplexDesc


class PDBDataset(torch.utils.data.Dataset):
    def __init__(self, pdb_paths: List[str], tgt_chains: List[List[str]], lig_chains: List[List[str]], template_config: BaseTemplate, n_samples: int=1):
        super().__init__()
        self.pdb_paths = pdb_paths
        self.tgt_chains = tgt_chains
        self.lig_chains = lig_chains
        self.cplxs, self.pocket_block_ids = [], []
        for path, tgt, lig in zip(pdb_paths, tgt_chains, lig_chains):
            cplx, pocket_block_ids = self._process_pdb(path, tgt, lig)
            self.cplxs.append(cplx)
            self.pocket_block_ids.append(pocket_block_ids)
        self.n_samples = n_samples
        self.config = template_config

    def __getitem__(self, idx):
        idx = idx // self.n_samples
        cplx, pocket_block_ids = self.cplxs[idx], self.pocket_block_ids[idx]

        # edit the cplx with template (create new cplx of ligand and merge)
        cplx_desc = self.config(ComplexDesc(
            id=os.path.basename(os.path.splitext(self.pdb_paths[idx])[0]),
            cplx=cplx,
            tgt_chains=list(self.tgt_chains[idx]),
            lig_chains=list(self.lig_chains[idx]),
            pocket_block_ids=pocket_block_ids
        ))

        dummy_lig_block_ids = cplx_desc.lig_block_ids

        data = transform_data(cplx_desc.cplx, pocket_block_ids + dummy_lig_block_ids)
        data['generate_mask'] = torch.tensor(cplx_desc.generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(cplx_desc.center_mask, dtype=torch.bool)
        if cplx_desc.topo_generate_mask is not None:
            data['topo_generate_mask'] = torch.tensor(cplx_desc.topo_generate_mask, dtype=torch.bool)

        data = self.config.set_control(data)
        data['cplx_desc'] = cplx_desc

        return data

    def __len__(self):
        return len(self.cplxs) * self.n_samples
    
    def _process_pdb(self, pdb_path, tgt_chains, lig_chains, pocket_dist_th=10.0):
        cplx = pdb_to_complex(pdb_path, tgt_chains + lig_chains)
        frag_lig_chains = []
        for mol in cplx:
            for c in lig_chains:
                if c + '_' in mol.id: frag_lig_chains.append(mol.id) # small molecules fragmented
        pocket_block_ids, _ = compute_pocket(cplx, tgt_chains, list(lig_chains) + frag_lig_chains, dist_th=pocket_dist_th)
        return cplx, pocket_block_ids
    
    def collate_fn(self, batch):
        results = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            elif key == 'bonds': # need to add offsets
                offset = 0
                for i, bonds in enumerate(values):
                    bonds[:, :2] = bonds[:, :2] + offset # src/dst
                    offset += len(batch[i]['A'])
                results[key] = torch.cat(values, dim=0)
            elif key == 'cplx_desc':
                results[key] = values
            else:
                results[key] = torch.cat(values, dim=0)
        return results
    

if __name__ == '__main__':
    import sys
    dataset = PDBDataset([sys.argv[1]], sys.argv[2], sys.argv[3], BaseTemplate(10, 12)) # e.g. xxx.pdb AB CD
    print(dataset[0])