#!/usr/bin/env python3

import os
import argparse
import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from tqdm import tqdm

def read_structure(path):
    pdb = PDBFile.read(path)
    arrays = pdb.get_structure()
    array = arrays[0]
    return array

def get_atom_pos(residue, name):
    mask = (residue.atom_name == name)
    idx = np.nonzero(mask)[0]
    return residue.coord[idx[0]] if idx.size > 0 else None

def ca_chirality(N, CA, C, CB):
    vN = N - CA
    vC = C - CA
    vB = CB - CA
    normal = np.cross(vN, vC)
    return "D" if float(np.dot(normal, vB)) < 0.0 else "L"

def residue_chirality(residue):
    if residue.res_name[0] == "GLY":
        return "G"
    N = get_atom_pos(residue, "N")
    CA = get_atom_pos(residue, "CA")
    C = get_atom_pos(residue, "C")
    CB = get_atom_pos(residue, "CB")
    if any(x is None for x in (N, CA, C, CB)):
        return None
    return ca_chirality(N, CA, C, CB)

def parse_chains_arg(s):
    if not s:
        return None
    chain_ids = []
    for chain_id in s.split(","):
        chain_ids.append(chain_id.strip())
    return chain_ids

def analyze_file(path, chains=None):
    array = read_structure(path)
    if chains is not None:
        chain_mask = np.isin(array.chain_id, chains)
        array = array[chain_mask]

    chiralty_array = []
    for item in struc.residue_iter(array):
        chiralty = residue_chirality(item)
        chiralty_array.append(chiralty)
    
    return chiralty_array

def main():
    parser = argparse.ArgumentParser(description="count L/D/G/None chirality in PDB files")
    parser.add_argument("folder", help="folder path containing PDB files (recursively traversed)")
    parser.add_argument("-c", "--chains", help="specify chains, e.g., 'A,B,C'")
    args = parser.parse_args()

    chains = parse_chains_arg(args.chains)

    files = []
    for root, _, files_in_root in os.walk(args.folder):
        for file in files_in_root:
            if file.endswith(".pdb"):
                files.append(os.path.join(root, file))

    count_L = 0
    count_D = 0
    count_CA = 0  
    count_missing = 0  

    for file in tqdm(files, desc="processing", unit="it"):
        chirality_array = analyze_file(file, chains=chains)
        for chirality in chirality_array:
            if chirality == "L":
                count_L += 1
                count_CA += 1
            elif chirality == "D":
                count_D += 1
                count_CA += 1
            elif chirality is None:
                count_missing += 1

    print("\nstatistics:")
    print(f"total CA number (valid residues): {count_CA}")
    print(f"L-type residues count: {count_L}, ratio: {count_L/count_CA:.4f}")
    print(f"D-type residues count: {count_D}, ratio: {count_D/count_CA:.4f}")
    print(f"residues with missing critical atoms count: {count_missing}")

if __name__ == "__main__":
    main()
