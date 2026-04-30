##Introduction

This is for reproducing the results in the manuscript "Cross-Chirality Generalization by Axial Vectors for Hetero-Chiral Protein-Peptide Interaction Design " submitted to ICML 2026.


## Setup

We provide conda environment configurations for **cuda 11.7 + pytorch 1.13.1** (`environment..yaml`) 

```bash
conda env create -f environment..yaml
```

The environment can be actibated by

```bash
conda activate PepMirror
```

## Training

The datasets used for training can be downloaded from zenodo.
```bash
# PepBench
wget https://zenodo.org/records/13373108/files/train_valid.tar.gz?download=1 -O datasets/pepbench.tar.gz
# ProtFrag
wget https://zenodo.org/records/13373108/files/ProtFrag.tar.gz?download=1 -O datasets/ProtFrag.tar.gz
```
We processed these dataset to mmap.
```bash
python -m scripts.data_process.peptide.pepbench --index ${PREFIX}/pepbench/all.txt --out_dir ${PREFIX}/pepbench/processed
python -m scripts.data_process.peptide.transform_index --train_index ${PREFIX}/pepbench/train.txt --valid_index ${PREFIX}/pepbench/valid.txt --all_index_for_non_standard ${PREFIX}/pepbench/all.txt --processed_dir ${PREFIX}/pepbench/processed/
python -m scripts.data_process.peptide.pepbench --index ${PREFIX}/ProtFrag/all.txt --out_dir ${PREFIX}/ProtFrag/processed
```
Then, we used 8 GPUs with 80G memmory each to train PepMirror,  which takes about 2 days to finish. We enabled TF32 for acceleration.
```bash
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
GPU=0,1,2,3,4,5,6,7 bash scripts/train_pipe.sh ./ckpts/pepmirror ./configs/IterAE/train.yaml ./configs/LDM/train.yaml
```
To set the type of axial vectors used in the model, you can change the `axial_type` in `./configs/IterAE/train.yaml` and `./configs/LDM/train.yaml`, where three types of axial features are implemented: cross, triple, and commutator, as discussed in the paper.

##Inference

We used LNR as our test set. First, we used Rosetta to clean the complex structures in LNR, and fixed PDB files mannually to avoid uncessary trouble (for example, some PDB have ACE/NME as capping, and these structures are recognized as a residue).

We used CB distances to define pockets. For Gly, we reconstructed virtual CB for pocket detection, yet this process requires chirality information. Therefore, we treat Gly as an L amino acid when dealing with L-LNR, and as a D amino acid when dealing with LNR_mirror, in order to ensure consistency between pockets of L/D-LNR.

The processed mmap of LNR and LNR_mirror can be found at `./dataset/LNR.tar.gz`. 

Then, we design 100 structures for each LNR complex.

```bash
python generate.py --config configs/test/test_pep.yaml --ckpt {path/to/checkpoint} --gpu 0 --save_dir {output/path}
```

The generated structures are minimized under Amber14 forcefield.

```bash
python evaluation/openmm_relaxer_mp.py {path/to/raw/structures} {path/for/minimized/structures} --nproc {number_of_protocols} 
```

Minimized structures are then evaluated as described in our manuscript.
