import os

DEFAULT_VINA = "vina"
DEFAULT_MGL_PYTHON = "auto"
DEFAULT_PREP_LIGAND = "prepare_ligand4.py"
DEFAULT_PREP_RECEPTOR = "prepare_receptor4.py"
DEFAULT_PYROSETTA_FLAGS = "-mute all -out:level 0 -ignore_unrecognized_res true -load_PDB_components false"
DEFAULT_PLIP = "plip"
DEFAULT_ROSETTA_XML = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RosettaRankingScript.xml")

INTERACTION_TYPES = [
    "hydrophobic_interactions",
    "hb_mainchain",
    "hb_sidechain",
    "salt_bridges",
    "pi_stacking",
    "cation_pi",
    "metal_complexes",
]

OUTPUT_COLUMNS = [
    "pdb_name",
    "absBSA",
    "relBSA",
    "vina score",
    "sc",
    "ec",
    "buried_Hbonds",
    "num(interaction)",
    "num(H_bonds)",
    "num(mainchain_Hbonds)",
    "hotspot_occupoed weighted",
]

ROSETTA_DDG_COLUMNS = [
    "rosetta_ddg",
    "rosetta_ddg_norepack",
]

ERROR_COLUMNS = ["pdb_name", "input_path", "errors"]
