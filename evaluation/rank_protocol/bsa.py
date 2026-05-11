import os
import tempfile

from .common import copy_or_unzip_pdb, line_chain_id, parse_chain_ids, write_ordered_chain_pdb


def calculate_bsa(pdb_path, receptor_chain_ids, ligand_chain_ids):
    """Calculate ligand BSA against the selected receptor/ligand chains only."""
    import freesasa

    receptor_chain_ids = parse_chain_ids(receptor_chain_ids, "receptor chain IDs")
    ligand_chain_ids = parse_chain_ids(ligand_chain_ids, "ligand chain IDs")

    local_tmp_path = None
    complex_tmp_path = None
    ligand_tmp_path = None
    try:
        source_pdb = pdb_path
        if pdb_path.endswith(".gz"):
            local_tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
            local_tmp_path = local_tmp.name
            local_tmp.close()
            copy_or_unzip_pdb(pdb_path, local_tmp_path)
            source_pdb = local_tmp_path

        complex_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
        complex_tmp_path = complex_tmp.name
        complex_tmp.close()
        write_ordered_chain_pdb(source_pdb, complex_tmp_path, receptor_chain_ids, ligand_chain_ids)

        struct_complex = freesasa.Structure(complex_tmp_path)
        result_complex = freesasa.calc(struct_complex)
        selections = [f"lig{i}, chain {chain_id}" for i, chain_id in enumerate(ligand_chain_ids)]
        selected = freesasa.selectArea(selections, struct_complex, result_complex)
        sasa_complex = sum(float(value) for value in selected.values())

        ligand_set = set(ligand_chain_ids)
        ligand_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
        ligand_tmp_path = ligand_tmp.name
        with open(complex_tmp_path) as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")) and line_chain_id(line) in ligand_set:
                    ligand_tmp.write(line)
        ligand_tmp.write("END\n")
        ligand_tmp.close()

        sasa_isolated = freesasa.calc(freesasa.Structure(ligand_tmp_path)).totalArea()
        abs_bsa = sasa_isolated - sasa_complex
        rel_bsa = abs_bsa / sasa_isolated if sasa_isolated > 0 else 0.0
        return round(abs_bsa, 4), round(rel_bsa, 4)
    finally:
        for tmp_path in (ligand_tmp_path, complex_tmp_path, local_tmp_path):
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

