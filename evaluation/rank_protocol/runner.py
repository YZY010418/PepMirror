from concurrent.futures import ProcessPoolExecutor, as_completed

from .bsa import calculate_bsa
from .common import make_pickle_safe, pdb_name, progress_iter
from .constants import OUTPUT_COLUMNS
from .plip import calculate_plip_metrics
from .rosetta import calculate_rosetta_metrics
from .vina import calculate_vina_score


def empty_result(path):
    row = {column: "" for column in OUTPUT_COLUMNS}
    row["pdb_name"] = pdb_name(path)
    row["_input_path"] = path
    row["_hotspot_residue_interactions"] = {}
    row["_errors"] = []
    return row


def process_one_pdb(task):
    pdb_path, receptor_chain_ids, ligand_chain_ids, paths = task
    row = empty_result(pdb_path)

    try:
        abs_bsa, rel_bsa = calculate_bsa(pdb_path, receptor_chain_ids, ligand_chain_ids)
        row["absBSA"] = abs_bsa
        row["relBSA"] = rel_bsa
    except Exception as exc:
        row["_errors"].append(f"BSA: {exc}")

    try:
        row.update(calculate_plip_metrics(pdb_path, receptor_chain_ids, ligand_chain_ids, paths))
    except Exception as exc:
        row["_errors"].append(f"PLIP: {exc}")

    try:
        row["vina score"] = calculate_vina_score(pdb_path, receptor_chain_ids, ligand_chain_ids, paths)
    except Exception as exc:
        row["_errors"].append(f"Vina: {exc}")

    try:
        row.update(calculate_rosetta_metrics(pdb_path, receptor_chain_ids, ligand_chain_ids, paths))
    except Exception as exc:
        row["_errors"].append(f"Rosetta: {exc}")

    return make_pickle_safe(row)


def run_tasks(tasks, num_processors):
    if num_processors <= 1:
        return [process_one_pdb(task) for task in progress_iter(tasks, desc="rank")]

    results = []
    with ProcessPoolExecutor(max_workers=num_processors) as executor:
        futures = [executor.submit(process_one_pdb, task) for task in tasks]
        for future in progress_iter(as_completed(futures), total=len(futures), desc="rank"):
            results.append(future.result())
    order = {task[0]: i for i, task in enumerate(tasks)}
    results.sort(key=lambda row: order.get(row.get("_input_path"), 10**12))
    return results

