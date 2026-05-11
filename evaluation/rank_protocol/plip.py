import os
import shutil
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET

from .common import (
    chain_ids_text,
    copy_or_unzip_pdb,
    filter_existing_chain_ids_in_pdb,
    line_chain_id,
    parse_chain_ids,
    pdb_atom_coord,
    pdb_name,
    pdb_residue_key,
    resolve_command,
    within_cutoff,
    write_ordered_chain_pdb,
)
from .constants import INTERACTION_TYPES


def write_plip_interface_pdb(input_pdb, output_pdb, full_chain_ids, trim_chain_ids, cutoff=20.0):
    """Write a local-interface PDB for PLIP.

    Chains in full_chain_ids are kept completely, while residues in trim_chain_ids
    are retained only if any atom is within cutoff Å of any atom in full_chain_ids.
    This is intended for PLIP-only analysis, where the interaction search is local
    and does not require the complete global protein structure.
    """
    full_chain_ids = parse_chain_ids(full_chain_ids, "PLIP full chain IDs")
    trim_chain_ids = parse_chain_ids(trim_chain_ids, "PLIP trimmed chain IDs")
    full_set = set(full_chain_ids)
    trim_set = set(trim_chain_ids) - full_set
    cutoff = float(cutoff)
    if cutoff <= 0:
        shutil.copy2(input_pdb, output_pdb)
        return

    with open(input_pdb) as handle:
        atom_lines = [line for line in handle if line.startswith(("ATOM", "HETATM"))]

    ref_coords = []
    for line in atom_lines:
        if line_chain_id(line) in full_set:
            try:
                ref_coords.append(pdb_atom_coord(line))
            except Exception:
                continue
    if not ref_coords:
        raise ValueError(
            f"PLIP trimming reference chain(s) {chain_ids_text(full_chain_ids)} have no atoms in {input_pdb}"
        )

    cutoff_sq = cutoff * cutoff
    keep_residues = set()
    for line in atom_lines:
        chain_id = line_chain_id(line)
        if chain_id not in trim_set:
            continue
        try:
            if within_cutoff(pdb_atom_coord(line), ref_coords, cutoff_sq):
                keep_residues.add(pdb_residue_key(line))
        except Exception:
            continue

    chain_order = []
    for chain_id in list(trim_chain_ids) + list(full_chain_ids):
        if chain_id not in chain_order:
            chain_order.append(chain_id)

    lines_by_chain = {chain_id: [] for chain_id in chain_order}
    for line in atom_lines:
        chain_id = line_chain_id(line)
        if chain_id in full_set:
            lines_by_chain.setdefault(chain_id, []).append(line)
        elif chain_id in trim_set and pdb_residue_key(line) in keep_residues:
            lines_by_chain.setdefault(chain_id, []).append(line)

    with open(output_pdb, "w") as fout:
        serial = 1
        for chain_id in chain_order:
            wrote = False
            last_atom = None
            for line in lines_by_chain.get(chain_id, []):
                fout.write(line[:6] + f"{serial:5d}" + line[11:])
                serial += 1
                wrote = True
                last_atom = line
            if wrote:
                fout.write(f"TER   {serial:5d}      {last_atom[17:20]} {chain_id:1s}{last_atom[22:27]}\n")
                serial += 1
        fout.write("END\n")


def run_plip(plip_path, pdb_path, peptide_chain_ids, tmp_dir, timeout=None):
    peptide_chain_ids = parse_chain_ids(peptide_chain_ids, "PLIP peptide chain IDs")
    work_id = f"{pdb_name(pdb_path)}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    temp_pdb = os.path.join(tmp_dir, f"{work_id}.pdb")
    copy_or_unzip_pdb(pdb_path, temp_pdb)
    out_dir = os.path.join(tmp_dir, work_id)
    cmd = [
        resolve_command(plip_path),
        "-f", temp_pdb,
        "--peptides", chain_ids_text(peptide_chain_ids),
        "-x",
        "-o", out_dir,
        "--quiet",
        "--maxthreads", "1",
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"PLIP timeout after {timeout:g}s for peptide chain(s) {chain_ids_text(peptide_chain_ids)}"
        )
    xml_path = os.path.join(out_dir, "report.xml")
    if result.returncode != 0 and not os.path.exists(xml_path):
        message = result.stderr.strip().splitlines() or result.stdout.strip().splitlines() or ["PLIP failed"]
        raise RuntimeError(message[0])
    if not os.path.exists(xml_path):
        raise RuntimeError(
            f"PLIP did not create report.xml for peptide chain(s) {chain_ids_text(peptide_chain_ids)}"
        )
    return xml_path


def xml_node_reschain(node):
    """Return PLIP's partner residue chain ID for an interaction node."""
    value = node.findtext("reschain")
    return value.strip() if value else ""


def plip_partner_allowed(node, allowed_reschains=None):
    """Check whether a PLIP interaction should be counted.

    In PLIP peptide mode, the reported reschain is the non-peptide partner
    residue chain.  When a local PDB contains several nearby chains, this
    filter prevents peptide--peptide or receptor--receptor contacts from being
    counted as the intended receptor--ligand interface.
    """
    if allowed_reschains is None:
        return True
    allowed = set(parse_chain_ids(allowed_reschains, "allowed PLIP partner chain IDs"))
    return xml_node_reschain(node) in allowed


def count_plip_interactions(xml_file, allowed_reschains=None):
    counts = {key: 0 for key in INTERACTION_TYPES}
    if not os.path.exists(xml_file):
        return counts

    tree = ET.parse(xml_file)
    root = tree.getroot()
    for site in root.findall(".//bindingsite"):
        interactions = site.find("interactions")
        if interactions is None:
            continue

        hydro = interactions.find("hydrophobic_interactions")
        if hydro is not None:
            for node in hydro.findall("hydrophobic_interaction"):
                if plip_partner_allowed(node, allowed_reschains):
                    counts["hydrophobic_interactions"] += 1

        hb_nodes = interactions.find("hydrogen_bonds")
        if hb_nodes is not None:
            for hb in hb_nodes.findall("hydrogen_bond"):
                if not plip_partner_allowed(hb, allowed_reschains):
                    continue
                sidechain = hb.find("sidechain")
                if sidechain is not None and sidechain.text == "False":
                    counts["hb_mainchain"] += 1
                else:
                    counts["hb_sidechain"] += 1

        salt = interactions.find("salt_bridges")
        if salt is not None:
            for node in salt.findall("salt_bridge"):
                if plip_partner_allowed(node, allowed_reschains):
                    counts["salt_bridges"] += 1

        pi = interactions.find("pi_stacks")
        if pi is not None:
            for node in pi.findall("pi_stack"):
                if plip_partner_allowed(node, allowed_reschains):
                    counts["pi_stacking"] += 1

        cation = interactions.find("pi_cation_interactions")
        if cation is not None:
            for node in cation.findall("pi_cation_interaction"):
                if plip_partner_allowed(node, allowed_reschains):
                    counts["cation_pi"] += 1

        metal = interactions.find("metal_complexes")
        if metal is not None:
            for node in metal.findall("metal_complex"):
                if plip_partner_allowed(node, allowed_reschains):
                    counts["metal_complexes"] += 1
    return counts


def extract_hotspot_residue_interactions(xml_file, allowed_reschains=None):
    residue_map = {}
    if not os.path.exists(xml_file):
        return residue_map

    allowed = None
    if allowed_reschains is not None:
        allowed = set(parse_chain_ids(allowed_reschains, "allowed hotspot receptor chain IDs"))

    def ensure(key):
        if key not in residue_map:
            residue_map[key] = {kind: 0 for kind in INTERACTION_TYPES}
        return residue_map[key]

    def res_key(node):
        resnr = node.findtext("resnr")
        restype = node.findtext("restype")
        reschain = node.findtext("reschain")
        if resnr is None or restype is None or reschain is None:
            return None
        reschain = reschain.strip()
        if allowed is not None and reschain not in allowed:
            return None
        return f"{reschain}:{resnr}:{restype}"

    tree = ET.parse(xml_file)
    root = tree.getroot()
    for site in root.findall(".//bindingsite"):
        interactions = site.find("interactions")
        if interactions is None:
            continue

        mapping = [
            ("hydrophobic_interactions", "hydrophobic_interaction", "hydrophobic_interactions"),
            ("salt_bridges", "salt_bridge", "salt_bridges"),
            ("pi_stacks", "pi_stack", "pi_stacking"),
            ("pi_cation_interactions", "pi_cation_interaction", "cation_pi"),
            ("metal_complexes", "metal_complex", "metal_complexes"),
        ]
        for parent_tag, child_tag, out_key in mapping:
            nodes = interactions.find(parent_tag)
            if nodes is None:
                continue
            for node in nodes.findall(child_tag):
                key = res_key(node)
                if key:
                    ensure(key)[out_key] += 1

        hb_nodes = interactions.find("hydrogen_bonds")
        if hb_nodes is not None:
            for hb in hb_nodes.findall("hydrogen_bond"):
                key = res_key(hb)
                if not key:
                    continue
                sidechain = hb.find("sidechain")
                if sidechain is not None and sidechain.text == "False":
                    ensure(key)["hb_mainchain"] += 1
                else:
                    ensure(key)["hb_sidechain"] += 1
    return residue_map


def sum_counts(counts):
    return sum(int(counts.get(kind, 0)) for kind in INTERACTION_TYPES)


def add_interaction_counts(total_counts, counts):
    """Add PLIP interaction counts into total_counts in place."""
    for kind in INTERACTION_TYPES:
        total_counts[kind] = int(total_counts.get(kind, 0)) + int(counts.get(kind, 0))
    return total_counts


def merge_residue_interaction_maps(total_map, residue_map):
    """Merge hotspot residue interaction maps in place."""
    for residue, counts in (residue_map or {}).items():
        total_counts = total_map.setdefault(residue, {kind: 0 for kind in INTERACTION_TYPES})
        add_interaction_counts(total_counts, counts)
    return total_map


def count_plip_interactions_for_chains(
    plip_path, pdb_path, peptide_chain_ids, tmp_dir, mode="combined", timeout=None,
    allowed_partner_chain_ids=None,
):
    """Run PLIP for peptide chain IDs and return summed interaction counts.

    mode="combined" runs one PLIP call with --peptides A,C. This is faster and
    should be used by default. mode="per_chain" mirrors the standalone backbone
    script by running one PLIP call per chain and summing counts.
    """
    peptide_chain_ids = parse_chain_ids(peptide_chain_ids, "PLIP peptide chain IDs")
    peptide_chain_ids = filter_existing_chain_ids_in_pdb(pdb_path, peptide_chain_ids)
    total = {kind: 0 for kind in INTERACTION_TYPES}
    if not peptide_chain_ids:
        return total

    if mode == "combined":
        xml_file = run_plip(plip_path, pdb_path, peptide_chain_ids, tmp_dir, timeout=timeout)
        add_interaction_counts(total, count_plip_interactions(xml_file, allowed_reschains=allowed_partner_chain_ids))
        return total

    if mode != "per_chain":
        raise ValueError(f"unsupported PLIP mode: {mode}")

    for chain_id in peptide_chain_ids:
        xml_file = run_plip(plip_path, pdb_path, [chain_id], tmp_dir, timeout=timeout)
        add_interaction_counts(total, count_plip_interactions(xml_file, allowed_reschains=allowed_partner_chain_ids))
    return total


def extract_hotspot_interactions_for_chains(
    plip_path, pdb_path, peptide_chain_ids, tmp_dir, mode="combined", timeout=None,
    allowed_receptor_chain_ids=None,
):
    """Run PLIP for ligand/design peptide chains and merge hotspot maps."""
    peptide_chain_ids = parse_chain_ids(peptide_chain_ids, "PLIP peptide chain IDs")
    peptide_chain_ids = filter_existing_chain_ids_in_pdb(pdb_path, peptide_chain_ids)
    merged = {}
    if not peptide_chain_ids:
        return merged

    if mode == "combined":
        xml_file = run_plip(plip_path, pdb_path, peptide_chain_ids, tmp_dir, timeout=timeout)
        merge_residue_interaction_maps(
            merged, extract_hotspot_residue_interactions(xml_file, allowed_reschains=allowed_receptor_chain_ids)
        )
        return merged

    if mode != "per_chain":
        raise ValueError(f"unsupported PLIP mode: {mode}")

    for chain_id in peptide_chain_ids:
        xml_file = run_plip(plip_path, pdb_path, [chain_id], tmp_dir, timeout=timeout)
        merge_residue_interaction_maps(
            merged, extract_hotspot_residue_interactions(xml_file, allowed_reschains=allowed_receptor_chain_ids)
        )
    return merged


def compute_hotspot_weighted_scores(results, topk):
    global_totals = {}
    global_type_counts = {}
    for result in results:
        for residue, counts in result.get("_hotspot_residue_interactions", {}).items():
            total = sum_counts(counts)
            if total <= 0:
                continue
            global_totals[residue] = global_totals.get(residue, 0) + total
            global_type_counts.setdefault(residue, {kind: 0 for kind in INTERACTION_TYPES})
            for kind in INTERACTION_TYPES:
                global_type_counts[residue][kind] += int(counts.get(kind, 0))

    top_residues = [residue for residue, _ in sorted(global_totals.items(), key=lambda item: item[1], reverse=True)[:topk]]
    top_set = set(top_residues)
    if not top_set:
        return {result["_input_path"]: 0.0 for result in results}

    per_residue_ratios = {}
    for residue in top_set:
        counts = global_type_counts.get(residue, {})
        total = sum_counts(counts)
        per_residue_ratios[residue] = {
            kind: (counts.get(kind, 0) / total) if total > 0 else 0.0
            for kind in INTERACTION_TYPES
        }

    scores = {}
    denom = len(top_set)
    for result in results:
        weighted_hits = 0.0
        rmap = result.get("_hotspot_residue_interactions", {})
        for residue in set(rmap).intersection(top_set):
            current_types = [kind for kind in INTERACTION_TYPES if int(rmap[residue].get(kind, 0)) > 0]
            weights = per_residue_ratios.get(residue, {})
            weighted_hits += min(1.0, sum(float(weights.get(kind, 0.0)) for kind in current_types))
        scores[result["_input_path"]] = round(100.0 * weighted_hits / denom, 4)
    return scores


def calculate_plip_metrics(pdb_path, receptor_chain_ids, ligand_chain_ids, paths):
    receptor_chain_ids = parse_chain_ids(receptor_chain_ids, "receptor chain IDs")
    ligand_chain_ids = parse_chain_ids(ligand_chain_ids, "ligand chain IDs")
    plip_path = paths["plip"]
    plip_timeout = paths.get("plip_timeout", None)

    # The chains used for PLIP backbone/mainchain counting do not have to be
    # identical to the whole receptor group used by Rosetta/Vina.  For example,
    # receptor_chain_ids may be A,C, but only C may be the peptide chain whose
    # backbone H-bonds should be counted.
    plip_count_chain_ids = paths.get("plip_count_chain_ids") or receptor_chain_ids
    plip_count_chain_ids = parse_chain_ids(plip_count_chain_ids, "PLIP count peptide chain IDs")

    # Keep count mode and hotspot mode separate. combined is the PepMirror
    # default because it avoids one PLIP subprocess per receptor chain. If a
    # PLIP build mishandles comma-separated peptide chains, use per_chain.
    plip_count_mode = paths.get("plip_count_mode", "combined")
    plip_hotspot_mode = paths.get("plip_hotspot_mode", "combined")

    with tempfile.TemporaryDirectory(prefix="rank_plip_") as tmp_dir:
        base = pdb_name(pdb_path)
        source_pdb = os.path.join(tmp_dir, f"{base}.pdb")
        selected_pdb = os.path.join(tmp_dir, f"{base}_selected.pdb")
        copy_or_unzip_pdb(pdb_path, source_pdb)
        write_ordered_chain_pdb(source_pdb, selected_pdb, receptor_chain_ids, ligand_chain_ids)

        plip_trim_cutoff = float(paths.get("plip_trim_cutoff", 20.0) or 0.0)
        count_pdb = selected_pdb
        hotspot_pdb = selected_pdb
        if plip_trim_cutoff > 0:
            # Only trim the PLIP count orientation.  Keep the ligand/design
            # chain(s) complete, and retain only the receptor-peptide residues
            # close to them.  Hotspot orientation deliberately uses the full
            # selected receptor-ligand structure to preserve the original
            # hotspot definition and avoid chain-fragment artifacts.
            count_pdb = os.path.join(tmp_dir, f"{base}_plip_count_interface.pdb")
            write_plip_interface_pdb(
                selected_pdb, count_pdb,
                full_chain_ids=ligand_chain_ids,
                trim_chain_ids=plip_count_chain_ids,
                cutoff=plip_trim_cutoff,
            )

        # 1) Count orientation: selected receptor-peptide chain(s) as PLIP
        #    --peptides; used for total interaction and mainchain-Hbond counts.
        #    Filter PLIP XML by partner chain = ligand_chain_ids, otherwise the
        #    local PDB may also count receptor--receptor contacts among trimmed
        #    nearby chains.
        counts = count_plip_interactions_for_chains(
            plip_path, count_pdb, plip_count_chain_ids, tmp_dir,
            mode=plip_count_mode, timeout=plip_timeout,
            allowed_partner_chain_ids=ligand_chain_ids,
        )

        # 2) Hotspot orientation: ligand/design chain(s) as PLIP --peptides;
        #    used for receptor-hotspot weighted coverage.  Use the full selected
        #    structure and filter reported receptor residues to receptor_chain_ids.
        hotspot_map = extract_hotspot_interactions_for_chains(
            plip_path, hotspot_pdb, ligand_chain_ids, tmp_dir,
            mode=plip_hotspot_mode, timeout=plip_timeout,
            allowed_receptor_chain_ids=receptor_chain_ids,
        )

    num_hbonds = counts["hb_mainchain"] + counts["hb_sidechain"]
    return {
        "num(interaction)": sum_counts(counts),
        "num(H_bonds)": num_hbonds,
        "num(mainchain_Hbonds)": counts["hb_mainchain"],
        "_hotspot_residue_interactions": hotspot_map,
    }

