import os
import re
import shutil
import subprocess
import tempfile
import time

from .common import (
    copy_or_unzip_pdb,
    get_conda_prefix,
    load_biopython,
    parse_chain_ids,
    pdb_name,
    resolve_command,
)


def find_prepare_script(script_name):
    """Resolve MGLTools prepare_*.py scripts robustly.

    Bioconda/legacy MGLTools installs do not always put prepare_ligand4.py
    and prepare_receptor4.py on PATH. In addition to PATH, search common
    locations under the active conda prefix.
    """
    if not script_name:
        return script_name
    if os.path.isabs(script_name) or os.path.sep in script_name:
        return script_name

    resolved = shutil.which(script_name)
    if resolved:
        return resolved

    conda_prefix = get_conda_prefix()
    if conda_prefix:
        candidates = [
            os.path.join(conda_prefix, "bin", script_name),
            os.path.join(conda_prefix, "MGLToolsPckgs", "AutoDockTools", "Utilities24", script_name),
            os.path.join(conda_prefix, "share", "mgltools", "MGLToolsPckgs", "AutoDockTools", "Utilities24", script_name),
            os.path.join(conda_prefix, "lib", "python2.7", "site-packages", "AutoDockTools", "Utilities24", script_name),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

    return script_name


def find_mgl_python_runner(python_runner="auto"):
    """Resolve the Python executable used for MGLTools scripts.

    MGLTools from conda/bioconda is Python-2 based. In environments where
    $CONDA_PREFIX/bin/python has been relinked to Python 3 for PyRosetta,
    directly executing prepare_ligand4.py may use the wrong interpreter.

    Resolution order for --mgl_python auto:
      1. $MGLTOOLS_PYTHON, if set
      2. $CONDA_PREFIX/bin/python2.7, if present
      3. python2.7 on PATH
      4. pythonsh on PATH
      5. pythonsh literal fallback
    """
    runner = (python_runner or "").strip()
    if runner.lower() in {"", "auto"}:
        env_runner = os.environ.get("MGLTOOLS_PYTHON", "").strip()
        if env_runner:
            return env_runner

        conda_prefix = get_conda_prefix()
        if conda_prefix:
            candidate = os.path.join(conda_prefix, "bin", "python2.7")
            if os.path.exists(candidate):
                return candidate

        for candidate in ("python2.7", "pythonsh"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return "pythonsh"

    if runner.lower() in {"none", "direct", "false", "0"}:
        return "direct"
    return resolve_command(runner)


def build_prepare_cmd(python_runner, prepare_script, *args):
    """Build a PDBQT preparation command.

    Default behavior is --mgl_python auto, which explicitly uses the Python-2.7
    interpreter from the active conda environment when it exists. This avoids
    accidental use of $CONDA_PREFIX/bin/python after it has been relinked to
    Python 3 for PyRosetta.
    """
    script = find_prepare_script(prepare_script)
    runner = find_mgl_python_runner(python_runner)
    if runner == "direct":
        return [script, *args]
    return [runner, script, *args]


def is_valid_pdbqt(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    with open(path) as handle:
        first = handle.readline()
    return any(first.startswith(tag) for tag in ("REMARK", "ROOT", "ATOM", "HETATM", "MODEL"))


def pdbqt_record_tag(line):
    """Return the PDBQT record tag used by Vina's parser.

    PDBQT ligand records include tags longer than six characters, such as
    ENDROOT, ENDBRANCH, and TORSDOF. Therefore we must parse the first
    whitespace-delimited token rather than line[:6].
    """
    text = str(line or "").strip()
    return text.split(None, 1)[0] if text else ""


def sanitize_pdbqt_file(path, mode):
    """Remove PDBQT records commonly emitted by MGLTools but rejected by Vina.

    Vina is stricter for ligand PDBQT than receptor PDBQT. In particular,
    top-level ATOM/HETATM records in a ligand file can trigger "Unknown or
    inappropriate tag found in flex residue or ligand" unless they are wrapped
    in a ROOT/ENDROOT torsion tree. This function only removes records; the
    torsion-tree wrapping is handled by ensure_ligand_pdbqt_torsion_tree().
    """
    if not path or not os.path.exists(path):
        return []

    mode = str(mode).lower()
    if mode == "ligand":
        allowed = {
            "REMARK", "ROOT", "ENDROOT", "BRANCH", "ENDBRANCH",
            "TORSDOF", "ATOM", "HETATM",
        }
    else:
        allowed = {"REMARK", "ATOM", "HETATM"}

    kept = []
    removed = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            tag = pdbqt_record_tag(line)
            if tag in allowed:
                kept.append(line)
            else:
                removed.append(tag or line.strip().split()[0])

    if kept:
        with open(path, "w") as handle:
            handle.writelines(kept)

    seen = set()
    unique = []
    for tag in removed:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def ensure_ligand_pdbqt_torsion_tree(path):
    """Wrap receptor-style ligand PDBQT atoms into a rigid ROOT tree.

    prepare_ligand4.py sometimes produces peptide-like ligand PDBQT files with
    only ATOM/HETATM records, or the fallback prepare_receptor4.py produces a
    receptor-style PDBQT. Vina parses ligand files as torsion trees, so this
    converts such rigid ligand files into:
        REMARK ...
        ROOT
        ATOM/HETATM ...
        ENDROOT
        TORSDOF 0
    Returns True if the file was rewritten.
    """
    if not path or not os.path.exists(path):
        return False

    with open(path) as handle:
        lines = [line for line in handle if line.strip()]
    tags = [pdbqt_record_tag(line) for line in lines]
    if "ROOT" in tags:
        return False

    remark_lines = [line for line in lines if pdbqt_record_tag(line) == "REMARK"]
    atom_lines = [line for line in lines if pdbqt_record_tag(line) in {"ATOM", "HETATM"}]
    if not atom_lines:
        return False

    with open(path, "w") as handle:
        handle.writelines(remark_lines)
        handle.write("ROOT\n")
        handle.writelines(atom_lines)
        handle.write("ENDROOT\n")
        handle.write("TORSDOF 0\n")
    return True


def summarize_pdbqt_tags(path, mode):
    """Return a compact tag summary to help diagnose remaining Vina failures."""
    if not path or not os.path.exists(path):
        return "missing"
    tags = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                tags.append(pdbqt_record_tag(line))
    if not tags:
        return "empty"
    counts = {}
    for tag in tags:
        counts[tag] = counts.get(tag, 0) + 1
    return ",".join(f"{k}:{counts[k]}" for k in sorted(counts))


def first_nonempty_line(text):
    for line in str(text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def try_run_mgl_tool(cmd, output_path):
    """Run an MGLTools preparation command and return (ok, diagnostic)."""
    last_error = ""
    for _ in range(2):
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            last_error = result.stderr.strip() or result.stdout.strip()
        except FileNotFoundError as exc:
            last_error = str(exc)
            break
        if is_valid_pdbqt(output_path):
            return True, ""
        time.sleep(0.5)
    return False, first_nonempty_line(last_error) or "no valid pdbqt was produced"


def run_mgl_tool(cmd, output_path):
    ok, error = try_run_mgl_tool(cmd, output_path)
    if not ok:
        raise RuntimeError(error)
    return True


def compute_vina_box_from_chains(chains, padding=8.0, min_size=18.0):
    coords = [atom.get_coord() for chain in chains for atom in chain.get_atoms()]
    if not coords:
        raise ValueError("ligand chain(s) have no atoms")
    mins = [min(coord[i] for coord in coords) for i in range(3)]
    maxs = [max(coord[i] for coord in coords) for i in range(3)]
    center = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
    size = [max((maxs[i] - mins[i]) + 2.0 * padding, min_size) for i in range(3)]
    return center + size


def get_chains(model, chain_ids):
    chain_ids = parse_chain_ids(chain_ids, "chain IDs")
    chain_map = {chain.id: chain for chain in model.get_chains()}
    missing = [chain_id for chain_id in chain_ids if chain_id not in chain_map]
    if missing:
        raise ValueError(
            f"missing chain(s) {chain_ids_text(missing)}; "
            f"available chain(s): {chain_ids_text(sorted(chain_map)) or 'None'}"
        )
    return [chain_map[chain_id] for chain_id in chain_ids]


def save_chains(chains, path, name):
    PDB = load_biopython()
    structure = PDB.Structure.Structure(name)
    model = PDB.Model.Model(0)
    structure.add(model)
    for chain in chains:
        model.add(chain.copy())
    writer = PDB.PDBIO()
    writer.set_structure(structure)
    writer.save(path)


def calculate_vina_score(pdb_path, receptor_chain_ids, ligand_chain_ids, paths):
    receptor_chain_ids = parse_chain_ids(receptor_chain_ids, "receptor chain IDs")
    ligand_chain_ids = parse_chain_ids(ligand_chain_ids, "ligand chain IDs")

    PDB = load_biopython()
    with tempfile.TemporaryDirectory(prefix="rank_vina_") as tmp_dir:
        base = pdb_name(pdb_path)
        local_pdb = os.path.join(tmp_dir, f"{base}.pdb")
        copy_or_unzip_pdb(pdb_path, local_pdb)

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(base, local_pdb)
        model = next(structure.get_models())
        rec_chains = get_chains(model, receptor_chain_ids)
        lig_chains = get_chains(model, ligand_chain_ids)

        rec_pdb = os.path.join(tmp_dir, f"{base}_R.pdb")
        lig_pdb = os.path.join(tmp_dir, f"{base}_L.pdb")
        rec_pdbqt = os.path.join(tmp_dir, f"{base}_R.pdbqt")
        lig_pdbqt = os.path.join(tmp_dir, f"{base}_L.pdbqt")
        save_chains(rec_chains, rec_pdb, base)
        save_chains(lig_chains, lig_pdb, base)

        ok_rec, rec_err = try_run_mgl_tool(
            build_prepare_cmd(paths["mgl_python"], paths["prep_receptor"], "-r", rec_pdb, "-o", rec_pdbqt, "-A", "checkhydrogens"),
            rec_pdbqt,
        )
        if not ok_rec:
            raise RuntimeError(f"prepare_receptor failed: {rec_err}")

        ok_lig, lig_err = try_run_mgl_tool(
            build_prepare_cmd(paths["mgl_python"], paths["prep_ligand"], "-l", lig_pdb, "-o", lig_pdbqt, "-A", "checkhydrogens"),
            lig_pdbqt,
        )
        if not ok_lig:
            ok_lig_fallback, lig_fallback_err = try_run_mgl_tool(
                build_prepare_cmd(paths["mgl_python"], paths["prep_receptor"], "-r", lig_pdb, "-o", lig_pdbqt, "-A", "checkhydrogens"),
                lig_pdbqt,
            )
            if not ok_lig_fallback:
                raise RuntimeError(
                    "prepare_ligand failed: "
                    f"prepare_ligand4.py: {lig_err}; fallback prepare_receptor4.py: {lig_fallback_err}"
                )

        rec_removed_tags = sanitize_pdbqt_file(rec_pdbqt, "receptor")
        lig_removed_tags = sanitize_pdbqt_file(lig_pdbqt, "ligand")
        lig_wrapped_root = ensure_ligand_pdbqt_torsion_tree(lig_pdbqt)

        cx, cy, cz, sx, sy, sz = compute_vina_box_from_chains(lig_chains)
        vina_mode = paths.get("vina_mode", "local_only")
        if vina_mode not in {"local_only", "score_only"}:
            raise ValueError(f"unsupported Vina mode: {vina_mode}")
        cmd = [
            resolve_command(paths["vina"]),
            "--receptor", rec_pdbqt,
            "--ligand", lig_pdbqt,
            "--cpu", "1",
            f"--{vina_mode}",
            "--center_x", f"{cx:.3f}",
            "--center_y", f"{cy:.3f}",
            "--center_z", f"{cz:.3f}",
            "--size_x", f"{sx:.3f}",
            "--size_y", f"{sy:.3f}",
            "--size_z", f"{sz:.3f}",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 and vina_mode == "local_only":
            # If local_only fails for a rigid peptide-like ligand, try score_only
            # before reporting the failure. Both modes use the same PDBQT files.
            score_only_cmd = [x for x in cmd if x != "--local_only"]
            score_only_cmd.insert(score_only_cmd.index("--center_x"), "--score_only")
            score_result = subprocess.run(score_only_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if score_result.returncode == 0:
                result = score_result
            else:
                message = (result.stderr.strip().splitlines() or result.stdout.strip().splitlines() or ["vina failed"])[0]
                score_message = (score_result.stderr.strip().splitlines() or score_result.stdout.strip().splitlines() or [""])[0]
                diag = []
                if rec_removed_tags:
                    diag.append("removed receptor PDBQT tags=" + ",".join(rec_removed_tags))
                if lig_removed_tags:
                    diag.append("removed ligand PDBQT tags=" + ",".join(lig_removed_tags))
                if lig_wrapped_root:
                    diag.append("ligand PDBQT wrapped as ROOT/ENDROOT/TORSDOF 0")
                if score_message:
                    diag.append("score_only also failed=" + score_message)
                diag.append("receptor tags=" + summarize_pdbqt_tags(rec_pdbqt, "receptor"))
                diag.append("ligand tags=" + summarize_pdbqt_tags(lig_pdbqt, "ligand"))
                raise RuntimeError(message + " | " + "; ".join(diag))
        elif result.returncode != 0:
            message = (result.stderr.strip().splitlines() or result.stdout.strip().splitlines() or ["vina failed"])[0]
            diag = []
            if rec_removed_tags:
                diag.append("removed receptor PDBQT tags=" + ",".join(rec_removed_tags))
            if lig_removed_tags:
                diag.append("removed ligand PDBQT tags=" + ",".join(lig_removed_tags))
            if lig_wrapped_root:
                diag.append("ligand PDBQT wrapped as ROOT/ENDROOT/TORSDOF 0")
            diag.append("receptor tags=" + summarize_pdbqt_tags(rec_pdbqt, "receptor"))
            diag.append("ligand tags=" + summarize_pdbqt_tags(lig_pdbqt, "ligand"))
            raise RuntimeError(message + " | " + "; ".join(diag))

        vina_patterns = [
            r"Affinity:\s+([-+]?\d+(?:\.\d+)?)",
            r"Estimated Free Energy of Binding\s*:\s*([-+]?\d+(?:\.\d+)?)",
        ]
        for pattern in vina_patterns:
            match = re.search(pattern, result.stdout)
            if match:
                return float(match.group(1))
        for line in result.stdout.splitlines():
            if re.match(r"^\s*1\s+[-+]?\d", line):
                return float(line.split()[1])
        raise RuntimeError("vina score not found; first stdout line: " + (first_nonempty_line(result.stdout) or "<empty>"))
