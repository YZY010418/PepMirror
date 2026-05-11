import contextlib
import gzip
import os
import re
import shutil
import sys


def pdb_name(path):
    name = os.path.basename(path)
    if name.endswith(".pdb.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def progress_iter(iterable, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


@contextlib.contextmanager
def suppress_stdout_stderr(enabled=True):
    """Temporarily silence Python and C/C++ stdout/stderr in the current process."""
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)


def pdb_files(input_dir):
    files = []
    for root, _, names in os.walk(input_dir):
        for name in sorted(names):
            if name.lower().endswith((".pdb", ".pdb.gz")):
                files.append(os.path.join(root, name))
    return files


def copy_or_unzip_pdb(pdb_path, out_path):
    if pdb_path.endswith(".gz"):
        with gzip.open(pdb_path, "rb") as fin, open(out_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    else:
        shutil.copy2(pdb_path, out_path)


def load_biopython():
    from Bio import PDB
    return PDB


def parse_chain_ids(value, arg_name="chain IDs"):
    """Parse one or multiple PDB chain IDs from CLI strings.

    Accepted forms:
      --receptor_chain_ids A
      --receptor_chain_ids A,B
      --receptor_chain_ids "A B"
    """
    if isinstance(value, (list, tuple)):
        raw = []
        for item in value:
            raw.extend(parse_chain_ids(item, arg_name))
        parts = raw
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{arg_name} is empty")
        parts = [x for x in re.split(r"[,;\s]+", text) if x]

    seen = set()
    chain_ids = []
    for chain_id in parts:
        if len(chain_id) != 1:
            raise ValueError(
                f"Invalid chain ID {chain_id!r} in {arg_name}; "
                "PDB chain IDs must be single characters. Use comma-separated IDs, e.g. A,B."
            )
        if chain_id not in seen:
            seen.add(chain_id)
            chain_ids.append(chain_id)
    if not chain_ids:
        raise ValueError(f"{arg_name} is empty")
    return chain_ids


def chain_ids_text(chain_ids):
    return ",".join(chain_ids)


def resolve_command(command):
    """Return an executable path if available on PATH; otherwise return command.

    Absolute paths are preserved. For environment-based installs, command names
    such as "vina", "plip", "pythonsh", and "prepare_ligand4.py" are resolved
    from the active conda environment.
    """
    if not command:
        return command
    if os.path.isabs(command) or os.path.sep in command:
        return command
    return shutil.which(command) or command


def command_exists(command):
    if not command:
        return False
    if os.path.isabs(command) or os.path.sep in command:
        return os.path.exists(command)
    return shutil.which(command) is not None


def get_conda_prefix():
    """Return the active conda prefix if available."""
    return os.environ.get("CONDA_PREFIX", "").strip()


def line_chain_id(line):
    return line[21].strip()


def get_pdb_chain_ids(pdb_path):
    chains = []
    seen = set()
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain_id = line_chain_id(line)
            if chain_id not in seen:
                seen.add(chain_id)
                chains.append(chain_id)
    return chains


def require_chains_in_pdb(pdb_path, chain_ids, context="input PDB"):
    available = set(get_pdb_chain_ids(pdb_path))
    missing = [chain_id for chain_id in chain_ids if chain_id not in available]
    if missing:
        raise ValueError(
            f"{context}: missing chain(s) {chain_ids_text(missing)}; "
            f"available chain(s): {chain_ids_text(sorted(available)) or 'None'}"
        )


def pdb_residue_key(line):
    """Return a residue key preserving PDB chain/residue identity."""
    return (line_chain_id(line), line[22:26], line[26:27])


def pdb_atom_coord(line):
    """Parse XYZ coordinates from a PDB ATOM/HETATM line."""
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def within_cutoff(coord, ref_coords, cutoff_sq):
    x, y, z = coord
    for rx, ry, rz in ref_coords:
        dx = x - rx
        dy = y - ry
        dz = z - rz
        if dx * dx + dy * dy + dz * dz <= cutoff_sq:
            return True
    return False


def filter_existing_chain_ids_in_pdb(pdb_path, chain_ids):
    """Return chain IDs that are present in a PDB file, preserving input order."""
    requested = parse_chain_ids(chain_ids, "chain IDs")
    available = set(get_pdb_chain_ids(pdb_path))
    return [chain_id for chain_id in requested if chain_id in available]


def write_ordered_chain_pdb(input_pdb, output_pdb, receptor_chain_ids, ligand_chain_ids):
    """Write selected receptor and ligand chains, in the user-specified order.

    The original PDB chain IDs are preserved, so RosettaScripts Chain selectors can
    use the explicit chain IDs passed in the command line.
    """
    receptor_chain_ids = parse_chain_ids(receptor_chain_ids, "receptor chain IDs")
    ligand_chain_ids = parse_chain_ids(ligand_chain_ids, "ligand chain IDs")
    chain_order = receptor_chain_ids + ligand_chain_ids
    overlap = set(receptor_chain_ids).intersection(ligand_chain_ids)
    if overlap:
        raise ValueError(f"chain(s) cannot be both receptor and ligand: {chain_ids_text(sorted(overlap))}")

    with open(input_pdb) as fin:
        lines = [line for line in fin if line.startswith(("ATOM", "HETATM"))]

    available = {line_chain_id(line) for line in lines}
    missing = [chain_id for chain_id in chain_order if chain_id not in available]
    if missing:
        raise ValueError(
            f"missing chain(s) {chain_ids_text(missing)} in {input_pdb}; "
            f"available chain(s): {chain_ids_text(sorted(available)) or 'None'}"
        )

    with open(output_pdb, "w") as fout:
        serial = 1
        for chain_id in chain_order:
            wrote = False
            last_atom = None
            for line in lines:
                if line_chain_id(line) != chain_id:
                    continue
                fout.write(line[:6] + f"{serial:5d}" + line[11:])
                serial += 1
                wrote = True
                last_atom = line
            if wrote:
                fout.write(f"TER   {serial:5d}      {last_atom[17:20]} {chain_id:1s}{last_atom[22:27]}\n")
                serial += 1
        fout.write("END\n")


def to_python_scalar(value):
    """Convert PyRosetta/C++ wrapped values to pickle-safe Python scalars."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return float(value)
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        pass
    # Do not return PyRosetta std::map or other wrapped objects to the parent
    # process. They are not pickleable under ProcessPoolExecutor.
    return str(value)


def make_pickle_safe(obj):
    """Recursively remove non-pickleable PyRosetta wrapper objects."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_pickle_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_pickle_safe(v) for v in obj]
    return to_python_scalar(obj)

