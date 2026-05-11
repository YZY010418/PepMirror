import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET

from .common import (
    chain_ids_text,
    copy_or_unzip_pdb,
    make_pickle_safe,
    parse_chain_ids,
    pdb_name,
    suppress_stdout_stderr,
    to_python_scalar,
    write_ordered_chain_pdb,
)


# PyRosetta must be initialized once per Python process. With ProcessPoolExecutor,
# each worker process will initialize itself on its first Rosetta calculation.
_PYROSETTA_INITIALIZED = False


def patch_rosetta_xml(xml_path, output_xml_path, receptor_chain_ids, ligand_chain_ids, ddg_repeats=1):
    """Patch RosettaScripts XML for chain IDs and the requested DDG repeat count."""
    receptor_chain_ids = parse_chain_ids(receptor_chain_ids, "receptor chain IDs")
    ligand_chain_ids = parse_chain_ids(ligand_chain_ids, "ligand chain IDs")
    ddg_repeats = max(1, int(ddg_repeats))

    tree = ET.parse(xml_path)
    root = tree.getroot()
    found_receptor = False
    found_ligand = False

    for node in root.findall(".//Chain"):
        name = node.attrib.get("name")
        if name == "ReceptorChain":
            node.set("chains", chain_ids_text(receptor_chain_ids))
            found_receptor = True
        elif name == "LigandChain":
            node.set("chains", chain_ids_text(ligand_chain_ids))
            found_ligand = True

    if not found_receptor or not found_ligand:
        raise ValueError(
            "Rosetta XML must contain Chain selectors named "
            "'ReceptorChain' and 'LigandChain'"
        )

    for node in root.findall(".//Ddg"):
        if node.attrib.get("name") == "ddg":
            node.set("repeats", str(ddg_repeats))

    tree.write(output_xml_path, encoding="unicode")


def flatten_json(obj, out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                flatten_json(value, out)
            else:
                out[key] = value
    elif isinstance(obj, list):
        for value in obj:
            flatten_json(value, out)
    return out


def read_rosetta_scorefile(scorefile):
    records = []
    if not os.path.exists(scorefile):
        return {}
    with open(scorefile) as handle:
        text = handle.read().strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        records.append(flatten_json(parsed))
    except Exception:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(flatten_json(json.loads(line)))
            except Exception:
                continue

    if not records:
        return {}
    return records[-1]


def extract_avg_metric(value):
    """Return avg from Rosetta map-like SimpleMetric output.

    ElectrostaticComplementarityMetric commonly returns a C++ map printed like
    map_std_string_double{avg: 0.742768, p: 0.803099, s: 0.682438}. For the
    CSV we keep only avg.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return float(value)

    try:
        for getter_name in ("get", "at"):
            getter = getattr(value, getter_name, None)
            if getter is not None:
                try:
                    return float(getter("avg"))
                except Exception:
                    pass
        try:
            return float(value["avg"])
        except Exception:
            pass
        try:
            for k, v in value.items():
                if str(k) == "avg":
                    return float(v)
        except Exception:
            pass
    except Exception:
        pass

    text = str(value)
    match = re.search(r"(?:^|[,{\s])avg\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
    if match:
        return float(match.group(1))
    try:
        return float(text)
    except Exception:
        return text


def metric_value(record, names):
    for name in names:
        if name in record:
            return to_python_scalar(record[name])
    for key, value in record.items():
        for name in names:
            if str(key).endswith(name):
                return to_python_scalar(value)
    return ""


def init_pyrosetta_once(flags, quiet=True):
    """Initialize PyRosetta once in the current process."""
    global _PYROSETTA_INITIALIZED
    if _PYROSETTA_INITIALIZED:
        return

    with suppress_stdout_stderr(quiet):
        import pyrosetta

        pyrosetta.init(flags)
    _PYROSETTA_INITIALIZED = True


def flatten_mapping(obj, out=None):
    """Convert nested PyRosetta score/cache mappings into a flat dict."""
    if out is None:
        out = {}
    try:
        items = obj.items()
    except Exception:
        return out
    for key, value in items:
        key = str(key)
        if hasattr(value, "items"):
            flatten_mapping(value, out)
        else:
            out[key] = value
    return out


def collect_pose_scores(pose):
    """Collect only pickle-safe primitive scores from pose caches.

    Avoid rosetta.core.pose.getPoseExtraScore() as a fallback: some PyRosetta
    builds call Rosetta's C++ assertion handler when a key is absent, which
    cannot be caught reliably by Python try/except.
    """
    record = {}

    for attr in ("scores", "cache"):
        try:
            cache = getattr(pose, attr)
            temp = {}
            flatten_mapping(cache, temp)
            for k, v in temp.items():
                if isinstance(v, (str, int, float, bool)):
                    record[str(k)] = v
                else:
                    try:
                        record[str(k)] = float(v)
                    except Exception:
                        # Skip PyRosetta std::map and other C++ wrapper objects.
                        pass
        except Exception:
            pass

    return record


def add_filter_value(record, xml_objects, pose, filter_name):
    """Compute a RosettaScripts filter value if it is not already cached."""
    if filter_name in record and record[filter_name] != "":
        return
    try:
        record[filter_name] = to_python_scalar(xml_objects.get_filter(filter_name).report_sm(pose))
    except Exception as exc:
        record[filter_name] = ""
        record[f"_{filter_name}_error"] = str(exc).splitlines()[0] if str(exc) else "failed"


def add_simple_metric_value(record, xml_objects, pose, metric_name):
    """Compute a RosettaScripts SimpleMetric value if it is not already cached."""
    if metric_name in record and record[metric_name] != "":
        return
    try:
        raw_value = xml_objects.get_simple_metric(metric_name).calculate(pose)
        if metric_name == "ec_metric":
            record[metric_name] = extract_avg_metric(raw_value)
        else:
            record[metric_name] = to_python_scalar(raw_value)
    except Exception as exc:
        record[metric_name] = ""
        record[f"_{metric_name}_error"] = str(exc).splitlines()[0] if str(exc) else "failed"


def collect_metric_errors(record):
    errors = []
    for key, value in record.items():
        if key.startswith("_") and key.endswith("_error") and value:
            metric = key[1:-6]
            errors.append(f"{metric}: {value}")
    return errors


def setup_interface_foldtree_by_chains(pose, receptor_chain_ids, ligand_chain_ids):
    """Define jump 1 as receptor-chain group versus ligand-chain group.

    Rosetta filters such as Ddg(jump=1) and BuriedUnsatHbonds2(jump_number=1)
    are jump-based, not selector-based. For multichain systems, explicitly set
    the FoldTree so jump 1 corresponds to the user-specified chain groups, e.g.
    receptor A,C and ligand B -> partners string "AC_B".
    """
    receptor_chain_ids = parse_chain_ids(receptor_chain_ids, "receptor chain IDs")
    ligand_chain_ids = parse_chain_ids(ligand_chain_ids, "ligand chain IDs")
    partners = "".join(receptor_chain_ids) + "_" + "".join(ligand_chain_ids)

    try:
        from pyrosetta.rosetta.protocols.docking import setup_foldtree
        from pyrosetta.rosetta.utility import vector1_int

        movable_jumps = vector1_int()
        movable_jumps.append(1)
        setup_foldtree(pose, partners, movable_jumps)
    except Exception as exc:
        raise RuntimeError(f"failed to set Rosetta FoldTree for partners {partners!r}: {exc}")
    return partners


def calculate_rosetta_metrics(pdb_path, receptor_chain_ids, ligand_chain_ids, paths):
    receptor_chain_ids = parse_chain_ids(receptor_chain_ids, "receptor chain IDs")
    ligand_chain_ids = parse_chain_ids(ligand_chain_ids, "ligand chain IDs")
    quiet = paths.get("quiet_rosetta", True)
    init_pyrosetta_once(paths["pyrosetta_flags"], quiet=quiet)

    with suppress_stdout_stderr(quiet):
        import pyrosetta
        from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

        with tempfile.TemporaryDirectory(prefix="rank_pyrosetta_") as tmp_dir:
            base = pdb_name(pdb_path)
            local_pdb = os.path.join(tmp_dir, f"{base}_selected.pdb")
            source_pdb = os.path.join(tmp_dir, f"{base}.pdb")
            local_xml = os.path.join(tmp_dir, "rosetta_chainids.xml")
            copy_or_unzip_pdb(pdb_path, source_pdb)

            # Keep only the user-selected receptor and ligand chains, but preserve
            # their original PDB chain IDs. The XML is patched accordingly below.
            write_ordered_chain_pdb(source_pdb, local_pdb, receptor_chain_ids, ligand_chain_ids)
            patch_rosetta_xml(
                paths["rosetta_xml"], local_xml, receptor_chain_ids, ligand_chain_ids,
                ddg_repeats=paths.get("rosetta_ddg_repeats", 1),
            )

            pose = pyrosetta.pose_from_pdb(local_pdb)
            setup_interface_foldtree_by_chains(pose, receptor_chain_ids, ligand_chain_ids)

            # Parse the XML but do not run the full ParsedProtocol. We only need the
            # five reported metrics, and explicit calculation avoids unsafe access to
            # pose extra scores as well as non-pickleable PyRosetta cache objects.
            xml_objects = XmlObjects.create_from_file(local_xml, pose)

            record = {}
            filter_names = ["sc_metric", "buried_Hbonds"]
            if paths.get("with_rosetta_ddg", False):
                filter_names.extend(["ddg", "ddg_norepack"])
            for filter_name in filter_names:
                add_filter_value(record, xml_objects, pose, filter_name)
            if not paths.get("skip_ec", False):
                with suppress_stdout_stderr(paths.get("quiet_apbs", True)):
                    add_simple_metric_value(record, xml_objects, pose, "ec_metric")
            else:
                record["ec_metric"] = ""

            result = {
                "sc": metric_value(record, ["sc_metric"]),
                "ec": extract_avg_metric(metric_value(record, ["ec_metric"])),
                "buried_Hbonds": metric_value(record, ["buried_Hbonds"]),
            }
            metric_errors = collect_metric_errors(record)
            if metric_errors:
                result["_rosetta_metric_errors"] = metric_errors
            if paths.get("with_rosetta_ddg", False):
                result.update({
                    "rosetta_ddg": metric_value(record, ["ddg"]),
                    "rosetta_ddg_norepack": metric_value(record, ["ddg_norepack"]),
                })
            return make_pickle_safe(result)
