#!/usr/bin/env python3
import argparse
import csv
import os
import sys

from rank_protocol.common import command_exists, parse_chain_ids, pdb_files
from rank_protocol import constants as rank_constants
from rank_protocol.plip import compute_hotspot_weighted_scores
from rank_protocol.runner import run_tasks
from rank_protocol.vina import find_mgl_python_runner, find_prepare_script

def warn_missing_runtime_tools(paths):
    '''
        Check for the existence of runtime tools and print warnings if they are not found.
    '''
    warnings_to_print = []
    if paths.get("skip_ec", False):
        print("[info] Rosetta EC/APBS is skipped because --skip_ec was set.")

    for label, key in (("AutoDock Vina", "vina"), ("PLIP", "plip")):
        if not command_exists(paths[key]):
            warnings_to_print.append(
                f"{label} command not found on PATH: {paths[key]!r}; related metrics will fail."
            )

    mgl_runner = find_mgl_python_runner(paths.get("mgl_python"))
    if mgl_runner != "direct" and not command_exists(mgl_runner):
        warnings_to_print.append(
            f"MGLTools Python-2 runner not found: {mgl_runner!r}. "
            "Set --mgl_python /path/to/python2.7 or export MGLTOOLS_PYTHON=/path/to/python2.7."
        )
    elif mgl_runner != "direct":
        print(f"[info] MGLTools scripts will be run with: {mgl_runner}")

    for label, key in (("prepare_ligand4.py", "prep_ligand"), ("prepare_receptor4.py", "prep_receptor")):
        resolved_script = find_prepare_script(paths[key])
        if not os.path.exists(resolved_script) and not command_exists(resolved_script):
            warnings_to_print.append(
                f"{label} not found: {paths[key]!r}; Vina PDBQT preparation will fail. "
                "Set --prepare_ligand/--prepare_receptor to the MGLTools Utilities24 script path."
            )
        else:
            print(f"[info] {label} resolved to: {resolved_script}")

    if not paths.get("skip_ec", False) and not command_exists("apbs"):
        warnings_to_print.append(
            "APBS command not found on PATH. Rosetta electrostatic complementarity may fail; "
            "install it with conda-forge::apbs if you need the EC metric."
        )

    for msg in warnings_to_print:
        print(f"[warning] {msg}")

def resolve_rosetta_xml_path(xml_path):
    """Resolve the RosettaScripts XML path with convenient local fallbacks."""
    if xml_path and os.path.exists(xml_path):
        return os.path.abspath(xml_path)

    evaluation_dir = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(evaluation_dir, "RosettaRankingScript.xml")
    if os.path.exists(default):
        return os.path.abspath(default)

    requested = os.path.abspath(xml_path) if xml_path else "<empty>"
    raise FileNotFoundError(
        f"Rosetta XML file not found: {requested}. "
        f"Default fallback is also missing: {default}"
    )


def help_text(show_advanced, text):
    return text if show_advanced else argparse.SUPPRESS


def build_parser(show_advanced=False):
    parser = argparse.ArgumentParser(
        description="Calculate ranking metrics for relaxed PDB files.",
        add_help=False,
    )

    # Major arguments
    major = parser.add_argument_group("Major arguments")
    major.add_argument("-h", "--help", action="store_true", help="show major options and exit")
    major.add_argument("-v", "--help-all", action="store_true",
                       dest="verbose_help", help="show all options and exit")
    major.add_argument("--verbose-help", "--verbose_help", "--help_all", action="store_true",
                       dest="verbose_help", help=argparse.SUPPRESS)
    major.add_argument("-i", "--input", required=True, help="input relaxed PDB directory")
    major.add_argument("-o", "--output", required=True, help="output CSV path")
    major.add_argument(
        "--ligand_chain_ids", "--ligand_chains", "--ligand_chain",
        dest="ligand_chain_ids", required=True,
        help="ligand chain ID(s), comma- or space-separated, e.g. L or B,C",
    )
    major.add_argument(
        "--receptor_chain_ids", "--receptor_chains", "--receptor_chain",
        dest="receptor_chain_ids", required=True,
        help="receptor chain ID(s), comma- or space-separated, e.g. A or A,B",
    )
    major.add_argument("-p", "--num_processors", type=int, default=1)
    major.add_argument("--topk", type=int, default=10, help="top-K hotspot residues")

    # PLIP arguments
    plip_group = parser.add_argument_group("PLIP arguments")
    plip_group.add_argument("--plip_count_mode", choices=["combined", "per_chain"], default=None,
                            help=help_text(show_advanced, "PLIP mode for num(interaction)/mainchain-Hbond counts; default combined for speed"))
    plip_group.add_argument("--plip_hotspot_mode", choices=["combined", "per_chain"], default=None,
                            help=help_text(show_advanced, "PLIP mode for hotspot coverage; default combined for speed"))
    plip_group.add_argument("--plip_count_chain_ids", default=None,
                            help=help_text(show_advanced, "chain ID(s) used as PLIP --peptides for num(interaction)/mainchain-Hbond counts; default = receptor_chain_ids"))
    plip_group.add_argument("--plip_timeout", type=float, default=300.0,
                            help=help_text(show_advanced, "timeout in seconds for each PLIP call; set <=0 to disable timeout"))
    plip_group.add_argument("--plip_trim_cutoff", type=float, default=20.0,
                            help=help_text(show_advanced, "trim only the PLIP count input to receptor-peptide residues within this distance (Angstrom) of ligand/design chain(s); hotspot PLIP uses the full selected structure. Set <=0 to disable count trimming"))

    # Vina arguments
    vina_group = parser.add_argument_group("Vina arguments")
    vina_group.add_argument("--vina_path", default=rank_constants.DEFAULT_VINA,
                            help=help_text(show_advanced, "vina executable name/path; default uses active environment PATH"))
    vina_group.add_argument("--vina_mode", choices=["local_only", "score_only"], default="local_only",
                            help=help_text(show_advanced, "Vina scoring mode; score_only is faster and avoids local minimization"))
    vina_group.add_argument("--vina_box_padding", type=float, default=2.0,
                            help=help_text(show_advanced, "padding in Angstrom around the ligand bounding box"))
    vina_group.add_argument("--vina_box_min_size", type=float, default=18.0,
                            help=help_text(show_advanced, "minimum Vina box size in Angstrom for each axis"))
    vina_group.add_argument("--vina_box_max_size", type=float, default=30.0,
                            help=help_text(show_advanced, "soft maximum Vina box size in Angstrom for each axis; set <=0 to disable the cap"))
    vina_group.add_argument("--vina_box_ligand_margin", type=float, default=0.5,
                            help=help_text(show_advanced, "minimum extra Angstrom margin that keeps the full ligand inside a capped Vina box"))
    vina_group.add_argument("--mgl_python", default=rank_constants.DEFAULT_MGL_PYTHON,
                            help=help_text(show_advanced, "MGLTools Python runner. Default 'auto' uses $MGLTOOLS_PYTHON, then $CONDA_PREFIX/bin/python2.7, then python2.7/pythonsh from PATH. Use direct/none to call prepare scripts directly"))
    vina_group.add_argument("--prepare_ligand", default=rank_constants.DEFAULT_PREP_LIGAND,
                            help=help_text(show_advanced, "prepare_ligand4.py name/path; default uses active environment PATH"))
    vina_group.add_argument("--prepare_receptor", default=rank_constants.DEFAULT_PREP_RECEPTOR,
                            help=help_text(show_advanced, "prepare_receptor4.py name/path; default uses active environment PATH"))
    
    # Rosetta arguments
    rosetta_group = parser.add_argument_group("Rosetta arguments")
    rosetta_group.add_argument("--pyrosetta_flags", default=rank_constants.DEFAULT_PYROSETTA_FLAGS,
                               help=help_text(show_advanced, "flags passed to pyrosetta.init()"))
    rosetta_group.add_argument("--show_rosetta_log", action="store_true",
                               help=help_text(show_advanced, "show PyRosetta/Rosetta stdout and stderr; APBS remains muted unless --show_apbs_log is set"))
    rosetta_group.add_argument("--show_apbs_log", action="store_true",
                               help=help_text(show_advanced, "show APBS stdout/stderr during EC calculation; muted by default"))
    rosetta_group.add_argument("--skip_ec", action="store_true",
                               help=help_text(show_advanced, "skip Rosetta ElectrostaticComplementarityMetric/APBS and leave ec blank"))
    
    # Some path settings
    path_group = parser.add_argument_group("Path settings")
    path_group.add_argument("--rosetta_xml", default=rank_constants.DEFAULT_ROSETTA_XML,
                            help=help_text(show_advanced, "RosettaScripts XML path"))
    path_group.add_argument("--plip_path", default=rank_constants.DEFAULT_PLIP,
                            help=help_text(show_advanced, "plip executable name/path; default uses active environment PATH"))
    path_group.add_argument("--error_output", default=None,
                            help=help_text(show_advanced, "optional TSV file for per-structure metric errors; default is <output>.errors.tsv"))
    return parser


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    wants_major_help = any(arg in {"-h", "--help"} for arg in argv)
    wants_full_help = any(
        arg in {"-v", "--verbose-help", "--verbose_help", "--help-all", "--help_all"}
        for arg in argv
    )

    if wants_major_help or wants_full_help:
        build_parser(show_advanced=wants_full_help).print_help()
        raise SystemExit(0)

    return build_parser(show_advanced=True).parse_args(argv)


def main():
    args = parse_args()

    paths = {
        "vina": args.vina_path,
        "vina_mode": args.vina_mode,
        "vina_box_padding": args.vina_box_padding,
        "vina_box_min_size": args.vina_box_min_size,
        "vina_box_max_size": args.vina_box_max_size,
        "vina_box_ligand_margin": args.vina_box_ligand_margin,
        "mgl_python": args.mgl_python,
        "prep_ligand": args.prepare_ligand,
        "prep_receptor": args.prepare_receptor,
        "pyrosetta_flags": args.pyrosetta_flags,
        "quiet_rosetta": not args.show_rosetta_log,
        "quiet_apbs": not args.show_apbs_log,
        "skip_ec": args.skip_ec,
        "rosetta_xml": resolve_rosetta_xml_path(args.rosetta_xml),
        "plip": args.plip_path,
        "plip_count_mode": args.plip_count_mode or "combined",
        "plip_hotspot_mode": args.plip_hotspot_mode or "combined",
        "plip_count_chain_ids": args.plip_count_chain_ids,
        "plip_timeout": None if args.plip_timeout is not None and args.plip_timeout <= 0 else args.plip_timeout,
        "plip_trim_cutoff": args.plip_trim_cutoff,
    }

    warn_missing_runtime_tools(paths)

    receptor_chain_ids = parse_chain_ids(args.receptor_chain_ids, "--receptor_chain_ids")
    ligand_chain_ids = parse_chain_ids(args.ligand_chain_ids, "--ligand_chain_ids")

    files = pdb_files(args.input)
    tasks = [(path, receptor_chain_ids, ligand_chain_ids, paths) for path in files]
    results = run_tasks(tasks, max(1, args.num_processors))
    hotspot_scores = compute_hotspot_weighted_scores(results, args.topk)
    for row in results:
        row["hotspot_occupoed weighted"] = hotspot_scores.get(row["_input_path"], 0.0)

    columns = rank_constants.OUTPUT_COLUMNS
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({column: row.get(column, "") for column in columns})

    failed = [row for row in results if row.get("_errors")]
    if failed:
        print(f"completed with metric errors for {len(failed)}/{len(results)} files")
        for row in failed[:20]:
            print(f"{row['pdb_name']}: {'; '.join(row['_errors'])}")

        error_output = args.error_output or (args.output + ".errors.tsv")
        with open(error_output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rank_constants.ERROR_COLUMNS, delimiter="\t")
            writer.writeheader()
            for row in failed:
                writer.writerow({
                    "pdb_name": row.get("pdb_name", ""),
                    "input_path": row.get("_input_path", ""),
                    "errors": "; ".join(row.get("_errors", [])),
                })
        print(f"wrote metric error report {error_output}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
