#!/usr/bin/env python3
import argparse
import csv
import os

if __package__:
    from .rank_protocol.common import command_exists, parse_chain_ids, pdb_files
    from .rank_protocol.constants import (
        DEFAULT_MGL_PYTHON,
        DEFAULT_PLIP,
        DEFAULT_PREP_LIGAND,
        DEFAULT_PREP_RECEPTOR,
        DEFAULT_PYROSETTA_FLAGS,
        DEFAULT_ROSETTA_XML,
        DEFAULT_VINA,
        ERROR_COLUMNS,
        OUTPUT_COLUMNS,
        ROSETTA_DDG_COLUMNS,
    )
    from .rank_protocol.plip import compute_hotspot_weighted_scores
    from .rank_protocol.runner import run_tasks
    from .rank_protocol.vina import find_mgl_python_runner, find_prepare_script
else:
    from rank_protocol.common import command_exists, parse_chain_ids, pdb_files
    from rank_protocol.constants import (
        DEFAULT_MGL_PYTHON,
        DEFAULT_PLIP,
        DEFAULT_PREP_LIGAND,
        DEFAULT_PREP_RECEPTOR,
        DEFAULT_PYROSETTA_FLAGS,
        DEFAULT_ROSETTA_XML,
        DEFAULT_VINA,
        ERROR_COLUMNS,
        OUTPUT_COLUMNS,
        ROSETTA_DDG_COLUMNS,
    )
    from rank_protocol.plip import compute_hotspot_weighted_scores
    from rank_protocol.runner import run_tasks
    from rank_protocol.vina import find_mgl_python_runner, find_prepare_script


def warn_missing_runtime_tools(paths):
    warnings_to_print = []
    if paths.get("skip_ec", False):
        print("[info] Rosetta EC/APBS is skipped; use --with_ec to fill the ec column.")

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
    candidates = [
        os.path.join(evaluation_dir, "RosettaRankingScript.xml")
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return os.path.abspath(xml_path)


def output_columns(include_rosetta_ddg=False):
    columns = list(OUTPUT_COLUMNS)
    if include_rosetta_ddg:
        insert_at = columns.index("vina score")
        columns[insert_at:insert_at] = ROSETTA_DDG_COLUMNS
    return columns


def main():
    parser = argparse.ArgumentParser(description="Calculate ranking metrics for relaxed PDB files.")
    parser.add_argument("-i", "--input", required=True, help="input relaxed PDB directory")
    parser.add_argument("-o", "--output", required=True, help="output CSV path")
    parser.add_argument(
        "--ligand_chain_ids", "--ligand_chains", "--ligand_chain",
        dest="ligand_chain_ids", required=True,
        help="ligand chain ID(s), comma- or space-separated, e.g. L or B,C",
    )
    parser.add_argument(
        "--receptor_chain_ids", "--receptor_chains", "--receptor_chain",
        dest="receptor_chain_ids", required=True,
        help="receptor chain ID(s), comma- or space-separated, e.g. A or A,B",
    )
    parser.add_argument("-p", "--num_processors", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10, help="top-K hotspot residues")
    parser.add_argument("--plip_mode", choices=["combined", "per_chain"], default=None,
                        help="legacy shortcut: set both --plip_count_mode and --plip_hotspot_mode")
    parser.add_argument("--plip_count_mode", choices=["combined", "per_chain"], default=None,
                        help="PLIP mode for num(interaction)/mainchain-Hbond counts; default combined for speed")
    parser.add_argument("--plip_hotspot_mode", choices=["combined", "per_chain"], default=None,
                        help="PLIP mode for hotspot coverage; default combined for speed")
    parser.add_argument("--plip_count_chain_ids", "--plip_count_chains", "--plip_backbone_chain_ids",
                        dest="plip_count_chain_ids", default=None,
                        help="chain ID(s) used as PLIP --peptides for num(interaction)/mainchain-Hbond counts; default = receptor_chain_ids")
    parser.add_argument("--plip_timeout", type=float, default=300.0,
                        help="timeout in seconds for each PLIP call; set <=0 to disable timeout")
    parser.add_argument("--plip_trim_cutoff", type=float, default=20.0,
                        help="trim only the PLIP count input to receptor-peptide residues within this distance (Angstrom) of ligand/design chain(s); hotspot PLIP uses the full selected structure. Set <=0 to disable count trimming")
    parser.add_argument("--vina_path", default=DEFAULT_VINA, help="vina executable name/path; default uses active environment PATH")
    parser.add_argument("--vina_mode", choices=["local_only", "score_only"], default="local_only",
                        help="Vina scoring mode; score_only is faster and avoids local minimization")
    parser.add_argument("--mgl_python", default=DEFAULT_MGL_PYTHON,
                        help="MGLTools Python runner. Default 'auto' uses $MGLTOOLS_PYTHON, then $CONDA_PREFIX/bin/python2.7, then python2.7/pythonsh from PATH. Use direct/none to call prepare scripts directly")
    parser.add_argument("--prepare_ligand", default=DEFAULT_PREP_LIGAND, help="prepare_ligand4.py name/path; default uses active environment PATH")
    parser.add_argument("--prepare_receptor", default=DEFAULT_PREP_RECEPTOR, help="prepare_receptor4.py name/path; default uses active environment PATH")
    parser.add_argument("--pyrosetta_flags", default=DEFAULT_PYROSETTA_FLAGS,
                        help="flags passed to pyrosetta.init()")
    parser.add_argument("--rosetta_ddg_repeats", type=int, default=1,
                        help="Rosetta Ddg repeats when --with_rosetta_ddg is set; use 5 to match CPMirror exactly")
    parser.add_argument("--with_rosetta_ddg", action="store_true",
                        help="calculate and output rosetta_ddg and rosetta_ddg_norepack; skipped by default")
    parser.add_argument("--show_rosetta_log", action="store_true",
                        help="show PyRosetta/Rosetta/APBS stdout and stderr instead of muting them")
    parser.add_argument("--with_ec", action="store_true",
                        help="compute Rosetta ElectrostaticComplementarityMetric/APBS; slow and noisy, skipped by default")
    parser.add_argument("--skip_ec", action="store_true",
                        help="legacy alias; EC/APBS is already skipped by default unless --with_ec is set")
    parser.add_argument("--rosetta_xml", default=DEFAULT_ROSETTA_XML)
    parser.add_argument("--plip_path", default=DEFAULT_PLIP, help="plip executable name/path; default uses active environment PATH")
    parser.add_argument("--error_output", default=None,
                        help="optional TSV file for per-structure metric errors; default is <output>.errors.tsv")
    args = parser.parse_args()

    paths = {
        "vina": args.vina_path,
        "vina_mode": args.vina_mode,
        "mgl_python": args.mgl_python,
        "prep_ligand": args.prepare_ligand,
        "prep_receptor": args.prepare_receptor,
        "pyrosetta_flags": args.pyrosetta_flags,
        "quiet_rosetta": not args.show_rosetta_log,
        "with_rosetta_ddg": args.with_rosetta_ddg,
        "rosetta_ddg_repeats": args.rosetta_ddg_repeats,
        "skip_ec": args.skip_ec or not args.with_ec,
        "rosetta_xml": resolve_rosetta_xml_path(args.rosetta_xml),
        "plip": args.plip_path,
        "plip_count_mode": args.plip_count_mode or args.plip_mode or "combined",
        "plip_hotspot_mode": args.plip_hotspot_mode or args.plip_mode or "combined",
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

    columns = output_columns(include_rosetta_ddg=args.with_rosetta_ddg)
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
            writer = csv.DictWriter(handle, fieldnames=ERROR_COLUMNS, delimiter="\t")
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
