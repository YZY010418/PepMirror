#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Move or remove PDBs with wrong residue chirality in one chain")
    p.add_argument("-i", "--input", required=True, help="Input directory of PDB files")
    p.add_argument("--failed", help="Directory for failed PDB files")
    p.add_argument("--chirality", required=True, choices=["L", "D"], help="Desired chirality")
    p.add_argument("-c", "--chain", required=True, help="Chain ID to check")
    p.add_argument("--remove_failed", action="store_true", help="Delete failed PDBs instead of moving them")
    return p.parse_args()


def atom_name(line):
    return line[12:16].strip()


def chain_id(line):
    return line[21].strip()


def residue_key(line):
    return line[22:27], line[17:20].strip()


def xyz(line):
    return tuple(float(line[i:i + 8]) for i in (30, 38, 46))


def chirality(n, ca, c, cb):
    v_n = [n[i] - ca[i] for i in range(3)]
    v_c = [c[i] - ca[i] for i in range(3)]
    v_b = [cb[i] - ca[i] for i in range(3)]
    normal = (
        v_n[1] * v_c[2] - v_n[2] * v_c[1],
        v_n[2] * v_c[0] - v_n[0] * v_c[2],
        v_n[0] * v_c[1] - v_n[1] * v_c[0],
    )
    dot = sum(normal[i] * v_b[i] for i in range(3))
    return "D" if dot < 0.0 else "L"


def fails_chirality(path, chain, desired):
    residues = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")) or chain_id(line) != chain:
            continue
        residues.setdefault(residue_key(line), {})[atom_name(line)] = xyz(line)

    for (_, res_name), atoms in residues.items():
        if res_name == "GLY":
            continue
        if all(name in atoms for name in ("N", "CA", "C", "CB")):
            if chirality(atoms["N"], atoms["CA"], atoms["C"], atoms["CB"]) != desired:
                return True
    return False


def main():
    args = parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    failed_dir = Path(args.failed) if args.failed else None
    if not args.remove_failed and failed_dir is None:
        raise ValueError("--failed is required unless --remove_failed is set")
    if failed_dir is not None:
        failed_dir.mkdir(parents=True, exist_ok=True)

    for pdb in sorted(input_dir.glob("*.pdb")):
        if not fails_chirality(pdb, args.chain, args.chirality):
            continue
        if args.remove_failed:
            pdb.unlink()
            print(f"[removed] {pdb}")
        else:
            dst = failed_dir / pdb.name
            shutil.move(str(pdb), str(dst))
            print(f"[failed] {pdb} -> {dst}")


if __name__ == "__main__":
    main()
