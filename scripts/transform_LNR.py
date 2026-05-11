#!/usr/bin/env python3
import argparse
import math
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform PDB coordinates by mirroring along x/y/z or reflecting across a random plane through the molecule center. In random mode, one shared random plane normal is sampled once per run and reused for all PDBs."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Input directory containing .pdb files",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Output directory for processed .pdb files",
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["x", "y", "z", "random"],
        help="Transformation mode: x, y, z, or random",
    )
    parser.add_argument(
        "--suffix",
        required=True,
        help="Suffix to append to output PDB basenames",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for -m random",
    )
    return parser.parse_args()


def is_atom_line(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")


def parse_xyz_from_pdb_line(line: str):
    """Parse x, y, z from standard PDB columns.

    PDB coordinate columns:
    - x: cols 31-38  -> line[30:38]
    - y: cols 39-46  -> line[38:46]
    - z: cols 47-54  -> line[46:54]
    """
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        return x, y, z
    except ValueError:
        return None


def format_pdb_line_with_xyz(line: str, x: float, y: float, z: float) -> str:
    return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"


def collect_atom_coords(lines):
    coords = []
    for line in lines:
        if is_atom_line(line):
            xyz = parse_xyz_from_pdb_line(line)
            if xyz is not None:
                coords.append(xyz)
    return coords


def compute_center(coords):
    if not coords:
        raise ValueError("No valid ATOM/HETATM coordinates found in the PDB file.")
    n = len(coords)
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    cz = sum(c[2] for c in coords) / n
    return cx, cy, cz


def random_unit_vector(rng: random.Random):
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        norm = math.sqrt(x * x + y * y + z * z)
        if norm > 1e-12:
            return x / norm, y / norm, z / norm


def reflect_across_plane(point, center, normal):
    """Reflect a point across a plane passing through center with unit normal.

    Reflection formula:
        p' = p - 2 * ((p - c) · n) * n
    """
    px, py, pz = point
    cx, cy, cz = center
    nx, ny, nz = normal

    vx = px - cx
    vy = py - cy
    vz = pz - cz

    dot = vx * nx + vy * ny + vz * nz

    rx = px - 2.0 * dot * nx
    ry = py - 2.0 * dot * ny
    rz = pz - 2.0 * dot * nz
    return rx, ry, rz


def transform_point(point, mode, center=None, normal=None):
    x, y, z = point
    if mode == "x":
        return -x, y, z
    if mode == "y":
        return x, -y, z
    if mode == "z":
        return x, y, -z
    if mode == "random":
        if center is None or normal is None:
            raise ValueError("center and normal are required for random reflection")
        return reflect_across_plane(point, center, normal)
    raise ValueError(f"Unsupported mode: {mode}")


def process_pdb_file(infile: Path, outfile: Path, mode: str, shared_normal=None):
    lines = infile.read_text().splitlines(keepends=True)
    coords = collect_atom_coords(lines)
    center = compute_center(coords)

    new_lines = []
    for line in lines:
        if is_atom_line(line):
            xyz = parse_xyz_from_pdb_line(line)
            if xyz is not None:
                tx, ty, tz = transform_point(
                    xyz,
                    mode,
                    center=center,
                    normal=shared_normal,
                )
                line = format_pdb_line_with_xyz(line, tx, ty, tz)
        new_lines.append(line)

    outfile.write_text("".join(new_lines))


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in input directory: {input_dir}")

    rng = random.Random(args.seed)

    shared_normal = None
    if args.mode == "random":
        shared_normal = random_unit_vector(rng)
        print(
            "[random-plane] shared unit normal for this run: "
            f"({shared_normal[0]:.8f}, {shared_normal[1]:.8f}, {shared_normal[2]:.8f})"
        )
        print(f"[random-plane] seed: {args.seed}")
        print("[random-plane] all PDBs in this run will use the same reflection plane normal")

    for pdb_file in pdb_files:
        out_name = f"{pdb_file.stem}{args.suffix}.pdb"
        out_path = output_dir / out_name
        process_pdb_file(
            pdb_file,
            out_path,
            args.mode,
            shared_normal=shared_normal,
        )
        print(f"[done] {pdb_file.name} -> {out_path.name}")


if __name__ == "__main__":
    main()

