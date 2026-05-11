#!/usr/bin/env python3
import argparse
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate mirrored D-protein PDBs from L-protein PDBs. "
            "The default mode is central inversion around each molecule center."
        )
    )
    parser.add_argument("-i", "--input", required=True, help="Input PDB file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output PDB file or directory")
    parser.add_argument(
        "-m",
        "--mode",
        default="central",
        choices=["central", "x", "y", "z", "random"],
        help=(
            "Transformation mode. central: invert through molecule center; "
            "x/y/z: mirror across the plane normal to that axis through molecule center; "
            "random: mirror across a random plane through molecule center."
        ),
    )
    parser.add_argument(
        "--recursive",
        "--subfolder",
        action="store_true",
        help="Process PDB files recursively when input is a directory",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of worker processes for directory inputs (default: all available cores)",
    )
    parser.add_argument(
        "--suffix",
        default="_mirror",
        help="Suffix appended to generated PDB basenames when output is a directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --mode random",
    )
    return parser.parse_args()


def is_atom_line(line):
    return line.startswith("ATOM") or line.startswith("HETATM")


def parse_xyz_from_pdb_line(line):
    try:
        return float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        return None


def format_pdb_line_with_xyz(line, x, y, z):
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
    n_coords = len(coords)
    return (
        sum(x for x, _, _ in coords) / n_coords,
        sum(y for _, y, _ in coords) / n_coords,
        sum(z for _, _, z in coords) / n_coords,
    )


def random_unit_vector(rng):
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        norm = math.sqrt(x * x + y * y + z * z)
        if norm > 1e-12:
            return x / norm, y / norm, z / norm


def reflect_across_plane(point, center, normal):
    px, py, pz = point
    cx, cy, cz = center
    nx, ny, nz = normal

    vx = px - cx
    vy = py - cy
    vz = pz - cz
    dot = vx * nx + vy * ny + vz * nz

    return (
        px - 2.0 * dot * nx,
        py - 2.0 * dot * ny,
        pz - 2.0 * dot * nz,
    )


def transform_point(point, mode, center, random_normal=None):
    x, y, z = point
    cx, cy, cz = center

    if mode == "central":
        return 2.0 * cx - x, 2.0 * cy - y, 2.0 * cz - z
    if mode == "x":
        return 2.0 * cx - x, y, z
    if mode == "y":
        return x, 2.0 * cy - y, z
    if mode == "z":
        return x, y, 2.0 * cz - z
    if mode == "random":
        if random_normal is None:
            raise ValueError("random_normal is required for random mode")
        return reflect_across_plane(point, center, random_normal)

    raise ValueError(f"Unsupported mode: {mode}")


def process_pdb_file(input_file, output_file, mode, random_normal=None):
    input_file = Path(input_file)
    output_file = Path(output_file)
    lines = input_file.read_text().splitlines(keepends=True)
    center = compute_center(collect_atom_coords(lines))

    modified_lines = []
    for line in lines:
        if is_atom_line(line):
            xyz = parse_xyz_from_pdb_line(line)
            if xyz is not None:
                tx, ty, tz = transform_point(xyz, mode, center, random_normal=random_normal)
                line = format_pdb_line_with_xyz(line, tx, ty, tz)
        modified_lines.append(line)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(modified_lines))
    return str(input_file), str(output_file)


def iter_input_pdbs(input_path, recursive):
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdb":
            raise ValueError(f"Input file is not a PDB file: {input_path}")
        return [input_path]

    if input_path.is_dir():
        candidates = input_path.rglob("*") if recursive else input_path.iterdir()
        pdbs = sorted(path for path in candidates if path.is_file() and path.suffix.lower() == ".pdb")
        if not pdbs:
            raise FileNotFoundError(f"No .pdb files found in input directory: {input_path}")
        return pdbs

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def output_path_for_file(input_file, input_root, output_root, suffix, single_file_input):
    output_root = Path(output_root)
    if single_file_input and output_root.suffix.lower() == ".pdb":
        return output_root

    if single_file_input:
        relative_parent = Path()
    else:
        relative_parent = input_file.parent.relative_to(input_root)

    return output_root / relative_parent / f"{input_file.stem}{suffix}{input_file.suffix}"


def build_tasks(input_path, output_path, recursive, suffix):
    input_path = Path(input_path)
    pdb_files = iter_input_pdbs(input_path, recursive)
    single_file_input = input_path.is_file()
    input_root = input_path.parent if single_file_input else input_path

    return [
        (
            str(pdb_file),
            str(output_path_for_file(pdb_file, input_root, output_path, suffix, single_file_input)),
        )
        for pdb_file in pdb_files
    ]


def process_single_task(task):
    input_file, output_file, mode, random_normal = task
    return process_pdb_file(input_file, output_file, mode, random_normal=random_normal)


def process_files(input_path, output_path, mode, recursive=False, jobs=None, suffix="_mirror", seed=None):
    tasks = build_tasks(input_path, output_path, recursive, suffix)

    random_normal = None
    if mode == "random":
        random_normal = random_unit_vector(random.Random(seed))
        print(
            "[random-plane] shared unit normal for this run: "
            f"({random_normal[0]:.8f}, {random_normal[1]:.8f}, {random_normal[2]:.8f})"
        )
        print(f"[random-plane] seed: {seed}")

    worker_tasks = [(input_file, output_file, mode, random_normal) for input_file, output_file in tasks]

    if len(worker_tasks) == 1 or jobs == 1:
        for task in tqdm(worker_tasks, desc="Processing PDBs"):
            input_file, output_file = process_single_task(task)
            print(f"[done] {input_file} -> {output_file}")
        return

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(process_single_task, task): task for task in worker_tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDBs"):
            input_file, output_file = future.result()
            print(f"[done] {input_file} -> {output_file}")


def main():
    args = parse_args()
    process_files(
        args.input,
        args.output,
        args.mode,
        recursive=args.recursive,
        jobs=args.jobs,
        suffix=args.suffix,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
