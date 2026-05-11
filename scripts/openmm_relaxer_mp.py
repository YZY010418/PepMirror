#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import io
import pdbfixer
import openmm
from openmm import app as openmm_app
from openmm import unit
import argparse
from tqdm import tqdm  
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms

class ForceFieldMinimizer(object):

    def __init__(
        self,
        stiffness=10.0,
        max_iterations=0,
        tolerance=2.39,
        platform='CPU',
        cuda_device_index=None,
        include_het=False,
    ):
        super().__init__()
        self.stiffness = stiffness * unit.kilojoules_per_mole/unit.nanometer / (unit.angstroms ** 2)
        self.max_iterations = max_iterations
        self.tolerance = tolerance * unit.kilojoules_per_mole/unit.nanometer
        assert platform in ('CUDA', 'CPU')
        self.platform = platform
        self.cuda_device_index = cuda_device_index
        self.include_het = include_het

    @staticmethod
    def _atom_serial(line):
        try:
            return int(line[6:11])
        except Exception:
            return None

    @staticmethod
    def _atom_chain_id(line):
        return line[21].strip()

    @staticmethod
    def _make_ter_line(last_atom_line, serial):
        return f"TER   {serial:5d}      {last_atom_line[17:20]} {last_atom_line[21:22]}{last_atom_line[22:27]}\n"

    @staticmethod
    def _record_type(line):
        if line.startswith("ATOM"):
            return "ATOM"
        if line.startswith("HETATM"):
            return "HETATM"
        return None

    def _preprocess_pdb(self, pdb_str):
        """Filter HETATM records and add TER records at protein/heterogen breaks.

        OpenMM/PDBFixer only recognizes N/C termini at chain boundaries.  HETATM
        records between protein residues should therefore split the surrounding
        ATOM fragments, so missing terminal atoms such as N-terminal hydrogens
        and C-terminal OXT can be added for both sides of the break.
        """
        lines = pdb_str.splitlines(keepends=True)
        all_atom_serials = {
            self._atom_serial(line)
            for line in lines
            if line.startswith(("ATOM", "HETATM"))
        }
        all_atom_serials.discard(None)
        next_ter_serial = (max(all_atom_serials) + 1) if all_atom_serials else 1
        serial_to_record = {
            self._atom_serial(line): self._record_type(line)
            for line in lines
            if line.startswith(("ATOM", "HETATM"))
        }
        serial_to_record.pop(None, None)

        kept_atom_serials = {
            self._atom_serial(line)
            for line in lines
            if line.startswith("ATOM")
        }
        kept_atom_serials.discard(None)

        out = []
        previous_atom_line = None
        previous_record = None
        removed_het_since_atom = False

        def append_ter_if_needed():
            nonlocal next_ter_serial
            if previous_atom_line is None:
                return
            if out and out[-1].startswith("TER"):
                return
            out.append(self._make_ter_line(previous_atom_line, next_ter_serial))
            next_ter_serial += 1

        for line in lines:
            record = self._record_type(line)

            if record == "HETATM" and not self.include_het:
                if previous_atom_line is not None:
                    removed_het_since_atom = True
                continue

            if record in {"ATOM", "HETATM"}:
                if previous_atom_line is not None:
                    chain_changed = self._atom_chain_id(previous_atom_line) != self._atom_chain_id(line)
                    crossed_removed_het = removed_het_since_atom and record == "ATOM"
                    crossed_kept_het = self.include_het and previous_record != record and {"ATOM", "HETATM"} == {previous_record, record}
                    if chain_changed or crossed_removed_het or crossed_kept_het:
                        append_ter_if_needed()

                out.append(line)
                previous_atom_line = line
                previous_record = record
                removed_het_since_atom = False
                continue

            if line.startswith("TER"):
                append_ter_if_needed()
                previous_atom_line = None
                previous_record = None
                removed_het_since_atom = False
                continue

            if line.startswith("CONECT"):
                serials = []
                for item in line.split()[1:]:
                    try:
                        serials.append(int(item))
                    except Exception:
                        pass
                if not serials:
                    continue
                if not self.include_het and all(serial in kept_atom_serials for serial in serials):
                    out.append(line)
                    continue
                # With HETATM retained, keep only bonds within the protein side
                # or within the heterogen side. Cross ATOM-HETATM bonds would
                # make the split protein residues non-terminal again.
                record_types = {serial_to_record.get(serial) for serial in serials}
                if self.include_het and None not in record_types and len(record_types) == 1:
                    out.append(line)
                continue

            if line.startswith(("END", "ENDMDL")):
                append_ter_if_needed()
                previous_atom_line = None
                previous_record = None
                removed_het_since_atom = False
                out.append(line)
                continue

            out.append(line)

        if previous_atom_line is not None:
            append_ter_if_needed()
        if not out or not out[-1].startswith("END"):
            out.append("END\n")
        return "".join(out)

    def _fix(self, pdb_str):
        pdb_str = self._preprocess_pdb(pdb_str)
        fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_str))
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)
        fixer.addMissingHydrogens()

        out_handle = io.StringIO()
        openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle, keepIds=True)
        return out_handle.getvalue()

    def _get_pdb_string(self, topology, positions):
        with io.StringIO() as f:
            openmm_app.PDBFile.writeFile(topology, positions, f, keepIds=True)
            return f.getvalue()
        
    def _minimize(self, pdb_str):
        pdb = openmm_app.PDBFile(io.StringIO(pdb_str))

        force_field = openmm_app.ForceField("amber14-all.xml")
        constraints = openmm_app.HBonds
        system = force_field.createSystem(pdb.topology, constraints=None)

        # Add constraints to non-generated regions
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", self.stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)
        
        for chain in pdb.topology.chains():
            for atom in chain.atoms():
                if atom.element.name != 'hydrogen':
                    force.addParticle(atom.index, pdb.positions[atom.index])
                
        system.addForce(force)
        
        # Set up the integrator and simulation
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName(self.platform)

        platform_props = {}
        if self.platform == 'CPU':
            platform_props['Threads'] = '1'
        elif self.platform == 'CUDA' and self.cuda_device_index is not None:
            platform_props['DeviceIndex'] = str(self.cuda_device_index)

        simulation = openmm_app.Simulation(pdb.topology, system, integrator, platform, platform_props)
        simulation.context.setPositions(pdb.positions)

        # Perform minimization
        ret = {}
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        simulation.minimizeEnergy(maxIterations=self.max_iterations, tolerance=self.tolerance)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
        ret["min_pdb"] = self._get_pdb_string(simulation.topology, state.getPositions())

        return ret['min_pdb'], ret
    
    def _add_energy_remarks(self, pdb_str, ret):
        pdb_lines = pdb_str.splitlines()
        pdb_lines.insert(1, "REMARK   1  FINAL ENERGY:   {:.3f} KCAL/MOL".format(ret['efinal']))
        pdb_lines.insert(1, "REMARK   1  INITIAL ENERGY: {:.3f} KCAL/MOL".format(ret['einit']))
        return "\n".join(pdb_lines)

    def __call__(self, pdb_str, out_path, return_info=True):
        if '\n' not in pdb_str and pdb_str.lower().endswith(".pdb"):
            with open(pdb_str) as f:
                pdb_str = f.read()

        pdb_fixed = self._fix(pdb_str)
        pdb_min, ret = self._minimize(pdb_fixed)
        pdb_min = self._add_energy_remarks(pdb_min, ret)
        with open(out_path, 'w') as f:
            f.write(pdb_min)
        if return_info:
            return pdb_min, ret
        else:
            return pdb_min


_WORKER_MINIMIZER = None
_WORKER_PLATFORM = None
_WORKER_INCLUDE_HET = False


def _init_worker(platform: str, gpu_ids, include_het=False):
    """Initializer for multiprocessing workers.

    For CUDA, bind each worker process to a (possibly shared) GPU by round-robin.
    """
    global _WORKER_MINIMIZER, _WORKER_PLATFORM, _WORKER_INCLUDE_HET
    _WORKER_PLATFORM = platform
    _WORKER_INCLUDE_HET = include_het

    cuda_device_index = None
    if platform == 'CUDA':
        # mp.current_process()._identity is typically (1..N) for pool workers.
        ident = 1
        try:
            ident = mp.current_process()._identity[0]  # 1-based
        except Exception:
            try:
                # Fallback: parse trailing digits from process name (e.g., 'SpawnProcess-3').
                name = mp.current_process().name
                digits = ''.join([c for c in name if c.isdigit()])
                if digits:
                    ident = int(digits)
            except Exception:
                ident = 1
        if not gpu_ids:
            gpu_ids = [0]
        cuda_device_index = gpu_ids[(ident - 1) % len(gpu_ids)]

    _WORKER_MINIMIZER = ForceFieldMinimizer(
        platform=platform,
        cuda_device_index=cuda_device_index,
        include_het=include_het,
    )


def _worker_minimize(task):
    """Worker entrypoint: minimize one PDB and write output.

    Returns:
        (input_file_path, error_message_or_None)
    """
    global _WORKER_MINIMIZER
    input_file_path, output_file_path = task
    try:
        out_dir = os.path.dirname(output_file_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if _WORKER_MINIMIZER is None:
            # Fallback (shouldn't happen if initializer runs)
            _WORKER_MINIMIZER = ForceFieldMinimizer(
                platform=_WORKER_PLATFORM or 'CPU',
                include_het=_WORKER_INCLUDE_HET,
            )
        _WORKER_MINIMIZER(input_file_path, output_file_path, return_info=False)
        return input_file_path, None
    except Exception as e:
        return input_file_path, str(e)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {'true', '1', 'yes', 'y', 't'}:
        return True
    if s in {'false', '0', 'no', 'n', 'f'}:
        return False
    raise argparse.ArgumentTypeError("--prefix must be true/false")


def process_directory(input_dir, output_dir, platform='CUDA', nproc=None, prefix=False, gpu_ids=None, include_het=False):
    if not os.path.exists(input_dir):
        print(f"{input_dir} doesn't exist")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    if nproc is None:
        if platform == 'CPU':
            nproc = os.cpu_count() or 1
        else:
            # Conservative default for GPU: one worker per GPU if provided, else 1.
            nproc = len(gpu_ids) if gpu_ids else 1

    pdb_files = []
    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        if relative_path == '.':
            relative_path = ''
        
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        for file in files:
            if file.lower().endswith('.pdb'):
                input_file_path = os.path.join(root, file)
                base_name, ext = os.path.splitext(file)
                output_file_name = f"{base_name}_relaxed{ext}"
                if prefix and relative_path:
                    # Use the containing subfolder name (direct parent folder) as prefix.
                    folder_prefix = os.path.basename(root)
                    if folder_prefix:
                        output_file_name = f"{folder_prefix}_{output_file_name}"
                output_file_path = os.path.join(output_subdir, output_file_name)
                pdb_files.append((input_file_path, output_file_path))

    if len(pdb_files) == 0:
        print(f"No PDB files found under: {input_dir}")
        return

    # Avoid spawning more workers than tasks.
    nproc = max(1, min(int(nproc), len(pdb_files)))

    if platform == 'CUDA':
        if not gpu_ids:
            gpu_ids = [0]
            if nproc > 1:
                print("[WARN] --platform CUDA with --nproc>1 and no --gpu_ids provided: all workers will use GPU 0.")
        if nproc > len(gpu_ids):
            print(f"[WARN] CUDA workers ({nproc}) exceed GPU ids ({len(gpu_ids)}); GPUs will be shared round-robin.")

    failed = []
    if nproc == 1:
        cuda_device_index = gpu_ids[0] if (platform == 'CUDA' and gpu_ids) else None
        minimizer = ForceFieldMinimizer(
            platform=platform,
            cuda_device_index=cuda_device_index,
            include_het=include_het,
        )
        for input_file_path, output_file_path in tqdm(pdb_files, desc="Processing files"):
            try:
                minimizer(input_file_path, output_file_path, return_info=False)
            except Exception as e:
                failed.append((input_file_path, str(e)))
    else:
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=nproc,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(platform, gpu_ids, include_het),
        ) as ex:
            # map() is generally faster than submitting one future per task for large batches.
            for input_file_path, err in tqdm(
                ex.map(_worker_minimize, pdb_files, chunksize=1),
                total=len(pdb_files),
                desc=f"Processing files ({nproc} workers)",
            ):
                if err:
                    failed.append((input_file_path, err))

    if failed:
        print(f"Failed files: {len(failed)}/{len(pdb_files)}")
        for p, err in failed[:20]:
            print(f"  - {p}: {err}")
        if len(failed) > 20:
            print("  ... (showing first 20 failures)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minimize PDB files using OpenMM')
    parser.add_argument('input', help='Input file or directory path')
    parser.add_argument('output', help='Output file or directory path')
    parser.add_argument('--platform', choices=['CPU', 'CUDA'], default='CUDA', 
                        help='Computation platform (default: CUDA)')
    parser.add_argument('--nproc', type=int, default=None,
                        help='Number of worker processes for directory mode. '
                             'CPU default: use all available CPU cores. '
                             'CUDA default: 1 (or one per --gpu_ids if provided).')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated GPU ids to use for CUDA (e.g., "0" or "0,1,2"). '
                             'Workers will be bound round-robin. If omitted, defaults to GPU 0.')
    parser.add_argument('--prefix', type=_str2bool, default=False,
                        help='true/false. If true, prepend the output filename with the '
                             'containing subfolder name, in addition to appending "_relaxed".')
    parser.add_argument('--include_het', action='store_true',
                        help='Keep HETATM records during relaxation. By default all HETATM records '
                             'are removed before PDBFixer/OpenMM. When enabled, TER records are '
                             'inserted around HETATM blocks so adjacent protein residues are treated '
                             'as N/C termini and can receive missing terminal atoms.')
    args = parser.parse_args()
    
    # Parse GPU ids
    gpu_ids = None
    if args.gpu_ids is not None:
        s = args.gpu_ids.strip()
        if s:
            gpu_ids = [int(x) for x in s.split(',') if x.strip() != '']

    if os.path.isfile(args.input):
        if not args.input.lower().endswith('.pdb'):
            print(f"Error: Input file '{args.input}' is not a PDB file")
            exit(1)
        
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if not args.output.lower().endswith('_relaxed.pdb'):
            base_name, ext = os.path.splitext(args.output)
            args.output = f"{base_name}_relaxed{ext}"
        
        cuda_device_index = None
        if args.platform == 'CUDA':
            cuda_device_index = (gpu_ids or [0])[0]

        minimizer = ForceFieldMinimizer(
            platform=args.platform,
            cuda_device_index=cuda_device_index,
            include_het=args.include_het,
        )
        minimizer(args.input, args.output, return_info=False)
    else:
        process_directory(
            args.input,
            args.output,
            platform=args.platform,
            nproc=args.nproc,
            prefix=args.prefix,
            gpu_ids=gpu_ids,
            include_het=args.include_het,
        )
