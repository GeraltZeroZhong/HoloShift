import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from Bio.PDB import PDBIO, PDBParser
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from evopoint_da.data.components import StructureParser, build_knn_edges
from evopoint_da.models.module import EvoPointLitModule


def _select_chain(chains: dict, chain_id: str | None):
    if chain_id is not None:
        if chain_id not in chains:
            raise ValueError(f"Requested chain_id={chain_id} not found. Available: {list(chains.keys())}")
        return chain_id, chains[chain_id]
    best_id = max(chains.keys(), key=lambda cid: len(chains[cid]["coords"]))
    return best_id, chains[best_id]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run conformal-safe displacement prediction, then FASPR side-chain packing and "
            "OpenMM restrained minimization."
        )
    )
    p.add_argument("--pdb_file", required=True)
    p.add_argument("--feature_pt", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--conformal_stats", required=True)
    p.add_argument("--reject_threshold", type=float, default=5.0)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--chain_id", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--output_dir", default="artifacts/relaxation")
    p.add_argument("--faspr_bin", default="FASPR", help="Path to FASPR executable")
    p.add_argument(
        "--faspr_extra_args",
        nargs="*",
        default=[],
        help="Extra args passed directly to FASPR, e.g. --faspr_extra_args -r 0",
    )

    p.add_argument("--restraint_k", type=float, default=1000.0, help="kJ/(mol*nm^2)")
    p.add_argument("--max_iterations", type=int, default=500)
    p.add_argument(
        "--restrain_selection",
        choices=["heavy", "ca"],
        default="heavy",
        help="Apply restraints to all heavy atoms or C-alpha atoms only",
    )
    return p.parse_args()


def _load_qhat(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return float(stats["qhat"])


def _predict_displacement(args: argparse.Namespace) -> tuple[str, np.ndarray, np.ndarray, float, bool]:
    parser = StructureParser()
    parsed_chains = parser.parse_ca_structure(args.pdb_file)
    if not parsed_chains:
        raise ValueError("Failed to parse input structure")

    selected_chain_id, parsed = _select_chain(parsed_chains, args.chain_id)

    feat = torch.load(args.feature_pt, weights_only=False)
    pos = torch.tensor(parsed["coords"], dtype=torch.float32)
    x = feat["x"].float()

    if x.size(0) != len(pos):
        n = min(x.size(0), len(pos))
        print(
            f"[warning] Feature length ({x.size(0)}) != selected chain length ({len(pos)}); truncating to {n}",
            file=sys.stderr,
        )
        x = x[:n]
        pos = pos[:n]

    edge_index, edge_attr = build_knn_edges(pos, k=args.k)
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr).to(args.device)

    model = EvoPointLitModule.load_from_checkpoint(args.ckpt_path, map_location=args.device)
    model.eval().to(args.device)

    with torch.no_grad():
        delta = model.predict_displacement(data)

    qhat = _load_qhat(args.conformal_stats)
    reject = qhat > args.reject_threshold
    safe_centers = pos if reject else (pos + delta)

    return (
        selected_chain_id,
        pos.detach().cpu().numpy(),
        safe_centers.detach().cpu().numpy(),
        qhat,
        reject,
    )


def _iter_chain_residues(structure, chain_id: str) -> Iterable:
    for model in structure:
        if chain_id in model:
            for res in model[chain_id]:
                if res.id[0] == " ":
                    yield res
            return
    raise ValueError(f"Chain {chain_id} not found in {structure.id}")


def _write_guardrailed_pdb(input_pdb: str, chain_id: str, source_ca: np.ndarray, target_ca: np.ndarray, out_pdb: str) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input", input_pdb)

    residues = list(_iter_chain_residues(structure, chain_id))
    if len(residues) < len(source_ca):
        raise ValueError(
            f"Selected chain has fewer residues in full-atom PDB ({len(residues)}) than CA trace ({len(source_ca)})"
        )

    for i, (src_ca, dst_ca) in enumerate(zip(source_ca, target_ca, strict=False)):
        shift = dst_ca - src_ca
        residue = residues[i]
        for atom in residue.get_atoms():
            atom.coord = atom.coord + shift

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)


def _run_faspr(faspr_bin: str, input_pdb: str, output_pdb: str, extra_args: list[str]) -> None:
    cmd = [faspr_bin, "-i", input_pdb, "-o", output_pdb, *extra_args]
    subprocess.run(cmd, check=True)


def _run_openmm_restrained_minimization(
    input_pdb: str,
    output_pdb: str,
    restraint_k: float,
    max_iterations: int,
    restrain_selection: str,
) -> None:
    try:
        import openmm
        from openmm import app, unit
    except ImportError as exc:
        raise ImportError(
            "OpenMM not found. Install openmm before running minimization, e.g. `conda install -c conda-forge openmm`."
        ) from exc

    pdb = app.PDBFile(input_pdb)
    forcefield = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    restraint = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", restraint_k * unit.kilojoule_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    for atom_idx, atom in enumerate(pdb.topology.atoms()):
        is_h = atom.element is not None and atom.element.symbol == "H"
        is_ca = atom.name == "CA"
        if restrain_selection == "heavy" and is_h:
            continue
        if restrain_selection == "ca" and not is_ca:
            continue
        pos_nm = pdb.positions[atom_idx].value_in_unit(unit.nanometer)
        restraint.addParticle(atom_idx, [pos_nm.x, pos_nm.y, pos_nm.z])

    system.addForce(restraint)

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.004 * unit.picoseconds,
    )
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=max_iterations)

    state = simulation.context.getState(getPositions=True, getEnergy=True)
    minimized_positions = state.getPositions()
    with open(output_pdb, "w", encoding="utf-8") as f:
        app.PDBFile.writeFile(pdb.topology, minimized_positions, f)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chain_id, source_ca, safe_ca, qhat, reject = _predict_displacement(args)

    guardrailed_pdb = output_dir / "01_guardrailed_backbone.pdb"
    faspr_pdb = output_dir / "02_faspr_packed.pdb"
    minimized_pdb = output_dir / "03_openmm_minimized.pdb"
    report_json = output_dir / "relax_report.json"

    _write_guardrailed_pdb(args.pdb_file, chain_id, source_ca, safe_ca, str(guardrailed_pdb))
    _run_faspr(args.faspr_bin, str(guardrailed_pdb), str(faspr_pdb), args.faspr_extra_args)
    _run_openmm_restrained_minimization(
        str(faspr_pdb),
        str(minimized_pdb),
        restraint_k=args.restraint_k,
        max_iterations=args.max_iterations,
        restrain_selection=args.restrain_selection,
    )

    report = {
        "chain_id": chain_id,
        "qhat": qhat,
        "reject_threshold": args.reject_threshold,
        "reject": reject,
        "safety_radius": qhat,
        "artifacts": {
            "guardrailed_pdb": str(guardrailed_pdb),
            "faspr_pdb": str(faspr_pdb),
            "minimized_pdb": str(minimized_pdb),
        },
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    status = "REJECT" if reject else "ACCEPT"
    print(f"[{status}] qhat={qhat:.4f}, chain={chain_id}")
    print(f"Guardrailed backbone: {guardrailed_pdb}")
    print(f"FASPR packed:         {faspr_pdb}")
    print(f"OpenMM minimized:     {minimized_pdb}")
    print(f"Report JSON:          {report_json}")


if __name__ == "__main__":
    main()
