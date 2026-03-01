import argparse
import glob
import json
import os
import torch

from evopoint_da.data.components import StructureParser, compute_displacement_target


def args_parser():
    p = argparse.ArgumentParser(description="Build Δr targets from AF2/Holo structure pairs.")
    p.add_argument("--af2_dir", default="data/raw_af2")
    p.add_argument("--holo_dir", default="data/raw_pdb")
    p.add_argument("--out_dir", default="data/processed_pairs")
    p.add_argument("--mapping_file", default="pdb_uniprot_mapping.json")
    return p.parse_args()


def load_pdb_to_uniprot_mapping(mapping_file):
    with open(mapping_file, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    return {pdb_id.upper(): uniprot_id for pdb_id, uniprot_id in raw_mapping.items()}


def build_uniprot_to_af2_path(af2_dir):
    uniprot_to_path = {}
    for af2_path in glob.glob(os.path.join(af2_dir, "*.pdb")):
        stem = os.path.splitext(os.path.basename(af2_path))[0]
        # get_af2.py stores AlphaFold files as AF-<UNIPROT>.pdb
        if stem.startswith("AF-"):
            uniprot_id = stem[3:]
        else:
            uniprot_id = stem
        uniprot_to_path[uniprot_id] = af2_path
    return uniprot_to_path


def main():
    args = args_parser()
    os.makedirs(args.out_dir, exist_ok=True)
    parser = StructureParser()

    pdb_to_uniprot = load_pdb_to_uniprot_mapping(args.mapping_file)
    uniprot_to_af2_path = build_uniprot_to_af2_path(args.af2_dir)

    built = 0
    for pdb_id, uniprot_id in sorted(pdb_to_uniprot.items()):
        af2 = uniprot_to_af2_path.get(uniprot_id)
        if not af2:
            continue

        holo = os.path.join(args.holo_dir, f"{pdb_id}.pdb")
        if not os.path.exists(holo):
            continue

        a = parser.parse_ca_structure(af2)
        h = parser.parse_ca_structure(holo)
        if not a or not h:
            continue
        try:
            delta_r, ids, af2_aligned = compute_displacement_target(a["coords"], h["coords"], a["residue_ids"], h["residue_ids"])
        except ValueError:
            continue

        out = {
            "pair_id": pdb_id,
            "residue_ids": ids,
            "af2_pos": torch.tensor(af2_aligned),
            "holo_pos": torch.tensor(af2_aligned + delta_r),
            "y_delta": torch.tensor(delta_r),
            "plddt": torch.tensor(a["plddts"][: len(ids)]).unsqueeze(1),
            "sequence": a["sequence"][: len(ids)],
        }
        torch.save(out, os.path.join(args.out_dir, f"{pdb_id}.pt"))
        built += 1

    print(f"Built {built} paired samples in {args.out_dir}")


if __name__ == "__main__":
    main()
