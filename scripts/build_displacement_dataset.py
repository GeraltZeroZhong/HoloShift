import argparse
import glob
import os
import torch

from evopoint_da.data.components import StructureParser, compute_displacement_target


def args_parser():
    p = argparse.ArgumentParser(description="Build Δr targets from AF2/Holo structure pairs.")
    p.add_argument("--af2_dir", default="data/raw_af2")
    p.add_argument("--holo_dir", default="data/raw_pdb")
    p.add_argument("--out_dir", default="data/processed_pairs")
    return p.parse_args()


def main():
    args = args_parser()
    os.makedirs(args.out_dir, exist_ok=True)
    parser = StructureParser()

    af2_files = sorted(glob.glob(os.path.join(args.af2_dir, "*.pdb")))
    built = 0
    for af2 in af2_files:
        stem = os.path.splitext(os.path.basename(af2))[0]
        holo = os.path.join(args.holo_dir, f"{stem}.pdb")
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
            "pair_id": stem,
            "residue_ids": ids,
            "af2_pos": torch.tensor(af2_aligned),
            "holo_pos": torch.tensor(af2_aligned + delta_r),
            "y_delta": torch.tensor(delta_r),
            "plddt": torch.tensor(a["plddts"][: len(ids)]).unsqueeze(1),
            "sequence": a["sequence"][: len(ids)],
        }
        torch.save(out, os.path.join(args.out_dir, f"{stem}.pt"))
        built += 1

    print(f"Built {built} paired samples in {args.out_dir}")


if __name__ == "__main__":
    main()
