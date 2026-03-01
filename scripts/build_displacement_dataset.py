import argparse
import glob
import json
import os

import numpy as np
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




def analyze_residue_name_matches(af2_struct, holo_struct, af2_idx, holo_idx):
    mismatches = []
    match_count = 0

    total = len(af2_idx)
    preview_count = min(10, total)
    print(f"[debug] Residue-name check preview (first {preview_count}/{total} aligned residue pairs):")

    for pair_pos, (i, j) in enumerate(zip(af2_idx.tolist(), holo_idx.tolist())):
        af2_rid = af2_struct["residue_ids"][i]
        holo_rid = holo_struct["residue_ids"][j]
        af2_name = af2_struct.get("residue_names", ["UNK"] * len(af2_struct["residue_ids"]))[i]
        holo_name = holo_struct.get("residue_names", ["UNK"] * len(holo_struct["residue_ids"]))[j]

        if pair_pos < preview_count:
            print(f"[debug]   AF2 {af2_rid}/{af2_name} <-> PDB {holo_rid}/{holo_name}")

        if af2_name == holo_name:
            match_count += 1
        else:
            mismatches.append((af2_rid, holo_rid, af2_name, holo_name))

    mismatch_count = len(mismatches)
    mismatch_ratio = (mismatch_count / total) if total else 0.0

    print(f"[debug] Residue-name consistency: matches={match_count}, mismatches={mismatch_count}, mismatch_ratio={mismatch_ratio:.2%}")
    if mismatches:
        print("[debug] Residue-name mismatch examples:")
        for af2_rid, holo_rid, af2_name, holo_name in mismatches[:10]:
            print(f"[debug]   AF2 {af2_rid}/{af2_name} <-> PDB {holo_rid}/{holo_name}")

    if mismatch_ratio >= 0.30:
        print("[warning] High amino-acid mismatch ratio detected after sequence alignment. This usually indicates residue numbering misalignment. Consider renumbering the PDB file to UniProt residue IDs before building the dataset.")


def compute_rmsd(delta_r):
    return float(np.sqrt(np.mean(np.sum(np.square(delta_r), axis=1))))

def build_case_insensitive_file_index(directory, pattern="*.pdb"):
    index = {}
    for path in glob.glob(os.path.join(directory, pattern)):
        stem = os.path.splitext(os.path.basename(path))[0]
        index[stem.lower()] = path
    return index


def main():
    args = args_parser()
    os.makedirs(args.out_dir, exist_ok=True)
    parser = StructureParser()

    pdb_to_uniprot = load_pdb_to_uniprot_mapping(args.mapping_file)
    uniprot_to_af2_path = build_uniprot_to_af2_path(args.af2_dir)
    af2_casefold = {k.lower(): v for k, v in uniprot_to_af2_path.items()}
    holo_casefold = build_case_insensitive_file_index(args.holo_dir)

    built = 0
    for pdb_id, uniprot_id in sorted(pdb_to_uniprot.items()):
        af2 = uniprot_to_af2_path.get(uniprot_id)
        if not af2:
            af2 = af2_casefold.get(uniprot_id.lower())
        if not af2:
            print(f"[debug] Skip {pdb_id}: AF2 structure not found for UniProt '{uniprot_id}'")
            continue

        holo = os.path.join(args.holo_dir, f"{pdb_id}.pdb")
        if not os.path.exists(holo):
            holo = holo_casefold.get(pdb_id.lower())
        if not holo or not os.path.exists(holo):
            print(f"[debug] Skip {pdb_id}: holo structure not found (case-insensitive lookup tried)")
            continue

        a = parser.parse_ca_structure(af2)
        h = parser.parse_ca_structure(holo)
        if not a or not h:
            print(f"[debug] Skip {pdb_id}: parse failed (af2_ok={bool(a)}, holo_ok={bool(h)})")
            continue
        try:
            delta_r, ids, af2_aligned, af2_idx, holo_idx = compute_displacement_target(
                a["coords"],
                h["coords"],
                a["residue_ids"],
                h["residue_ids"],
                a["sequence"],
                h["sequence"],
            )
        except ValueError:
            print(f"[debug] Skip {pdb_id}: compute_displacement_target raised ValueError")
            continue

        analyze_residue_name_matches(a, h, af2_idx, holo_idx)
        rmsd = compute_rmsd(delta_r)

        out = {
            "pair_id": pdb_id,
            "residue_ids": ids,
            "af2_pos": torch.tensor(af2_aligned),
            "holo_pos": torch.tensor(af2_aligned + delta_r),
            "y_delta": torch.tensor(delta_r),
            "plddt": torch.tensor(a["plddts"][af2_idx]).unsqueeze(1),
            "sequence": "".join(a["sequence"][i] for i in af2_idx.tolist()),
        }
        torch.save(out, os.path.join(args.out_dir, f"{pdb_id}.pt"))
        built += 1
        print(f"[debug] Built sample {pdb_id}: af2='{os.path.basename(af2)}', holo='{os.path.basename(holo)}', residues={len(ids)}, rmsd={rmsd:.4f} Å")

    print(f"Built {built} paired samples in {args.out_dir}")


if __name__ == "__main__":
    main()
