import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
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
            # print(f"[debug] Skip {pdb_id}: AF2 not found")
            continue

        holo = os.path.join(args.holo_dir, f"{pdb_id}.pdb")
        if not os.path.exists(holo):
            holo = holo_casefold.get(pdb_id.lower())
        if not holo or not os.path.exists(holo):
            continue

        # --- 修改开始: 适应新的字典返回结构 ---
        af2_chains = parser.parse_ca_structure(af2)
        holo_chains = parser.parse_ca_structure(holo)
        
        if not af2_chains or not holo_chains:
            print(f"[debug] Skip {pdb_id}: parse failed or all chains < 20 AA")
            continue

        try:
            # 传入整个链字典，让函数去挑选最佳匹配
            delta_r, ids, af2_aligned, af2_idx, holo_idx, best_af2_chain_id = compute_displacement_target(
                af2_chains,
                holo_chains,
            )
        except ValueError as e:
            print(f"[debug] Skip {pdb_id}: {e}")
            continue
        
        # 获取被选中的那条链的数据
        selected_af2_data = af2_chains[best_af2_chain_id]
        selected_holo_data = holo_chains[list(holo_chains.keys())[0]] # 仅用于debug打印，具体哪个holo链被选中在内部处理了，若需精确打印需修改返回值
        
        # --- 修改结束 ---

        # analyze_residue_name_matches 逻辑需要微调，或者直接注释掉，因为结构变了
        # 如果你想保留 debug，需要传入选中的字典:
        # analyze_residue_name_matches(selected_af2_data, selected_holo_data, af2_idx, holo_idx)

        rmsd = compute_rmsd(delta_r)

        out = {
            "pair_id": pdb_id,
            "residue_ids": ids,
            "af2_pos": torch.tensor(af2_aligned),
            "holo_pos": torch.tensor(af2_aligned + delta_r),
            "y_delta": torch.tensor(delta_r),
            # 注意：pLDDT 和 sequence 必须来自选中的那条链
            "plddt": torch.tensor(selected_af2_data["plddts"][af2_idx]).unsqueeze(1),
            "sequence": "".join(selected_af2_data["sequence"][i] for i in af2_idx.tolist()),
        }
        torch.save(out, os.path.join(args.out_dir, f"{pdb_id}.pt"))
        built += 1
        print(f"[debug] Built {pdb_id}: chain {best_af2_chain_id}, res={len(ids)}, rmsd={rmsd:.4f} Å")

    print(f"Built {built} paired samples.")




if __name__ == "__main__":
    main()
