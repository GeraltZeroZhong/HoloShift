import argparse
import glob
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from evopoint_da.data.components import ESMFeatureExtractor, PCAReducer, StructureParser


def get_args():
    parser = argparse.ArgumentParser(description="Extract ESM/PCA features for CA residues.")
    parser.add_argument("--data_dir", type=str, required=True, help="Input PDB/CIF directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output .pt directory")
    parser.add_argument("--model_name", type=str, required=True, help="Local ESM-C weights path")
    parser.add_argument("--pca_model_path", type=str, default="data/pca_esmc_128.pkl")
    parser.add_argument("--fit_pca", action="store_true")
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--is_af2", action="store_true", help="Normalize pLDDT into [0,1]")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_structure(struct_parser: StructureParser, path: str):
    parsed = struct_parser.parse_ca_structure(path)
    if not parsed:
        return None
    max_len = 1022
    if len(parsed["sequence"]) > max_len:
        parsed["sequence"] = parsed["sequence"][:max_len]
        parsed["coords"] = parsed["coords"][:max_len]
        parsed["plddts"] = parsed["plddts"][:max_len]
        parsed["residue_ids"] = parsed["residue_ids"][:max_len]
    return parsed


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pdb"))) + sorted(glob.glob(os.path.join(args.data_dir, "*.cif")))
    if not raw_files:
        print(f"No files found in {args.data_dir}")
        return

    esm_extractor = ESMFeatureExtractor(model_path=args.model_name, device=args.device)
    struct_parser = StructureParser()
    pca_reducer = PCAReducer(n_components=args.pca_dim)

    if args.fit_pca:
        print(f"Fitting PCA (Target Dim: {args.pca_dim})...")
        sample_files = np.random.choice(raw_files, min(len(raw_files), 500), replace=False)
        buffer = []
        for fpath in tqdm(sample_files, desc="PCA Sampling"):
            parsed = load_structure(struct_parser, fpath)
            if not parsed:
                continue
            try:
                emb = esm_extractor.extract_residue_embeddings(parsed["sequence"])
                buffer.append(emb)
            except Exception as e:
                print(f"PCA Fit Skip {fpath}: {e}")
        if not buffer:
            print("Error: No valid data extracted for PCA fitting.")
            return
        pca_reducer.fit(buffer)
        pca_reducer.save(args.pca_model_path)
        print(f"PCA model saved to {args.pca_model_path}")
    else:
        if not os.path.exists(args.pca_model_path):
            print(f"PCA model not found at {args.pca_model_path}. Please run with --fit_pca first.")
            return
        pca_reducer.load(args.pca_model_path)
        print(f"Loaded PCA model from {args.pca_model_path}")

    print(f"Processing {len(raw_files)} files (Is AF2: {args.is_af2})...")
    success_count = 0
    for fpath in tqdm(raw_files, desc="Processing"):
        file_id = os.path.splitext(os.path.basename(fpath))[0]
        output_path = os.path.join(args.output_dir, f"{file_id}.pt")

        parsed = load_structure(struct_parser, fpath)
        if not parsed:
            continue

        try:
            raw_emb = esm_extractor.extract_residue_embeddings(parsed["sequence"])
            reduced_emb = pca_reducer.transform(raw_emb)
            plddt = torch.from_numpy(parsed["plddts"]).float()
            if args.is_af2:
                plddt = plddt / 100.0
            payload = {
                "pos": torch.from_numpy(parsed["coords"]).float(),
                "x": reduced_emb.float(),
                "plddt": plddt.unsqueeze(1),
                "residue_ids": parsed["residue_ids"],
                "sequence": parsed["sequence"],
            }
            torch.save(payload, output_path)
            success_count += 1
        except Exception as e:
            print(f"Failed {fpath}: {e}")

    print(f"Done. Processed {success_count}/{len(raw_files)} files.")


if __name__ == "__main__":
    main()
