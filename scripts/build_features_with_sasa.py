import argparse
import glob
import json
import os
import torch
from tqdm import tqdm


PLDDT_SCALE_MAX = 100.0
SASA_SCALE_MAX = 250.0

from evopoint_da.data.components import (
    ESMFeatureExtractor,
    PCAReducer,
    build_knn_edges,
    compute_sasa_with_freesasa,
    parse_pae_matrix,
)


def get_args():
    p = argparse.ArgumentParser(description="Add ESM/PCA + pLDDT + SASA + KNN/PAE edge features.")
    p.add_argument("--pair_dir", default="data/processed_pairs")
    p.add_argument("--output_dir", default="data/processed_graphs")
    p.add_argument("--esm_weights", required=True)
    p.add_argument("--pca_path", default="data/pca_esmc_128.pkl")
    p.add_argument("--pca_dim", type=int, default=128)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--fit_pca", action="store_true")
    p.add_argument("--pae_dir", default="data/raw_af2")
    p.add_argument("--af2_structure_dir", default="data/raw_af2")
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
        if stem.startswith("AF-"):
            uniprot_id = stem[3:]
        else:
            uniprot_id = stem
        uniprot_to_path[uniprot_id] = af2_path
    return uniprot_to_path


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.pair_dir, "*.pt")))
    extractor = ESMFeatureExtractor(model_path=args.esm_weights)
    pca = PCAReducer(n_components=args.pca_dim)
    pdb_to_uniprot = load_pdb_to_uniprot_mapping(args.mapping_file)
    uniprot_to_af2_path = build_uniprot_to_af2_path(args.af2_structure_dir)
    af2_casefold = {k.lower(): v for k, v in uniprot_to_af2_path.items()}

    if args.fit_pca:
        buf = []
        for f in tqdm(files, desc="Fitting PCA", unit="file"):
            d = torch.load(f, weights_only=False)
            buf.append(extractor.extract_residue_embeddings(d["sequence"]))
        pca.fit(buf)
        pca.save(args.pca_path)
    else:
        pca.load(args.pca_path)

    for f in tqdm(files, desc="Building graph features", unit="file"):
        d = torch.load(f, weights_only=False)
        stem = d["pair_id"]
        emb = extractor.extract_residue_embeddings(d["sequence"])
        x_esm = pca.transform(emb)

        uniprot_id = pdb_to_uniprot.get(stem.upper())
        af2_file = None
        if uniprot_id:
            af2_file = uniprot_to_af2_path.get(uniprot_id)
            if not af2_file:
                af2_file = af2_casefold.get(uniprot_id.lower())

        if not af2_file:
            print(f"[debug] Skip {stem}: AF2 structure not found via mapping file")
            continue

        sasa_map = compute_sasa_with_freesasa(af2_file)
        sasa_raw = torch.tensor([sasa_map.get(rid, 0.0) for rid in d["residue_ids"]], dtype=torch.float32).unsqueeze(1)
        sasa = (sasa_raw / SASA_SCALE_MAX).clamp(0.0, 1.0)

        plddt_raw = d["plddt"].float()
        plddt = (plddt_raw / PLDDT_SCALE_MAX).clamp(0.0, 1.0)
        x = torch.cat([x_esm, plddt, sasa], dim=1)

        pae_path_npy = os.path.join(args.pae_dir, f"{stem}.npy")
        pae_path_json = os.path.join(args.pae_dir, f"{stem}.json")
        pae = parse_pae_matrix(pae_path_npy if os.path.exists(pae_path_npy) else pae_path_json, len(d["residue_ids"]))

        edge_index, edge_attr = build_knn_edges(d["af2_pos"].float(), k=args.k, pae=pae)

        out = {
            "pair_id": stem,
            "residue_ids": d["residue_ids"],
            "x": x,
            "pos": d["af2_pos"].float(),
            "y_delta": d["y_delta"].float(),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }
        torch.save(out, os.path.join(args.output_dir, f"{stem}.pt"))

    print(f"Processed {len(files)} graph files into {args.output_dir}")


if __name__ == "__main__":
    main()
