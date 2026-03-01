import argparse
import glob
import os
import torch

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
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.pair_dir, "*.pt")))
    extractor = ESMFeatureExtractor(model_path=args.esm_weights)
    pca = PCAReducer(n_components=args.pca_dim)

    if args.fit_pca:
        buf = []
        for f in files:
            d = torch.load(f, weights_only=False)
            buf.append(extractor.extract_residue_embeddings(d["sequence"]))
        pca.fit(buf)
        pca.save(args.pca_path)
    else:
        pca.load(args.pca_path)

    for f in files:
        d = torch.load(f, weights_only=False)
        stem = d["pair_id"]
        emb = extractor.extract_residue_embeddings(d["sequence"])
        x_esm = pca.transform(emb)

        af2_file = os.path.join(args.af2_structure_dir, f"{stem}.pdb")
        sasa_map = compute_sasa_with_freesasa(af2_file)
        sasa = torch.tensor([sasa_map.get(rid, 0.0) for rid in d["residue_ids"]], dtype=torch.float32).unsqueeze(1)
        plddt = d["plddt"].float()
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
