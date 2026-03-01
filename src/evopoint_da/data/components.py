import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import freesasa
import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1
from sklearn.decomposition import PCA

STANDARD_AA = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
}


class StructureParser:
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)

    def _get_structure(self, file_path: str):
        parser = self.pdb_parser if file_path.endswith(".pdb") else self.cif_parser
        return parser.get_structure("protein", file_path)

    def parse_ca_structure(self, file_path: str, chain_id: Optional[str] = None) -> Optional[Dict]:
        try:
            structure = self._get_structure(file_path)
            model = next(structure.get_models())
        except Exception:
            return None

        coords, plddts, residue_ids, seq_chars = [], [], [], []
        for chain in model:
            if chain_id and chain.id != chain_id:
                continue
            for res in chain:
                resname = res.get_resname().strip().upper()
                if res.id[0] != " " or resname not in STANDARD_AA or "CA" not in res:
                    continue
                ca = res["CA"]
                coords.append(ca.get_coord())
                plddts.append(float(ca.get_bfactor()))
                residue_ids.append(f"{chain.id}_{res.id[1]}")
                try:
                    seq_chars.append(seq1(resname))
                except Exception:
                    seq_chars.append("X")

        if not coords:
            return None

        return {
            "coords": np.asarray(coords, dtype=np.float32),
            "plddts": np.asarray(plddts, dtype=np.float32),
            "residue_ids": residue_ids,
            "sequence": "".join(seq_chars),
        }


def kabsch_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_c = src - src.mean(axis=0, keepdims=True)
    dst_c = dst - dst.mean(axis=0, keepdims=True)
    h = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return src_c @ r + dst.mean(axis=0, keepdims=True)


def compute_displacement_target(
    af2_coords: np.ndarray,
    holo_coords: np.ndarray,
    residue_ids_af2: List[str],
    residue_ids_holo: List[str],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    holo_map = {rid: i for i, rid in enumerate(residue_ids_holo)}
    pairs = [(i, holo_map[rid], rid) for i, rid in enumerate(residue_ids_af2) if rid in holo_map]
    if not pairs:
        raise ValueError("No residue overlap between AF2 and holo structures.")

    af2_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    holo_idx = np.array([p[1] for p in pairs], dtype=np.int64)
    common_ids = [p[2] for p in pairs]

    af2_sub = af2_coords[af2_idx]
    holo_sub = holo_coords[holo_idx]
    af2_aligned = kabsch_align(af2_sub, holo_sub)
    delta_r = holo_sub - af2_aligned
    return delta_r.astype(np.float32), common_ids, af2_aligned.astype(np.float32)


def parse_pae_matrix(pae_path: Optional[str], n: int) -> np.ndarray:
    if pae_path is None or not os.path.exists(pae_path):
        return np.zeros((n, n), dtype=np.float32)
    if pae_path.endswith(".npy"):
        pae = np.load(pae_path)
    else:
        with open(pae_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "predicted_aligned_error" in raw:
            pae = np.asarray(raw["predicted_aligned_error"], dtype=np.float32)
        else:
            pae = np.asarray(raw, dtype=np.float32)
    return pae.astype(np.float32)


def build_knn_edges(pos: torch.Tensor, k: int = 16, pae: Optional[np.ndarray] = None):
    n = pos.shape[0]
    dist = torch.cdist(pos, pos)
    knn_idx = dist.topk(k=min(k + 1, n), largest=False).indices[:, 1:]

    row = torch.arange(n, device=pos.device).unsqueeze(1).repeat(1, knn_idx.shape[1]).reshape(-1)
    col = knn_idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)

    edge_dist = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
    if pae is None:
        edge_pae = torch.zeros_like(edge_dist)
    else:
        pae_t = torch.as_tensor(pae, dtype=pos.dtype, device=pos.device)
        edge_pae = pae_t[row, col].unsqueeze(1)
    edge_attr = torch.cat([edge_dist, edge_pae], dim=1)
    return edge_index, edge_attr


class ESMFeatureExtractor:
    """Lightweight wrapper that reuses the original script contract while avoiding hard dependency at import."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        try:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein
            from esm.tokenization import EsmSequenceTokenizer
        except Exception as e:
            raise RuntimeError(f"ESM package unavailable: {e}")

        self.ESMProtein = ESMProtein
        self.tokenizer = EsmSequenceTokenizer()
        self.model = ESMC(tokenizer=self.tokenizer, d_model=1152, n_layers=36, n_heads=18).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.eval()

    @torch.no_grad()
    def extract_residue_embeddings(self, sequence: str) -> torch.Tensor:
        sequence = sequence[:1022]
        protein = self.ESMProtein(sequence=sequence)
        tokenized = self.model.encode(protein).sequence.unsqueeze(0).to(self.device)
        out = self.model(tokenized)
        return out.embeddings[0, 1:-1].cpu()


class PCAReducer:
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, data_list: List[torch.Tensor]):
        x = torch.cat(data_list, dim=0).numpy()
        self.pca.fit(x)
        self.is_fitted = True

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted")
        return torch.from_numpy(self.pca.transform(x.numpy())).float()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.pca, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.pca = pickle.load(f)
        self.is_fitted = True


def compute_sasa_with_freesasa(structure_path: str) -> Dict[str, float]:
    print(f"[DEBUG] Starting FreeSASA calculation for: {structure_path}")
    structure = freesasa.Structure(structure_path)
    print("[DEBUG] Structure loaded, running SASA calculation via Python API...")
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()

    per_res = {}
    for chain_id, residues in residue_areas.items():
        for res_id, residue_area in residues.items():
            key = f"{chain_id}_{int(res_id)}"
            per_res[key] = float(residue_area.total)

    print(f"[DEBUG] FreeSASA finished, parsed {len(per_res)} residue SASA entries.")
    return per_res
