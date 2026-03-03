# src/evopoint_da/data/components.py

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import freesasa
import numpy as np
import torch
from Bio import Align
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

STANDARD_AA = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
}

class StructureParser:
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)

    def _get_structure(self, file_path: str):
        lower_path = file_path.lower()
        parser = self.pdb_parser if lower_path.endswith((".pdb", ".ent")) else self.cif_parser
        return parser.get_structure("protein", file_path)

    def parse_ca_structure(self, file_path: str) -> Optional[Dict[str, Dict]]:
        try:
            structure = self._get_structure(file_path)
            model = next(structure.get_models())
        except Exception:
            return None

        chains_data = {}
        for chain in model:
            coords, plddts, residue_ids, residue_names, seq_chars = [], [], [], [], []
            for res in chain:
                resname = res.get_resname().strip().upper()
                if res.id[0] != " " or resname not in STANDARD_AA or "CA" not in res:
                    continue
                ca = res["CA"]
                coords.append(ca.get_coord())
                plddts.append(float(ca.get_bfactor()))
                
                # Full ID: Chain + ResNum + InsertionCode
                ins_code = res.id[2].strip()
                full_id = f"{chain.id}_{res.id[1]}{ins_code}"
                
                residue_ids.append(full_id)
                residue_names.append(resname)
                try:
                    seq_chars.append(seq1(resname))
                except Exception:
                    seq_chars.append("X")
            
            # Filter extremely short fragments/peptides
            if len(coords) < 15:
                continue

            chains_data[chain.id] = {
                "coords": np.asarray(coords, dtype=np.float32),
                "plddts": np.asarray(plddts, dtype=np.float32),
                "residue_ids": residue_ids,
                "residue_names": residue_names,
                "sequence": "".join(seq_chars),
            }
        return chains_data if chains_data else None

def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate rotation (R) and translation (t) from P to Q."""
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    t = Q_mean - P_mean @ R.T
    return R, t

def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return coords @ R.T + t

def iterative_kabsch(af2_coords: np.ndarray, holo_coords: np.ndarray, max_iter: int = 5, trim_ratio: float = 0.5) -> np.ndarray:
    """
    Robust alignment that ignores flexible loops/outliers to find the 'core' alignment.
    This ensures that delta_r vectors reflect true conformational changes relative to the stable core.
    """
    assert len(af2_coords) == len(holo_coords)
    curr_af2 = af2_coords.copy()
    
    # Initial global alignment
    R, t = kabsch_rotation(curr_af2, holo_coords)
    best_R, best_t = R, t
    min_core_rmsd = float("inf")
    
    for i in range(max_iter):
        aligned_iter = apply_transform(af2_coords, best_R, best_t)
        dists = np.linalg.norm(aligned_iter - holo_coords, axis=1)
        
        # Keep top X% closest residues (The Core)
        n_core = int(len(dists) * trim_ratio)
        if n_core < 10: n_core = 10 
        
        core_indices = np.argsort(dists)[:n_core]
        
        core_af2 = af2_coords[core_indices]
        core_holo = holo_coords[core_indices]
        
        R_new, t_new = kabsch_rotation(core_af2, core_holo)
        
        core_aligned = apply_transform(core_af2, R_new, t_new)
        core_rmsd = np.sqrt(np.mean(np.sum((core_aligned - core_holo)**2, axis=1)))
        
        if core_rmsd < min_core_rmsd:
            min_core_rmsd = core_rmsd
            best_R, best_t = R_new, t_new
        else:
            break

    # Apply the best CORE transformation to the WHOLE protein
    return apply_transform(af2_coords, best_R, best_t)

def compute_displacement_target(
    af2_chains: Dict[str, Dict],
    holo_chains: Dict[str, Dict],
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray, str]:
    
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -4.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    best_score = -float("inf")
    best_pair = None

    # 1. Chain Selection
    for af2_id, af2_data in af2_chains.items():
        for holo_id, holo_data in holo_chains.items():
            # Length sanity check (avoid matching 1000aa to 50aa)
            len_a = len(af2_data["sequence"])
            len_h = len(holo_data["sequence"])
            if min(len_a, len_h) / max(len_a, len_h) < 0.4:
                continue

            try:
                score = aligner.score(af2_data["sequence"], holo_data["sequence"])
            except:
                continue
            
            if score > best_score:
                best_score = score
                best_pair = (af2_id, holo_id)

    if not best_pair:
        raise ValueError("No alignable chains found.")

    best_af2_id, best_holo_id = best_pair
    seq_af2 = af2_chains[best_af2_id]["sequence"]
    seq_holo = holo_chains[best_holo_id]["sequence"]
    
    # 2. Detailed Alignment
    alignment = next(iter(aligner.align(seq_af2, seq_holo)))
    af2_blocks, holo_blocks = alignment.aligned
    
    af2_idx = []
    holo_idx = []
    for (a_s, a_e), (h_s, h_e) in zip(af2_blocks, holo_blocks):
        l = min(a_e - a_s, h_e - h_s)
        af2_idx.extend(range(a_s, a_s + l))
        holo_idx.extend(range(h_s, h_s + l))

    if len(af2_idx) < 15:
        raise ValueError(f"Alignment too short ({len(af2_idx)} residues).")
    
    # === CRITICAL: Sequence Identity Check ===
    matches = sum(1 for i, j in zip(af2_idx, holo_idx) if seq_af2[i] == seq_holo[j])
    identity = matches / len(af2_idx)
    
    # Threshold: 90% strict identity for aligned region. 
    # Allows for a few mutations/errors, but rejects mismatched isoforms/paralogs.
    if identity < 0.90:
        raise ValueError(f"Low sequence identity ({identity:.2%}). Likely wrong chain or severe mutation.")

    af2_idx = np.asarray(af2_idx, dtype=np.int64)
    holo_idx = np.asarray(holo_idx, dtype=np.int64)

    # 3. Coordinate Processing
    coords_af2_full = af2_chains[best_af2_id]["coords"]
    coords_holo_full = holo_chains[best_holo_id]["coords"]
    
    af2_sub = coords_af2_full[af2_idx]
    holo_sub = coords_holo_full[holo_idx]
    
    # Use Iterative Kabsch to align on the Rigid Core
    # We DO NOT filter by RMSD result, we just use it to get the best superposition.
    af2_aligned = iterative_kabsch(af2_sub, holo_sub, max_iter=5, trim_ratio=0.6)
    
    delta_r = holo_sub - af2_aligned
    
    common_ids = [af2_chains[best_af2_id]["residue_ids"][i] for i in af2_idx]
    
    return delta_r.astype(np.float32), common_ids, af2_aligned.astype(np.float32), af2_idx, holo_idx, best_af2_id


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
        if not self.is_fitted: raise RuntimeError("PCA not fitted")
        return torch.from_numpy(self.pca.transform(x.numpy())).float()
    def save(self, path: str):
        with open(path, "wb") as f: pickle.dump(self.pca, f)
    def load(self, path: str):
        with open(path, "rb") as f: self.pca = pickle.load(f)
        self.is_fitted = True

class ESMFeatureExtractor:
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
        if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
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

def compute_sasa_with_freesasa(structure_path: str) -> Dict[str, float]:
    print(f"[DEBUG] Starting FreeSASA calculation for: {structure_path}")
    structure = freesasa.Structure(structure_path)
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()
    per_res = {}
    for chain_id, residues in tqdm(residue_areas.items(), desc="FreeSASA chains", unit="chain"):
        for res_id, residue_area in tqdm(residues.items(), desc=f"Chain {chain_id} residues", unit="res", leave=False):
            key = f"{chain_id}_{int(res_id)}"
            per_res[key] = float(residue_area.total)
    return per_res

def parse_pae_matrix(pae_path: Optional[str], n: int) -> np.ndarray:
    if pae_path is None or not os.path.exists(pae_path): return np.zeros((n, n), dtype=np.float32)
    if pae_path.lower().endswith(".npy"): pae = np.load(pae_path)
    else:
        with open(pae_path, "r", encoding="utf-8") as f: raw = json.load(f)
        if isinstance(raw, dict) and "predicted_aligned_error" in raw: pae = np.asarray(raw["predicted_aligned_error"], dtype=np.float32)
        else: pae = np.asarray(raw, dtype=np.float32)
    return pae.astype(np.float32)

def build_knn_edges(pos: torch.Tensor, k: int = 16, pae: Optional[np.ndarray] = None):
    n = pos.shape[0]
    dist = torch.cdist(pos, pos)
    knn_idx = dist.topk(k=min(k + 1, n), largest=False).indices[:, 1:]
    row = torch.arange(n, device=pos.device).unsqueeze(1).repeat(1, knn_idx.shape[1]).reshape(-1)
    col = knn_idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    edge_dist = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
    if pae is None: edge_pae = torch.zeros_like(edge_dist)
    else:
        pae_t = torch.as_tensor(pae, dtype=pos.dtype, device=pos.device)
        edge_pae = pae_t[row, col].unsqueeze(1)
    edge_attr = torch.cat([edge_dist, edge_pae], dim=1)
    return edge_index, edge_attr
