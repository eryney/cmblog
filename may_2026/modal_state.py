"""
STATE in-silico Dabrafenib perturbation: melanoma vs. colorectal (BRAF V600E).

The biological question: can a virtual cell model predict why Dabrafenib kills
BRAF V600E melanoma cells but fails in BRAF V600E colorectal cancer? Same mutation,
different tumor context, opposite clinical outcomes — the canonical cell line false
positive. This is what AUC data alone can't tell you.

Model: STATE (ST-HVG-Tahoe, fewshot) by Arc Institute.
Data:  CellxGene Census scRNA-seq (melanoma skin / colorectal large intestine).

Results saved to Modal Volume "state-results".
Usage:  modal run may_2026/modal_state.py
"""

import modal

# ── Container image ────────────────────────────────────────────────────────────
# Install order matters: arc-state brings numpy>=2.2.6 and torch>=2.7.
# cellxgene-census==1.17.0 requires only numpy>=1.23 (no upper bound), so
# numpy 2.x stays intact. Earlier versions (<=1.16.0) pinned numpy<2.0.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "libhdf5-dev", "pkg-config", "build-essential")
    .run_commands(
        "git lfs install",
        "pip install arc-state huggingface_hub --quiet",
    )
    .pip_install(
        "cellxgene-census==1.17.0",
        "tiledbsoma>=1.15.0",
        "anndata>=0.11.4",
        "scanpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "pandas",
    )
)

volume       = modal.Volume.from_name("state-results",       create_if_missing=True)
model_volume = modal.Volume.from_name("state-model-weights", create_if_missing=True)
app          = modal.App("state-braf-inference", image=image)

# HuggingFace repo and local paths
HF_REPO      = "arcinstitute/ST-HVG-Tahoe"
HF_SUBDIR    = "fewshot/state_generalization_X_hvg"
LOCAL_MODEL  = f"/models/{HF_SUBDIR}"

# Perturbation labels matching the Tahoe dataset convention
DMSO         = "DMSO_TF"
DABRAFENIB   = "[('Dabrafenib', 0.5, 'uM')]"
ERLOTINIB    = "[('Erlotinib', 0.5, 'uM')]"


# ── Download model weights (checkpoint + metadata only, not training data) ─────
@app.function(
    volumes={"/models": model_volume},
    timeout=7200,
    memory=8192,
)
def download_model():
    from huggingface_hub import hf_hub_download
    import os

    os.makedirs(LOCAL_MODEL, exist_ok=True)

    # Only download the files needed for inference; the 285 GB training data
    # lives in subdirectories we're not touching.
    # eval_best.ckpt is a *directory* with pre-computed results, not a checkpoint.
    # The actual PyTorch weights live in checkpoints/ (1.07 GB each).
    model_files = [
        f"{HF_SUBDIR}/config.yaml",
        f"{HF_SUBDIR}/var_dims.pkl",
        f"{HF_SUBDIR}/pert_onehot_map.pt",
        f"{HF_SUBDIR}/cell_type_onehot_map.pkl",
        f"{HF_SUBDIR}/batch_onehot_map.pkl",
        f"{HF_SUBDIR}/data_module.torch",
        f"{HF_SUBDIR}/checkpoints/best.ckpt",   # 1.07 GB
    ]

    for hf_path in model_files:
        dest = f"/models/{hf_path}"
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"  cached: {os.path.basename(hf_path)}")
            continue
        print(f"  downloading: {hf_path}  ...", end=" ", flush=True)
        hf_hub_download(repo_id=HF_REPO, filename=hf_path, local_dir="/models/")
        mb = os.path.getsize(dest) / 1e6
        print(f"{mb:.0f} MB")

    model_volume.commit()
    print("Model ready at", LOCAL_MODEL)
    return LOCAL_MODEL


# ── Main STATE inference (T4 GPU) ──────────────────────────────────────────────
@app.function(
    gpu="T4",
    timeout=3600,
    memory=32768,
    volumes={
        "/results": volume,
        "/models":  model_volume,
    },
)
def run_state_inference():
    import pickle, os, subprocess
    import numpy as np
    import pandas as pd
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp
    import cellxgene_census

    print("=== Loading STATE HVG vocabulary ===")
    with open(f"{LOCAL_MODEL}/var_dims.pkl", "rb") as f:
        d = pickle.load(f)
    hvg_names = list(d['gene_names'][:d['hvg_dim']])
    print(f"  {len(hvg_names)} HVGs expected, e.g. {hvg_names[:4]}")
    hvg_set = set(hvg_names)

    print("\n=== Fetching baseline scRNA-seq from CellxGene Census ===")
    os.makedirs("/results", exist_ok=True)
    np.random.seed(42)

    def fetch_baseline(obs_filter, label, n_sample=300):
        with cellxgene_census.open_soma() as census:
            adata = cellxgene_census.get_anndata(
                census,
                organism="homo_sapiens",
                obs_value_filter=obs_filter,
                obs_column_names=["cell_type", "tissue_general", "disease"],
                var_column_names=["feature_id", "feature_name"],
            )
        print(f"  {label} raw: {adata.shape}")
        if adata.n_obs == 0:
            raise RuntimeError(f"0 cells returned for {label} — check obs_value_filter")
        print(f"  {label} disease dist: {adata.obs['disease'].value_counts().head(5).to_dict()}")
        if adata.n_obs > n_sample:
            idx = np.random.choice(adata.n_obs, n_sample, replace=False)
            adata = adata[idx].copy()
        return adata

    mel = fetch_baseline(
        "tissue_general == 'skin of body' "
        "and disease == 'melanoma' "
        "and is_primary_data == True",
        "melanoma",
    )
    crc = fetch_baseline(
        "tissue_general == 'large intestine' "
        "and disease == 'colorectal cancer' "
        "and is_primary_data == True",
        "colorectal",
    )

    print("\n=== Preprocessing (normalize → log1p) ===")
    for adata, name in [(mel, "melanoma"), (crc, "colorectal")]:
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"  {name}: {adata.shape}, var_names[0]={adata.var_names[0]}")

    def align_to_hvg(adata, hvg_list, name):
        """Extract expression for STATE's exact 2000 HVGs; zero-fill missing genes."""
        # Census var_names are Ensembl IDs; symbols are in var["feature_name"]
        if "feature_name" in adata.var.columns:
            sym_map = {sym: i for i, sym in enumerate(adata.var["feature_name"].values)}
        else:
            sym_map = {sym: i for i, sym in enumerate(adata.var_names)}

        X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
        X = X.astype(np.float32)

        X_hvg = np.zeros((adata.n_obs, len(hvg_list)), dtype=np.float32)
        n_found = 0
        for j, gene in enumerate(hvg_list):
            if gene in sym_map:
                X_hvg[:, j] = X[:, sym_map[gene]]
                n_found += 1

        pct = 100 * n_found / len(hvg_list)
        print(f"  {name}: {n_found}/{len(hvg_list)} HVGs matched ({pct:.0f}%)")
        return X_hvg

    mel_hvg = align_to_hvg(mel, hvg_names, "melanoma")
    crc_hvg = align_to_hvg(crc, hvg_names, "colorectal")

    def make_state_input(expr_hvg, hvg_list, label, drug_label):
        """Build h5ad with DMSO controls + drug-labeled clones for a given drug."""
        n = expr_hvg.shape[0]
        var_df = pd.DataFrame(index=pd.Index(hvg_list, name="gene"))

        obs_ctrl = pd.DataFrame({
            "drugname_drugconc": [DMSO] * n,
            "cell_type":         [label] * n,
            "plate":             ["census_batch"] * n,
        })
        obs_drug = obs_ctrl.copy()
        obs_drug["drugname_drugconc"] = drug_label

        ctrl = ad.AnnData(X=expr_hvg.copy(), obs=obs_ctrl, var=var_df)
        ctrl.obsm["X_hvg"] = expr_hvg.copy()

        drug = ad.AnnData(X=expr_hvg.copy(), obs=obs_drug, var=var_df)
        drug.obsm["X_hvg"] = expr_hvg.copy()

        combined = ad.concat([ctrl, drug], axis=0)
        combined.obs_names_make_unique()
        return combined

    # Dabrafenib inputs (BRAF V600E inhibitor — works in melanoma, fails in CRC)
    mel_dab = make_state_input(mel_hvg, hvg_names, "melanoma",   DABRAFENIB)
    crc_dab = make_state_input(crc_hvg, hvg_names, "colorectal", DABRAFENIB)
    mel_dab.write_h5ad("/results/mel_dab_input.h5ad")
    crc_dab.write_h5ad("/results/crc_dab_input.h5ad")

    # Erlotinib inputs (EGFR inhibitor — failed Phase 2 in melanoma; EGFR not
    # a dependency there. CRC does use EGFR signaling.)
    mel_erl = make_state_input(mel_hvg, hvg_names, "melanoma",   ERLOTINIB)
    crc_erl = make_state_input(crc_hvg, hvg_names, "colorectal", ERLOTINIB)
    mel_erl.write_h5ad("/results/mel_erl_input.h5ad")
    crc_erl.write_h5ad("/results/crc_erl_input.h5ad")

    print(f"\n  mel_dab_input: {mel_dab.shape}  |  crc_dab_input: {crc_dab.shape}")
    print(f"  mel_erl_input: {mel_erl.shape}  |  crc_erl_input: {crc_erl.shape}")

    print("\n=== Running STATE inference ===")
    which = subprocess.run(["which", "state"], capture_output=True, text=True)
    print(f"  state CLI: {which.stdout.strip() or 'NOT FOUND'}")

    ckpt = f"{LOCAL_MODEL}/checkpoints/best.ckpt"

    for label, in_path, out_path in [
        ("melanoma/dabrafenib",   "/results/mel_dab_input.h5ad", "/results/mel_dab_predicted.h5ad"),
        ("colorectal/dabrafenib", "/results/crc_dab_input.h5ad", "/results/crc_dab_predicted.h5ad"),
        ("melanoma/erlotinib",    "/results/mel_erl_input.h5ad", "/results/mel_erl_predicted.h5ad"),
        ("colorectal/erlotinib",  "/results/crc_erl_input.h5ad", "/results/crc_erl_predicted.h5ad"),
    ]:
        print(f"\n  → {label}")
        cmd = [
            "state", "tx", "infer",
            "--model-dir",    LOCAL_MODEL,
            "--checkpoint",   ckpt,
            "--adata",        in_path,
            "--embed-key",    "X_hvg",
            "--pert-col",     "drugname_drugconc",
            "--control-pert", DMSO,
            "--output",       out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.stdout:
            print(result.stdout[-2000:])
        if result.returncode != 0:
            print("STDERR:", result.stderr[-3000:])
            raise RuntimeError(f"STATE failed for {label} (exit {result.returncode})")
        print(f"  ✓ {label} → {out_path}")

    with open("/results/hvg_names.pkl", "wb") as f:
        pickle.dump(hvg_names, f)

    volume.commit()
    print("\n=== Done. Results in /results ===")
    return "done"


# ── List files in volume ───────────────────────────────────────────────────────
@app.function(volumes={"/results": volume})
def list_results():
    import os
    rows = []
    for root, _, fnames in os.walk("/results"):
        for f in fnames:
            p = os.path.join(root, f)
            rows.append(f"{p}  ({os.path.getsize(p)/1e6:.1f} MB)")
    return rows


# ── Local entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("Step 1 — downloading STATE model weights (once, ~500 MB)...")
    download_model.remote()

    print("\nStep 2 — running STATE inference on T4 GPU...")
    result = run_state_inference.remote()
    print(f"Inference result: {result}")

    print("\nFiles saved to volume 'state-results':")
    for f in list_results.remote():
        print(f"  {f}")

    print("\nTo pull results locally, for each of:")
    for f in ["mel_dab_input","crc_dab_input","mel_dab_predicted","crc_dab_predicted",
              "mel_erl_input","crc_erl_input","mel_erl_predicted","crc_erl_predicted","hvg_names"]:
        ext = ".pkl" if f == "hvg_names" else ".h5ad"
        print(f"  modal volume get state-results {f}{ext} /tmp/state_results_dir/{f}{ext}")
    print("\nThen: python may_2026/state_plot.py")
