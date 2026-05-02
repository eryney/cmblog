# cmblog

Computational work behind [Ask Almost a Doctor](https://corememorycm.substack.com/), a biology and medicine blog. Each monthly folder contains the code for one post: Modal scripts for cloud GPU inference, local plotting scripts, and the final figures.

Everything here runs on free-tier infrastructure and public datasets, so you can reproduce it.

---

## May 2026: Can a virtual cell model predict drug outcomes?

**The question:** Dabrafenib targets the BRAF V600E mutation. It works in melanoma — ~50% response rate, FDA-approved. In colorectal cancer with the same mutation, it basically fails (~5% ORR). Could a foundation model have predicted that difference from first principles, before any trial?

**The model:** [STATE](https://github.com/ArcInstitute/state), built by the Arc Institute. Trained on the Tahoe drug perturbation dataset. Give it a cell's baseline gene expression profile across 2000 highly variable genes and a drug label, and it predicts the post-treatment expression profile.

**The data:** Baseline single-cell RNA-seq pulled from [CellxGene Census](https://chanzuckerberg.github.io/cellxgene-census/) — 300 melanoma cells (skin of body, primary tumors) and 300 colorectal cancer cells (large intestine, primary tumors). No CCLE, no proprietary data.

### Files

| File | What it does |
|------|--------------|
| `may_2026/modal_state.py` | Runs STATE inference on Modal (T4 GPU, free tier). Downloads model weights from HuggingFace, fetches cells from CellxGene Census, runs predictions for Dabrafenib and Erlotinib in both cancer types. |
| `may_2026/state_plot.py` | Generates the 3-panel figure from the saved results. |
| `shared/cm_style.py` | Shared matplotlib style (Core Memory palette). |

### Reproducing it

**Step 1 — Run inference on Modal (once, ~30 min, free tier):**

```bash
pip install modal
modal token new        # authenticate
modal run may_2026/modal_state.py
```

This downloads the STATE checkpoint (~1 GB) into a persistent Modal volume and runs inference for all four conditions. Results save to a Modal volume called `state-results`.

**Step 2 — Pull results locally:**

```bash
mkdir -p /tmp/state_results_dir
for f in mel_dab_input mel_dab_predicted crc_dab_input crc_dab_predicted \
          mel_erl_input mel_erl_predicted crc_erl_input crc_erl_predicted hvg_names; do
  ext=".h5ad"; [ "$f" = "hvg_names" ] && ext=".pkl"
  modal volume get state-results ${f}${ext} /tmp/state_results_dir/${f}${ext}
done
```

**Step 3 — Plot (requires numpy < 2 due to h5py ABI):**

```bash
python3 -m venv /tmp/state_plot_venv
/tmp/state_plot_venv/bin/pip install "numpy<2" h5py "anndata>=0.10" \
    matplotlib seaborn scipy pandas adjustText
/tmp/state_plot_venv/bin/python may_2026/state_plot.py
```

Output: `may_2026/STATE_Final_Scientific_Rigor_v6.png`

### Dependencies

- [Modal](https://modal.com) — serverless GPU inference (free tier sufficient)
- [arc-state](https://github.com/ArcInstitute/state) — STATE model and CLI
- [cellxgene-census](https://github.com/chanzuckerberg/cellxgene-census) — CellxGene Census Python API
- Model weights: `arcinstitute/ST-HVG-Tahoe` on HuggingFace (downloaded automatically)

---

## Structure

```
/shared      reusable style and utility code
/may_2026    May 2026 — STATE virtual cell experiment
```
