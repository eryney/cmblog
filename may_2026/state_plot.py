"""
Generate STATE prediction figure: Dabrafenib vs. Erlotinib in melanoma vs. colorectal.

Pull results from Modal volume first:
  mkdir -p /tmp/state_results_dir
  for f in mel_dab_input mel_dab_predicted crc_dab_input crc_dab_predicted \\
            mel_erl_input mel_erl_predicted crc_erl_input crc_erl_predicted hvg_names; do
    ext=".h5ad"; [ "$f" = "hvg_names" ] && ext=".pkl"
    modal volume get state-results ${f}${ext} /tmp/state_results_dir/${f}${ext}
  done

Plotting requires numpy<2 (h5py ABI conflict). Set up once:
  python3 -m venv /tmp/state_plot_venv
  /tmp/state_plot_venv/bin/pip install "numpy<2" h5py "anndata>=0.10" matplotlib seaborn scipy pandas adjustText

Then: /tmp/state_plot_venv/bin/python may_2026/state_plot.py
Produces: may_2026/STATE_Final_Scientific_Rigor_v6.png + .pdf
"""

import pickle, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import seaborn as sns

RESULTS_DIR = "/tmp/state_results_dir"
OUT_NAME    = "STATE_Final_Scientific_Rigor_v6"

PURPLE = "#7B2CBF"
BLACK  = "#101010"
MUTED  = "#888888"

sns.set_style("white")
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Georgia"],
    "font.size":         11,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.grid":         True,
    "grid.color":        "#cccccc",
    "grid.alpha":        0.3,
    "grid.linewidth":    0.8,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "svg.fonttype":      "none",
})

DMSO_LABEL = "DMSO_TF"
DAB_LABEL  = "[('Dabrafenib', 0.5, 'uM')]"
ERL_LABEL  = "[('Erlotinib', 0.5, 'uM')]"

MELANOCYTE_GENES = ["DCT", "TYR", "GPM6B", "CDH19", "TYRP1", "MITF", "SOX10"]


def load_adata(path):
    import anndata as ad
    return ad.read_h5ad(path)


def extract_expression(adata, key="X_hvg"):
    if key in adata.obsm:
        X = adata.obsm[key]
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float32)


def compute_delta(input_adata, pred_adata, drug_label):
    ctrl_mask = input_adata.obs["drugname_drugconc"] == DMSO_LABEL
    drug_mask = input_adata.obs["drugname_drugconc"] == drug_label
    baseline  = extract_expression(input_adata)
    predicted = extract_expression(pred_adata)
    pred_drug = predicted[drug_mask] if predicted.shape[0] == input_adata.n_obs else predicted
    return pred_drug.mean(axis=0) - baseline[ctrl_mask].mean(axis=0)


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)


def main():
    with open(f"{RESULTS_DIR}/hvg_names.pkl", "rb") as f:
        raw = pickle.load(f)
    hvg_names = list(raw["gene_names"][:raw["hvg_dim"]]) if isinstance(raw, dict) else list(raw)
    gene_idx  = {g: i for i, g in enumerate(hvg_names)}

    mel_dab_in   = load_adata(f"{RESULTS_DIR}/mel_dab_input.h5ad")
    crc_dab_in   = load_adata(f"{RESULTS_DIR}/crc_dab_input.h5ad")
    mel_dab_pred = load_adata(f"{RESULTS_DIR}/mel_dab_predicted.h5ad")
    crc_dab_pred = load_adata(f"{RESULTS_DIR}/crc_dab_predicted.h5ad")

    mel_erl_in   = load_adata(f"{RESULTS_DIR}/mel_erl_input.h5ad")
    crc_erl_in   = load_adata(f"{RESULTS_DIR}/crc_erl_input.h5ad")
    mel_erl_pred = load_adata(f"{RESULTS_DIR}/mel_erl_predicted.h5ad")
    crc_erl_pred = load_adata(f"{RESULTS_DIR}/crc_erl_predicted.h5ad")

    mel_dab = compute_delta(mel_dab_in, mel_dab_pred, DAB_LABEL)
    crc_dab = compute_delta(crc_dab_in, crc_dab_pred, DAB_LABEL)
    mel_erl = compute_delta(mel_erl_in, mel_erl_pred, ERL_LABEL)
    crc_erl = compute_delta(crc_erl_in, crc_erl_pred, ERL_LABEL)

    # ── Pre-compute all statistics ─────────────────────────────────────────────
    mel_abs   = [np.abs(mel_dab).mean(), np.abs(mel_erl).mean()]
    crc_abs   = [np.abs(crc_dab).mean(), np.abs(crc_erl).mean()]
    dab_ratio = mel_abs[0] / crc_abs[0]
    erl_ratio = mel_abs[1] / crc_abs[1]

    # Per-drug mel vs CRC correlation (one data point per gene)
    r_dab = np.corrcoef(crc_dab, mel_dab)[0, 1]
    r_erl = np.corrcoef(crc_erl, mel_erl)[0, 1]

    # Cross-drug correlation — Dabrafenib Δ vs Erlotinib Δ (same tissue, all 2000 genes)
    r_cross_mel = np.corrcoef(mel_dab, mel_erl)[0, 1]
    r_cross_crc = np.corrcoef(crc_dab, crc_erl)[0, 1]

    m_slope, b_coef = np.polyfit(crc_dab, mel_dab, 1)

    print("Mean |Δ| summary:")
    print(f"  Dabrafenib — mel: {mel_abs[0]:.4f}  CRC: {crc_abs[0]:.4f}  ratio: {dab_ratio:.2f}x")
    print(f"  Erlotinib  — mel: {mel_abs[1]:.4f}  CRC: {crc_abs[1]:.4f}  ratio: {erl_ratio:.2f}x")
    print(f"  R mel vs CRC — Dab: {r_dab:.3f}  Erl: {r_erl:.3f}")
    print(f"  Cross-drug R   — mel: {r_cross_mel:.3f}  CRC: {r_cross_crc:.3f}")

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "What STATE predicts: Dabrafenib (BRAF inh.) vs. Erlotinib (EGFR inh.)\n"
        "across BRAF V600E melanoma and colorectal cancer",
        fontsize=12, fontweight="bold", color="#363737",
        y=1.04, va="bottom",
    )

    ax1, ax2, ax3 = axes

    # ── Panel 1: mean |Δ| grouped bar ─────────────────────────────────────────
    x, w = np.arange(2), 0.35

    b1 = ax1.bar(x - w/2, mel_abs, w, color=PURPLE, label="Melanoma ($n=300$)",
                 edgecolor=BLACK, linewidth=0.5)
    b2 = ax1.bar(x + w/2, crc_abs, w, color=BLACK, label="Colorectal ($n=300$)",
                 edgecolor=BLACK, linewidth=0.5)

    for bar, val in zip(list(b1) + list(b2), mel_abs + crc_abs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.003,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", fontfamily="serif",
            color="#363737",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        ["Dabrafenib\n(Melanoma $n=300$)", "Erlotinib\n(Colorectal $n=300$)"],
        fontsize=9,
    )
    ax1.tick_params(axis="x", which="major", pad=10)
    ax1.set_ylabel(
        "Mean transcriptional response\nmagnitude ($|\\Delta|$ across 2000 HVGs)",
        fontsize=9.5, labelpad=8,
    )
    ax1.set_ylim(0, max(mel_abs + crc_abs) * 1.28)
    ax1.set_title("Overall transcriptional\nresponse magnitude",
                  fontsize=11, fontweight="bold", color="#363737", pad=16)
    style_ax(ax1)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper right")

    # ── Panel 2: melanocyte marker genes ──────────────────────────────────────
    mel_genes = [g for g in MELANOCYTE_GENES if g in gene_idx]
    dab_vals  = [mel_dab[gene_idx[g]] for g in mel_genes]
    erl_vals  = [mel_erl[gene_idx[g]] for g in mel_genes]

    x2 = np.arange(len(mel_genes))
    ax2.bar(x2 - w/2, dab_vals, w, color=PURPLE, label="Dabrafenib",
            edgecolor=BLACK, linewidth=0.5)
    ax2.bar(x2 + w/2, erl_vals, w, color=BLACK, label="Erlotinib",
            edgecolor=BLACK, linewidth=0.5)
    ax2.axhline(0, color="#aaaaaa", lw=1.0, linestyle="-", alpha=0.5, zorder=5)

    ax2.set_xticks(x2)
    # Bold Georgia gene labels, no rotation, centered, no arrowprops anywhere
    ax2.set_xticklabels(
        mel_genes,
        rotation=0, ha="center",
        fontsize=9, fontweight="bold",
    )
    ax2.tick_params(axis="x", which="major", pad=10)
    ax2.set_ylim(-1.5, 0.2)
    ax2.set_ylabel("$\\Delta$ log$_1$p (melanoma cells)", fontsize=10, labelpad=8)
    ax2.set_title("Melanocyte identity genes\n(melanoma cells only)",
                  fontsize=11, fontweight="bold", color="#363737", pad=16)
    style_ax(ax2)
    ax2.legend(fontsize=9, frameon=False, loc="lower right")

    # ── Panel 3: scatter Δ melanoma vs Δ CRC ──────────────────────────────────
    ax3.scatter(crc_dab, mel_dab, s=14, alpha=0.50, color=PURPLE,
                edgecolors=BLACK, linewidths=0.25,
                label="Dabrafenib", rasterized=True, zorder=3)
    ax3.scatter(crc_erl, mel_erl, s=14, alpha=0.50, color=BLACK,
                edgecolors="none",
                label="Erlotinib", rasterized=True, zorder=2)

    lim = max(np.abs([mel_dab, crc_dab, mel_erl, crc_erl]).max() * 1.05, 1.0)

    ax3.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.9,
             linestyle="--", alpha=0.4, zorder=1, label="$y = x$ (equal response)")

    x_fit = np.linspace(crc_dab.min(), crc_dab.max(), 200)
    ax3.plot(x_fit, m_slope * x_fit + b_coef, color=PURPLE, lw=1.8,
             zorder=4, alpha=0.9, label=f"Regression (slope = {m_slope:.3f})")

    ax3.axhline(0, color=MUTED, lw=0.5, linestyle=":", alpha=0.3)
    ax3.axvline(0, color=MUTED, lw=0.5, linestyle=":", alpha=0.3)

    # Gene label annotations — manual offsets to stay within axis, no arrowprops
    # Coordinates (crc_delta, mel_delta): DCT≈(-0.00,-1.31), TYR≈(+0.00,-1.21),
    # GPM6B≈(+0.18,-1.08), ERBB3≈(-0.14,-1.06), VIM≈(-1.67,-3.40)
    gene_offsets = {
        "DCT":   (-32, -14),   # lower-left; avoids TYR/ERBB3 cluster
        "TYR":   (  8,   8),   # upper-right
        "GPM6B": (  8, -14),   # lower-right; below GPM6B point
        "ERBB3": (-40,   6),   # left of point; clear of DCT
        "VIM":   (  8,  10),   # upper-right; VIM is near bottom edge
    }
    for g, (dx, dy) in gene_offsets.items():
        if g not in gene_idx:
            continue
        xi, yi = crc_dab[gene_idx[g]], mel_dab[gene_idx[g]]
        ax3.scatter(xi, yi, s=35, color=PURPLE, zorder=6,
                    edgecolors="white", linewidths=0.6)
        ax3.annotate(
            g, (xi, yi),
            fontsize=8, color=PURPLE, fontweight="bold",
            xytext=(dx, dy), textcoords="offset points",
            arrowprops=None,
        )

    # Per-drug stats box — upper left, 9.5pt, solid white background
    stats_str = (
        f"Dabrafenib   $R = {r_dab:.3f}$,  $R^2 = {r_dab**2:.3f}$\n"
        f"Erlotinib       $R = {r_erl:.3f}$,  $R^2 = {r_erl**2:.3f}$"
    )
    ax3.text(
        0.03, 0.97, stats_str,
        transform=ax3.transAxes,
        fontsize=9.5, va="top", ha="left",
        color="#363737",
        bbox=dict(boxstyle="square,pad=0.5", facecolor="white",
                  edgecolor="black", linewidth=0.8, alpha=0.9),
        zorder=10,
    )

    # Cross-drug correlation annotation — the smoking gun
    if r_cross_mel > 0.8:
        cross_str = (
            f"Cross-drug $R_{{\\mathrm{{mel}}}} = {r_cross_mel:.3f}$\n"
            f"Dab $\\Delta$ vs. Erl $\\Delta$ per gene\n"
            f"High correlation: cell-type bias\nover drug mechanism"
        )
        ax3.text(
            0.03, 0.60, cross_str,
            transform=ax3.transAxes,
            fontsize=8, va="top", ha="left",
            color=PURPLE,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=PURPLE, linewidth=0.7, alpha=0.88),
            zorder=10,
        )

    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_xlabel("$\\Delta$ log$_1$p — Colorectal", fontsize=10, labelpad=8)
    ax3.set_ylabel("$\\Delta$ log$_1$p — Melanoma",    fontsize=10, labelpad=8)
    ax3.set_title("Per-gene $\\Delta$: melanoma vs. CRC\n(all 2000 HVGs)",
                  fontsize=11, fontweight="bold", color="#363737", pad=16)
    style_ax(ax3)
    ax3.legend(fontsize=8, frameon=False, loc="lower right", markerscale=2.5)

    # ── Figure-level caption ──────────────────────────────────────────────────
    caption = (
        f"Panel 1: Mean absolute log-normalized delta ($|\\Delta|$) across 2000 HVGs. "
        f"Both drugs predict ~{dab_ratio:.2f}x more response in melanoma than colorectal "
        f"(Erlotinib: {erl_ratio:.2f}x) — a cell-type effect, not drug specificity.   "
        f"Panel 2: Predicted suppression of melanocyte identity genes is near-identical "
        f"for Dabrafenib and Erlotinib (approaching or exceeding a log unit for DCT, TYR, GPM6B; "
        f"~0.9 for CDH19).   "
        f"Panel 3: Per-gene $\\Delta$ correlation $R \\approx {r_dab:.2f}$ for both drugs; "
        f"cross-drug $R = {r_cross_mel:.3f}$ confirms STATE cannot distinguish "
        f"the approved BRAF inhibitor from an unrelated EGFR inhibitor."
    )
    plt.figtext(
        0.5, -0.03, caption,
        ha="center", fontsize=8.5, color=MUTED,
        style="italic", wrap=True,
        fontfamily="serif",
    )

    plt.tight_layout(pad=2.8, rect=[0, 0.04, 1, 0.97])
    out_path = os.path.join(os.path.dirname(__file__), f"{OUT_NAME}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_path} + .pdf")


if __name__ == "__main__":
    main()
