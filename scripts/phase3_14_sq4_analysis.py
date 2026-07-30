#!/usr/bin/env python3
"""
Phase 3 - SQ4 step 2: do acoustic characteristics predict transfer success?

Unit of analysis is the individual from the Phase 2 pooled 69-way task; the
outcome is that individual's mean F1 across the 5 SSL transformers under a
frozen encoder. Only individuals with >= MIN_SUPPORT test items are used.

The analysis is deliberately run three ways, because they disagree and the
disagreement is the finding:

  1. naive        - correlation over all individuals, ignoring dataset. This
                    treats individuals as independent when they are not.
  2. within       - both variables centred on their dataset mean, so only
                    variation among individuals recorded under the SAME
                    conditions remains.
  3. between      - dataset means only (N = number of datasets).

plus a mixed model with a dataset random intercept, and a leave-one-dataset-out
sweep of the naive correlations.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

PREDICTORS = [
    ("centroid_hz", "Spectral centroid (Hz)"),
    ("bandwidth_hz", "Bandwidth, rolloff95-05 (Hz)"),
    ("snr_proxy_db", "SNR proxy (dB)"),
    ("duration_s", "Mean recording duration (s)"),
    ("frac_energy_above_8k", "Energy above 8 kHz (fraction)"),
]
MIN_SUPPORT = 5


def load(f1_csv, ac_csv):
    f1 = pd.read_csv(f1_csv)
    ac = pd.read_csv(ac_csv)
    d = f1.merge(ac, on=['individual', 'dataset'])
    return d, d[d['support'] >= MIN_SUPPORT].copy()


def analyse(u, logger_print):
    rows = []
    # centred copies for the within-dataset view
    w = u.copy()
    w['f1_c'] = w['f1_mean'] - w.groupby('dataset')['f1_mean'].transform('mean')

    g = u.groupby('dataset').agg(
        f1=('f1_mean', 'mean'),
        **{p: (p, 'mean') for p, _ in PREDICTORS}).reset_index()

    import statsmodels.formula.api as smf
    z = u.copy()
    for p, _ in PREDICTORS:
        z['z_' + p] = (z[p] - z[p].mean()) / z[p].std()

    for p, label in PREDICTORS:
        rs_n, ps_n = stats.spearmanr(u[p], u['f1_mean'])
        w[p + '_c'] = w[p] - w.groupby('dataset')[p].transform('mean')
        rs_w, ps_w = stats.spearmanr(w[p + '_c'], w['f1_c'])
        rs_b, ps_b = stats.spearmanr(g[p], g['f1'])
        m = smf.mixedlm(f"f1_mean ~ z_{p}", z, groups=z['dataset']).fit(reml=False)

        # leave-one-dataset-out on the naive correlation
        loo = {}
        for ds in sorted(u['dataset'].unique()):
            s = u[u['dataset'] != ds]
            loo[ds] = stats.spearmanr(s[p], s['f1_mean'])[0]

        rows.append({
            'predictor': p, 'label': label,
            'naive_rho': rs_n, 'naive_p': ps_n,
            'within_rho': rs_w, 'within_p': ps_w,
            'between_rho': rs_b, 'between_p': ps_b, 'between_n': len(g),
            'mixed_beta': m.params['z_' + p], 'mixed_p': m.pvalues['z_' + p],
            'loo_rho_min': min(loo.values()), 'loo_rho_max': max(loo.values()),
            'loo_worst_dataset': min(loo, key=lambda k: abs(loo[k])),
            'loo_rho_when_worst_dropped': loo[min(loo, key=lambda k: abs(loo[k]))],
        })

    m0 = smf.mixedlm("f1_mean ~ 1", z, groups=z['dataset']).fit(reml=False)
    icc = m0.cov_re.iloc[0, 0] / (m0.cov_re.iloc[0, 0] + m0.scale)
    return pd.DataFrame(rows), g, icc


def plot(u, res, out_dir):
    fig, axes = plt.subplots(1, len(PREDICTORS), figsize=(4 * len(PREDICTORS), 4.2))
    datasets = sorted(u['dataset'].unique())
    colors = dict(zip(datasets, plt.cm.tab10(np.linspace(0, 1, len(datasets)))))

    for ax, (p, label) in zip(axes, PREDICTORS):
        for ds in datasets:
            s = u[u['dataset'] == ds]
            ax.scatter(s[p], s['f1_mean'], s=42, color=colors[ds], label=ds,
                       edgecolor='white', linewidth=0.6, zorder=3)
        r = res[res['predictor'] == p].iloc[0]
        ax.set_xlabel(label, fontsize=9)
        ax.set_title(f"naive ρ={r['naive_rho']:+.2f} (p={r['naive_p']:.3f})\n"
                     f"within-dataset ρ={r['within_rho']:+.2f} (p={r['within_p']:.2f})",
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Mean per-individual F1 (5 models, frozen)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(datasets), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("SQ4: acoustic predictors vs frozen-transfer success "
                 f"(N={len(u)} individuals, {len(datasets)} datasets)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = Path(out_dir) / "sq4_predictors_vs_transfer.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1", default="outputs/phase3/sq4/per_individual_f1.csv")
    ap.add_argument("--acoustic", default="outputs/phase3/sq4/acoustic_predictors.csv")
    ap.add_argument("--out-dir", default="outputs/phase3/sq4")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d, u = load(args.f1, args.acoustic)
    print(f"individuals total={len(d)} usable(support>={MIN_SUPPORT})={len(u)} "
          f"datasets={u['dataset'].nunique()}")

    res, g, icc = analyse(u, print)
    res.to_csv(out_dir / "sq4_correlations.csv", index=False)
    g.to_csv(out_dir / "sq4_dataset_means.csv", index=False)
    png = plot(u, res, out_dir)

    print("\n" + res[['predictor', 'naive_rho', 'naive_p', 'within_rho', 'within_p',
                      'between_rho', 'mixed_beta', 'mixed_p',
                      'loo_rho_when_worst_dropped']].round(4).to_string(index=False))
    print(f"\nICC = {icc:.3f}")

    with open(out_dir / "sq4_report.md", 'w') as f:
        f.write("# SQ4: Do acoustic characteristics predict transfer success?\n\n")
        f.write(f"Unit: individual. Outcome: mean per-individual F1 across 5 frozen SSL "
                f"transformers on the pooled 69-way task.\n")
        f.write(f"N = {len(u)} individuals with >= {MIN_SUPPORT} test items, "
                f"across {u['dataset'].nunique()} datasets.\n\n")
        f.write(f"**{icc*100:.0f}% of the variance in per-individual F1 is between "
                f"datasets** (ICC = {icc:.3f}), so individuals are far from independent.\n\n")
        f.write("| Predictor | naive rho (p) | within-dataset rho (p) | between-dataset rho | "
                "mixed beta (p) | naive rho after dropping most influential dataset |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, r in res.iterrows():
            f.write(f"| {r['label']} | {r['naive_rho']:+.3f} ({r['naive_p']:.4f}) | "
                    f"{r['within_rho']:+.3f} ({r['within_p']:.3f}) | "
                    f"{r['between_rho']:+.3f} | "
                    f"{r['mixed_beta']:+.4f} ({r['mixed_p']:.3f}) | "
                    f"{r['loo_rho_when_worst_dropped']:+.3f} (drop {r['loo_worst_dataset']}) |\n")
        f.write("\n## Dataset means\n\n")
        cols = list(g.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for _, r in g.iterrows():
            f.write("| " + " | ".join(
                f"{r[c]:.3f}" if isinstance(r[c], (int, float, np.floating)) else str(r[c])
                for c in cols) + " |\n")
    print(f"\nWrote sq4_correlations.csv, sq4_dataset_means.csv, sq4_report.md, {png.name}")


if __name__ == "__main__":
    main()
