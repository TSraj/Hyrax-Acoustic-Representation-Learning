#!/usr/bin/env python3
"""
Build the corrected results set: one PNG per figure, one CSV per figure.

Everything here uses POST-AUDIT values (see PROBE_AUDIT.md). The pre-audit
figures in outputs/figures_paper/ are untouched and must not be mixed with
these.

THREE CONDITIONS, kept separate and labelled everywhere
-------------------------------------------------------
  frozen            base encoder, no training, converged linear probe
  species-adapted   the staged design: LoRA-adapted on the 7-class species task
                    (hyrax excluded), then FROZEN and probed. This is what
                    "animal-domain fine-tuning" means for hyrax.
  hyrax-fine-tuned  the older LoRA runs trained directly on hyrax. A DIFFERENT
                    setup - shown only as a labelled reference line, never
                    merged into the staged comparison.

Any value that is pre-audit and has not been re-measured is labelled
`pre_audit=True` in the CSV and annotated in the figure.
"""

import json
import csv
import glob
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/phase3/results_corrected")
AUDIT = Path("outputs/phase3/probe_audit")
PROBE = Path("outputs/phase3/staged_lora/probe")
STAGED = Path("outputs/phase3/staged_lora/species7")
SWEEP = Path("outputs/phase3/lora_sweep_V2")
SPECIES_SWEEP_INDEX = ("/private/tmp/claude-501/"
                       "-Users-raj-Documents-Hyrax-Acoustic-Representation-Learning/"
                       "c017a62a-245f-45a7-8525-ad552f5a8b6b/scratchpad/"
                       "sweep_merge/index.json")

MODEL_LABEL = {
    'hubert_base': 'HuBERT',
    'xls_r': 'XLS-R',
    'wavlm': 'WavLM',
    'wav2vec2_base': 'wav2vec2',
    'wav2vec2_base_960h': 'wav2vec2-960h',
    'ecapa_tdnn': 'ECAPA-TDNN',
}
MODEL_COLOR = {
    'hubert_base': '#0173B2',
    'xls_r': '#DE8F05',
    'wavlm': '#029E73',
    'wav2vec2_base': '#CC78BC',
    'wav2vec2_base_960h': '#CA9161',
    'ecapa_tdnn': '#949494',
}
COND_COLOR = {'frozen': '#0173B2', 'adapted': '#DE8F05',
              'within_session': '#CC78BC', 'session_holdout': '#029E73'}

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'axes.grid': True, 'grid.alpha': 0.25, 'axes.axisbelow': True,
})

MANIFEST = []          # provenance rows


def save(fig, name, rows, fieldnames, note=""):
    """Write PNG + the CSV of exactly what was plotted."""
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    fig.savefig(png, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    csv_path = OUT / f"{name}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    MANIFEST.append({'figure': png.name, 'csv': csv_path.name,
                     'n_values': len(rows), 'note': note})
    print(f"  {png.name}  +  {csv_path.name}  ({len(rows)} values)")


def bar_labels(ax, bars, fmt="{:.3f}", size=8):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.008, fmt.format(h),
                ha='center', va='bottom', fontsize=size)


# ---------------------------------------------------------------- data loaders

def audit_rows(pattern):
    out = {}
    for f in glob.glob(str(AUDIT / pattern)):
        r = json.load(open(f))
        out[r['model']] = r
    return out


def phase_c(model, condition):
    p = PROBE / model / condition / "staged_probe_results.json"
    return json.load(open(p)) if p.exists() else None


def species_sweep():
    """Merged 8-class species sweep: {(model, frac): [f1, ...]}."""
    idx = json.load(open(SPECIES_SWEEP_INDEX))
    out = {}
    for k, p in idx.items():
        model, frac, seed = k.split('|')
        out.setdefault((model, int(frac)), []).append(
            json.load(open(p))['test_metrics']['f1_macro'])
    return out


def hyrax_sweep():
    out = {}
    for m in ['hubert_base', 'xls_r']:
        for fr in [10, 25, 50, 100]:
            ps = (glob.glob(str(SWEEP / f"hyrax_session_holdout/{m}/frac{fr}/lora_fine_tuning_results.json"))
                  + glob.glob(str(SWEEP / f"hyrax_session_holdout/{m}/frac{fr}/seed*/lora_fine_tuning_results.json")))
            v = [json.load(open(p))['test_metrics']['f1_macro'] for p in ps]
            if v:
                out[(m, fr)] = v
    return out


# ===================================================================== figures

def fig_frozen_species(sp7):
    order = sorted(sp7, key=lambda m: -sp7[m]['corrected_internal_val']['test_f1_macro'])
    vals = [sp7[m]['corrected_internal_val']['test_f1_macro'] for m in order]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar([MODEL_LABEL[m] for m in order], vals,
                  color=[MODEL_COLOR[m] for m in order], width=0.62)
    bar_labels(ax, bars)
    ax.axhline(1 / 7, color='black', ls='-.', lw=1, alpha=0.6)
    ax.text(0.995, 1 / 7 + 0.012, 'chance (1/7)', transform=ax.get_yaxis_transform(),
            ha='right', fontsize=8, color='black', alpha=0.7)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Frozen speech encoders on 7-way species identification\n"
                 "(converged linear probe, hyrax excluded)")
    rows = [{'model': m, 'model_label': MODEL_LABEL[m],
             'test_f1_macro_corrected': round(sp7[m]['corrected_internal_val']['test_f1_macro'], 4),
             'published_pre_audit': sp7[m]['published_test_f1_macro'],
             'condition': 'frozen', 'task': 'species_7way',
             'n_classes': 7, 'chance': round(1 / 7, 4), 'pre_audit': False}
            for m in order]
    save(fig, "frozen_transfer_species", rows, list(rows[0]),
         "corrected audit values; published column shown for traceability only")


def fig_frozen_hyrax(sh):
    order = sorted(sh, key=lambda m: -sh[m]['corrected_internal_val']['test_f1_macro'])
    vals = [sh[m]['corrected_internal_val']['test_f1_macro'] for m in order]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar([MODEL_LABEL[m] for m in order], vals,
                  color=[MODEL_COLOR[m] for m in order], width=0.62)
    bar_labels(ax, bars)
    ax.axhline(0.125, color='black', ls='-.', lw=1.1)
    ax.text(0.995, 0.125 + 0.006, 'chance (1/8)', transform=ax.get_yaxis_transform(),
            ha='right', fontsize=8)
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Frozen speech encoders on hyrax individual identification\n"
                 "(session-holdout, converged linear probe)")
    rows = [{'model': m, 'model_label': MODEL_LABEL[m],
             'test_f1_macro_corrected': round(sh[m]['corrected_internal_val']['test_f1_macro'], 4),
             'published_pre_audit': sh[m]['published_test_f1_macro'],
             'condition': 'frozen', 'task': 'hyrax_session_holdout',
             'n_classes': 8, 'chance': 0.125, 'pre_audit': False}
            for m in order]
    save(fig, "frozen_transfer_hyrax", rows, list(rows[0]))


def fig_mono_vs_multi(sp7, sh):
    tasks = [("Species (7-way)", sp7, 1 / 7), ("Hyrax individual ID", sh, 0.125)]
    x = np.arange(len(tasks)); w = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    rows = []
    for i, m in enumerate(['hubert_base', 'xls_r']):
        v = [d[m]['corrected_internal_val']['test_f1_macro'] for _, d, _ in tasks]
        bars = ax.bar(x + (i - 0.5) * w, v, w, label=f"{MODEL_LABEL[m]} "
                      f"({'monolingual' if m == 'hubert_base' else 'multilingual'})",
                      color=MODEL_COLOR[m])
        bar_labels(ax, bars)
        for (tname, _, ch), val in zip(tasks, v):
            rows.append({'task': tname, 'model': m, 'model_label': MODEL_LABEL[m],
                         'pretraining': 'monolingual' if m == 'hubert_base' else 'multilingual',
                         'test_f1_macro_corrected': round(val, 4),
                         'condition': 'frozen', 'chance': round(ch, 4),
                         'pre_audit': False})
    for xi, (_, _, ch) in zip(x, tasks):
        ax.plot([xi - 0.5, xi + 0.5], [ch, ch], color='black', ls='-.', lw=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([t for t, _, _ in tasks])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Test macro-F1")
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title("Monolingual vs multilingual pretraining, frozen encoders\n"
                 "(dash-dot lines mark chance for each task)")
    save(fig, "monolingual_vs_multilingual_frozen", rows, list(rows[0]))


def fig_adaptation_effect(sp7):
    """Frozen vs species-adapted. Honest magnitudes, both selection modes for hyrax."""
    rows, panels = [], []

    # --- species task: frozen probe vs the adapted model's own species score
    for m in ['hubert_base', 'xls_r']:
        fr = sp7[m]['corrected_internal_val']['test_f1_macro']
        ad = json.load(open(STAGED / m / "seed42" / "lora_fine_tuning_results.json"))
        panels.append(("Species (7-way)", m, fr, ad['test_metrics']['f1_macro']))
        rows.append({'task': 'species_7way', 'model': m, 'model_label': MODEL_LABEL[m],
                     'frozen': round(fr, 4),
                     'species_adapted': round(ad['test_metrics']['f1_macro'], 4),
                     'delta': round(ad['test_metrics']['f1_macro'] - fr, 4),
                     'selection': 'n/a (task score, not a probe)',
                     'source_frozen': 'probe_audit species7 corrected',
                     'source_adapted': 'staged_lora species7 seed42',
                     'pre_audit': False})

    # --- hyrax: Phase C, val-selected (headline) AND test-oracle (honest ceiling)
    for m in ['hubert_base', 'xls_r']:
        b, a = phase_c(m, 'base'), phase_c(m, 'adapted')
        bv = b['best_by_variant']['mean']['test_f1_macro']
        av = a['best_by_variant']['mean']['test_f1_macro']
        bo = max(c['test_f1_macro'] for c in b['cells'] if c['variant'] == 'mean')
        ao = max(c['test_f1_macro'] for c in a['cells'] if c['variant'] == 'mean')
        panels.append(("Hyrax individual ID", m, bv, av))
        for sel, fv, av_ in [('val-selected', bv, av), ('test-oracle', bo, ao)]:
            rows.append({'task': 'hyrax_session_holdout', 'model': m,
                         'model_label': MODEL_LABEL[m],
                         'frozen': round(fv, 4), 'species_adapted': round(av_, 4),
                         'delta': round(av_ - fv, 4), 'selection': sel,
                         'source_frozen': 'phase C base sweep',
                         'source_adapted': 'phase C adapted sweep',
                         'pre_audit': False})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, task in zip(axes, ["Species (7-way)", "Hyrax individual ID"]):
        sub = [p for p in panels if p[0] == task]
        x = np.arange(len(sub)); w = 0.34
        f = [p[2] for p in sub]; a = [p[3] for p in sub]
        b1 = ax.bar(x - w / 2, f, w, label='frozen', color=COND_COLOR['frozen'])
        b2 = ax.bar(x + w / 2, a, w, label='species-adapted, then frozen probe',
                    color=COND_COLOR['adapted'])
        bar_labels(ax, b1); bar_labels(ax, b2)
        # Neutral colour: a red/green split would read as sign, and every delta
        # here is positive. Magnitude is the point, not direction.
        for xi, (_, _, fv, av) in zip(x, sub):
            ax.annotate(f"{av - fv:+.3f}", xy=(xi, max(fv, av) + 0.05),
                        ha='center', fontsize=9.5, fontweight='bold', color='#333333')
        ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL[p[1]] for p in sub])
        ax.set_ylim(0, 1.22 if 'Species' in task else 0.74)
        ax.set_ylabel("Test macro-F1")
        ax.set_title(task)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.95)
        if 'Hyrax' in task:
            # The val-selected gain is selection-dependent; state the oracle in
            # the panel so the bars are never read on their own.
            oracle = [r for r in rows if r['task'] == 'hyrax_session_holdout'
                      and r['selection'] == 'test-oracle']
            txt = "  |  ".join(f"{MODEL_LABEL[r['model']]} {r['delta']:+.3f}"
                               for r in oracle)
            # Right-aligned and below the legend band, which sits upper-left.
            ax.text(0.985, 0.78, f"at the test-oracle layer: {txt}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                    color='#B33A3A',
                    bbox=dict(fc='white', ec='#B33A3A', lw=0.8, alpha=0.97, pad=3))
    fig.suptitle("Effect of animal-domain adaptation: frozen vs species-adapted encoder\n"
                 "Adaptation is worth ~+0.015 on species. On hyrax the gain shown is "
                 "val-selected; at the test-oracle layer it is +0.000 / -0.005 (see CSV).",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "adaptation_effect", rows, list(rows[0]),
         "hyrax rows include both val-selected and test-oracle selection")


def fig_per_layer():
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.5))
    for ax, m in zip(axes, ['hubert_base', 'xls_r']):
        b, a = phase_c(m, 'base'), phase_c(m, 'adapted')
        cb = sorted([c for c in b['cells'] if c['variant'] == 'mean'], key=lambda c: c['layer'])
        ca = {c['layer']: c for c in a['cells'] if c['variant'] == 'mean'}
        layers = [c['layer'] for c in cb]
        x = np.arange(len(layers)); w = 0.4
        vb = [c['test_f1_macro'] for c in cb]
        va = [ca[l]['test_f1_macro'] for l in layers]
        ax.bar(x - w / 2, vb, w, label='frozen (base)', color=COND_COLOR['frozen'])
        ax.bar(x + w / 2, va, w, label='species-adapted', color=COND_COLOR['adapted'])
        # Best-layer callouts: marker on the bar plus one corner box. Two
        # stacked text annotations collided whenever both bests landed on the
        # same layer, which they do for HuBERT (both at L0).
        ymax = max(max(vb), max(va))
        best_txt = []
        for lab, vals, off, col in [('frozen', vb, -w / 2, COND_COLOR['frozen']),
                                    ('adapted', va, w / 2, COND_COLOR['adapted'])]:
            i = int(np.argmax(vals))
            # Sit the marker ABOVE the rotated value label, which extends
            # roughly 0.10*ymax upward from the bar top.
            ax.plot(x[i] + off, vals[i] + 0.13 * ymax, marker='v', ms=7, color=col,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=6)
            best_txt.append(f"best {lab}: L{layers[i]} = {vals[i]:.3f}")
        ax.text(0.99, 0.97, "\n".join(best_txt), transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5,
                bbox=dict(fc='white', ec='#888888', lw=0.7, alpha=0.95, pad=4))
        # Value labels on every bar, rotated so 26/50 bars stay legible.
        for xi, v in list(zip(x - w / 2, vb)) + list(zip(x + w / 2, va)):
            ax.text(xi, v + 0.006, f"{v:.3f}", ha='center', va='bottom',
                    fontsize=5.2, rotation=90)
        for l, bv, av in zip(layers, vb, va):
            rows.append({'model': m, 'model_label': MODEL_LABEL[m], 'layer': l,
                         'frozen': round(bv, 4), 'species_adapted': round(av, 4),
                         'delta': round(av - bv, 4), 'pooling': 'mean',
                         'identical_by_construction': l == 0, 'pre_audit': False})
        ax.set_xticks(x); ax.set_xticklabels(layers, fontsize=8)
        ax.set_xlabel("Layer (hidden_states index; 0 = pre-transformer feature projection)")
        ax.set_ylabel("Test macro-F1")
        ax.set_ylim(0, max(max(vb), max(va)) * 1.42)
        ax.set_title(f"{MODEL_LABEL[m]} - hyrax individual ID by layer, mean pooling")
        ax.legend(fontsize=8, loc='upper left')
    fig.suptitle("Per-layer hyrax performance, frozen vs species-adapted\n"
                 "L0 is bit-identical between conditions by construction: LoRA acts on "
                 "attention, which the feature projection precedes.", fontsize=11)
    fig.tight_layout()
    save(fig, "per_layer_performance", rows, list(rows[0]),
         "L0 identical by construction; verified delta = 0.0000")


def fig_pooling():
    rows = []
    fig, ax = plt.subplots(figsize=(9, 4.8))
    groups, labels = [], []
    for m in ['hubert_base', 'xls_r']:
        for cond, key in [('frozen', 'base'), ('species-adapted', 'adapted')]:
            r = phase_c(m, key)
            mean_c = r['best_by_variant']['mean']
            head_c = r['best_by_variant']['head0']
            groups.append((mean_c['test_f1_macro'], head_c['test_f1_macro']))
            labels.append(f"{MODEL_LABEL[m]}\n{cond}")
            for variant, cell in [('mean', mean_c), ('head0', head_c)]:
                rows.append({'model': m, 'model_label': MODEL_LABEL[m],
                             'condition': cond, 'pooling': variant,
                             'best_layer': cell['layer'],
                             'test_f1_macro': round(cell['test_f1_macro'], 4),
                             'selection': 'val macro-F1', 'pre_audit': False})
    x = np.arange(len(groups)); w = 0.36
    b1 = ax.bar(x - w / 2, [g[0] for g in groups], w, label='mean pooling', color='#0173B2')
    b2 = ax.bar(x + w / 2, [g[1] for g in groups], w, label='head 0 only', color='#DE8F05')
    bar_labels(ax, b1); bar_labels(ax, b2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.60); ax.set_ylabel("Test macro-F1")
    ax.legend(fontsize=9)
    ax.set_title("Pooling strategy at each condition's best layer (hyrax individual ID)\n"
                 "Mean pooling beats a single attention head by 0.14-0.17 everywhere")
    save(fig, "pooling_comparison", rows, list(rows[0]))


def fig_data_efficiency(sp8, sh):
    sw_sp, sw_hy = species_sweep(), hyrax_sweep()
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    for ax, (title, sweep, fracs, frozen, task) in zip(axes, [
        ("Species identification (8-class)", sw_sp, [1, 2, 5, 10, 25, 50, 100],
         {m: sp8[m]['corrected_internal_val']['test_f1_macro'] for m in ['hubert_base', 'xls_r']},
         'species_8class'),
        ("Hyrax individual identification", sw_hy, [10, 25, 50, 100],
         {m: sh[m]['corrected_internal_val']['test_f1_macro'] for m in ['hubert_base', 'xls_r']},
         'hyrax_session_holdout'),
    ]):
        for m in ['hubert_base', 'xls_r']:
            xs, mus, sds, ns = [], [], [], []
            for fr in fracs:
                v = sweep.get((m, fr))
                if not v:
                    continue
                xs.append(fr); mus.append(statistics.mean(v))
                sds.append(statistics.stdev(v) if len(v) > 1 else 0.0); ns.append(len(v))
            ax.plot(xs, mus, marker='o', ms=4, lw=1.7, color=MODEL_COLOR[m],
                    label=f"{MODEL_LABEL[m]} fine-tuned on this task")
            ax.fill_between(xs, np.array(mus) - np.array(sds), np.array(mus) + np.array(sds),
                            color=MODEL_COLOR[m], alpha=0.16)
            ax.axhline(frozen[m], color=MODEL_COLOR[m], ls='--', lw=1.3)
            # Anchor at the left edge with an opaque box; right-aligned inline
            # labels collided with each other and with the dashed lines.
            ax.annotate(f"{MODEL_LABEL[m]} frozen {frozen[m]:.3f}",
                        xy=(xs[0], frozen[m]), xytext=(4, 4 if m == 'xls_r' else -11),
                        textcoords='offset points', fontsize=7.5,
                        color=MODEL_COLOR[m],
                        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2))
            for fr, mu, sd, n in zip(xs, mus, sds, ns):
                rows.append({'task': task, 'model': m, 'model_label': MODEL_LABEL[m],
                             'fraction_pct': fr, 'finetuned_mean': round(mu, 4),
                             'finetuned_sd': round(sd, 4), 'n_seeds': n,
                             'frozen_corrected': round(frozen[m], 4),
                             'finetuned_minus_frozen': round(mu - frozen[m], 4),
                             'pre_audit': False})
        ax.set_xscale('log'); ax.set_xticks(fracs)
        ax.set_xticklabels([f"{f}%" for f in fracs])
        ax.set_xlabel("Training data used for fine-tuning")
        ax.set_ylabel("Test macro-F1")
        ax.set_title(title); ax.legend(fontsize=8, loc='lower right')
    fig.suptitle("Data efficiency: fine-tuning on the task vs a frozen encoder\n"
                 "Dashed lines are the corrected frozen baselines. Fine-tuning falls "
                 "BELOW frozen at low data fractions.", fontsize=11)
    fig.tight_layout()
    save(fig, "data_efficiency", rows, list(rows[0]),
         "shaded band = +-1 SD across seeds; 50%/100% species are single-seed")


def fig_leakage():
    cells = {}
    for f in glob.glob(str(AUDIT / "probe_audit_xls_r_screen_*.json")):
        tag = Path(f).stem.split('screen_')[1]
        ver, task = tag.split('_', 1)
        cells[(ver, task)] = json.load(open(f))
    vers = ['original', 'bioda', 'aca']
    rows = []
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(vers)); w = 0.36
    wi = [cells[(v, 'within')]['corrected_internal_val']['test_f1_macro'] for v in vers]
    ho = [cells[(v, 'holdout')]['corrected_internal_val']['test_f1_macro'] for v in vers]
    b1 = ax.bar(x - w / 2, wi, w, label='within-session (train and test share sessions)',
                color=COND_COLOR['within_session'])
    b2 = ax.bar(x + w / 2, ho, w, label='session-holdout (disjoint sessions)',
                color=COND_COLOR['session_holdout'])
    bar_labels(ax, b1); bar_labels(ax, b2)
    for xi, a, b in zip(x, wi, ho):
        ax.annotate(f"leakage\n+{a - b:.3f}", xy=(xi, max(a, b) + 0.055), ha='center',
                    fontsize=9, fontweight='bold', color='#B33A3A')
    for v, a, b in zip(vers, wi, ho):
        rows.append({'denoiser_version': v, 'within_session': round(a, 4),
                     'session_holdout': round(b, 4), 'leakage_gap': round(a - b, 4),
                     'published_gap_pre_audit': round(
                         cells[(v, 'within')]['published_test_f1_macro']
                         - cells[(v, 'holdout')]['published_test_f1_macro'], 4),
                     'model': 'xls_r', 'pre_audit': False})
    ax.set_xticks(x); ax.set_xticklabels([v.upper() for v in vers])
    ax.set_ylim(0, 0.86); ax.set_ylabel("Test macro-F1")
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title("Session leakage in hyrax individual ID (XLS-R, converged probe)\n"
                 "Pre-audit gaps were ~0.00 and read as 'no leakage'; corrected they are 0.30-0.42")
    save(fig, "session_leakage", rows, list(rows[0]),
         "published_gap_pre_audit column shows what the undertrained probe reported")


# ======================================================================== main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp7 = audit_rows("probe_audit_*_species7_*.json")
    sp8 = audit_rows("probe_audit_*_species8_*.json")
    sh = audit_rows("probe_audit_*_sh.json")
    for name, d in [('species7', sp7), ('species8', sp8), ('hyrax', sh)]:
        if len(d) != 6:
            print(f"WARNING: {name} has {len(d)} models, expected 6")

    print(f"Writing to {OUT}/\n")
    fig_frozen_species(sp7)
    fig_frozen_hyrax(sh)
    fig_mono_vs_multi(sp7, sh)
    fig_adaptation_effect(sp7)
    fig_per_layer()
    fig_pooling()
    fig_data_efficiency(sp8, sh)
    fig_leakage()

    with open(OUT / "MANIFEST.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['figure', 'csv', 'n_values', 'note'])
        w.writeheader(); w.writerows(MANIFEST)
    print(f"\n{len(MANIFEST)} figures, each with a matching CSV. MANIFEST.csv written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
