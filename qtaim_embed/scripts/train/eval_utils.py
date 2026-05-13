"""
eval_utils.py
=============
Standalone evaluation utility for the SOTA QTAIM-embed pipeline.
Produces consistent metrics tables and plots for baseline, QTAIM, and ELF models.

Called at the end of train_qtaim_graph_suba.py after training completes.

Outputs saved to log_save_dir/eval/:
  metrics_summary.csv        — train/val/test MAE, RMSE, R² per target
  metrics_summary.json       — same as JSON for easy comparison
  loss_curve.png             — train/val loss over epochs
  parity_HOMO.png            — predicted vs true (test set)
  parity_LUMO.png
  parity_Gap.png
  hexbin_HOMO.png            — density plot predicted vs true
  hexbin_LUMO.png
  hexbin_Gap.png
  residual_HOMO.png          — residuals vs true
  residual_LUMO.png
  residual_Gap.png
  error_distributions.png    — histogram of absolute errors
  molecule_sample.txt        — one molecule graph structure

Usage (called from train_qtaim_graph_suba.py):
  from eval_utils import run_eval
  run_eval(
      model=model,
      trainer=trainer,
      dm=dm,
      config=config,
      model_name="baseline",   # "baseline" | "qtaim" | "elf"
      preds_test=preds_unscaled,
      labels_test=labels_unscaled,
  )
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.dpi": 120,
})

TARGETS     = ["HOMO", "LUMO", "Gap"]
TARGET_UNITS = "Ha"
COLORS      = {"HOMO": "#185FA5", "LUMO": "#1D9E75", "Gap": "#BA7517"}


# ------------------------------------------------------------------ #
# Main entry point                                                    #
# ------------------------------------------------------------------ #
def run_eval(
    model,
    trainer,
    dm,
    config: dict,
    model_name: str,
    preds_test: np.ndarray,
    labels_test: np.ndarray,
    preds_val:   np.ndarray = None,
    labels_val:  np.ndarray = None,
    preds_train: np.ndarray = None,
    labels_train: np.ndarray = None,
):
    """
    Main evaluation entry point. Call after training completes.

    Parameters
    ----------
    model       : trained pytorch lightning model
    trainer     : pl.Trainer instance
    dm          : QTAIMGraphTaskDataModule
    config      : full config dict
    model_name  : short label for this model ("baseline", "qtaim", "elf")
    preds_test  : np.ndarray [n_test, 3] — unscaled predictions (test)
    labels_test : np.ndarray [n_test, 3] — unscaled true labels (test)
    preds_val   : optional val predictions
    labels_val  : optional val labels
    preds_train : optional train predictions
    labels_train: optional train labels
    """
    log_dir  = Path(config["dataset"]["log_save_dir"])
    eval_dir = log_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EVALUATION — {model_name.upper()}")
    print(f"  Output: {eval_dir}")
    print(f"{'='*60}")

    # ---- compute metrics ----
    metrics = compute_metrics(
        preds_test, labels_test,
        preds_val, labels_val,
        preds_train, labels_train,
    )

    # ---- save metrics table ----
    save_metrics_table(metrics, model_name, eval_dir)

    # ---- loss curve from TensorBoard ----
    # find metrics.csv — try multiple possible TensorBoard log paths
    tb_log_dir = None
    for candidate in [
        log_dir / "test_logs" / "version_0",
        log_dir / "test_logs" / "version_1",
        log_dir / "version_0",
    ]:
        if (candidate / "metrics.csv").exists():
            tb_log_dir = candidate
            break
    if tb_log_dir is None:
        # fallback: search recursively
        import glob
        found = glob.glob(str(log_dir / "**" / "metrics.csv"), recursive=True)
        if found:
            tb_log_dir = Path(found[0]).parent
    plot_loss_curve(tb_log_dir, eval_dir, model_name)

    # ---- parity plots ----
    plot_parity(preds_test, labels_test, eval_dir, model_name, split="test")
    if preds_val is not None:
        plot_parity(preds_val, labels_val, eval_dir, model_name, split="val")

    # ---- hexbin density plots ----
    plot_hexbin(preds_test, labels_test, eval_dir, model_name)

    # ---- residual plots ----
    plot_residuals(preds_test, labels_test, eval_dir, model_name)

    # ---- error distributions ----
    plot_error_distributions(preds_test, labels_test, eval_dir, model_name)

    # ---- combined parity (all 3 targets in one figure) ----
    plot_parity_combined(preds_test, labels_test, eval_dir, model_name)

    # ---- molecule sample ----
    save_molecule_sample(dm, config, eval_dir)

    print(f"\n✓ Evaluation complete. All outputs in: {eval_dir}/")
    print_metrics_table(metrics)


# ------------------------------------------------------------------ #
# Metrics computation                                                 #
# ------------------------------------------------------------------ #
def compute_metrics(preds_test, labels_test,
                    preds_val=None, labels_val=None,
                    preds_train=None, labels_train=None):
    metrics = {}
    for split, preds, labels in [
        ("test",  preds_test,  labels_test),
        ("val",   preds_val,   labels_val),
        ("train", preds_train, labels_train),
    ]:
        if preds is None or labels is None:
            continue
        split_metrics = {}
        for i, t in enumerate(TARGETS):
            mae  = float(mean_absolute_error(labels[:, i], preds[:, i]))
            rmse = float(np.sqrt(mean_squared_error(labels[:, i], preds[:, i])))
            r2   = float(r2_score(labels[:, i], preds[:, i]))
            split_metrics[t] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        avg_mae  = np.mean([split_metrics[t]["MAE"]  for t in TARGETS])
        avg_rmse = np.mean([split_metrics[t]["RMSE"] for t in TARGETS])
        avg_r2   = np.mean([split_metrics[t]["R2"]   for t in TARGETS])
        split_metrics["avg"] = {"MAE": float(avg_mae),
                                 "RMSE": float(avg_rmse),
                                 "R2": float(avg_r2)}
        metrics[split] = split_metrics
    return metrics


def print_metrics_table(metrics):
    print(f"\n{'='*70}")
    print(f"  METRICS SUMMARY ({TARGET_UNITS})")
    print(f"{'='*70}")
    for split in ["train (sample)", "train", "val", "test"]:
        if split not in metrics:
            continue
        m = metrics[split]
        print(f"\n  {split.upper()}")
        print(f"  {'Target':<8} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
        print(f"  {'-'*38}")
        for t in TARGETS + ["avg"]:
            if t in m:
                print(f"  {t:<8} {m[t]['MAE']:>10.5f} "
                      f"{m[t]['RMSE']:>10.5f} {m[t]['R2']:>8.4f}")
    print(f"{'='*70}")


def save_metrics_table(metrics, model_name, eval_dir):
    rows = []
    for split, split_m in metrics.items():
        for target, tm in split_m.items():
            rows.append({
                "model": model_name, "split": split, "target": target,
                "MAE": tm["MAE"], "RMSE": tm["RMSE"], "R2": tm["R2"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(eval_dir / "metrics_summary.csv", index=False, float_format="%.6f")

    with open(eval_dir / "metrics_summary.json", "w") as f:
        json.dump({"model": model_name, "metrics": metrics}, f, indent=2)
    print(f"  ✓ Metrics saved: {eval_dir}/metrics_summary.csv")


# ------------------------------------------------------------------ #
# Loss curve                                                          #
# ------------------------------------------------------------------ #
def _read_tfevents(tb_log_dir):
    """Read scalars from TensorBoard events file. Returns dict {tag: [(step, value)]}."""
    import glob
    tf_files = glob.glob(str(Path(tb_log_dir) / "events.out.tfevents.*"))
    if not tf_files:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(tb_log_dir))
        ea.Reload()
        result = {}
        for tag in ea.Tags().get("scalars", []):
            result[tag] = [(s.step, s.value) for s in ea.Scalars(tag)]
        return result
    except Exception:
        return None


def plot_loss_curve(tb_log_dir, eval_dir, model_name):
    """Read metrics from TensorBoard and plot loss + R² curves."""
    # try metrics.csv first
    df = None
    if tb_log_dir is not None:
        metrics_csv = Path(tb_log_dir) / "metrics.csv"
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)

    # fallback: read tfevents directly
    scalars = None
    if df is None and tb_log_dir is not None:
        scalars = _read_tfevents(tb_log_dir)

    if df is None and scalars is None:
        print(f"  ⚠  No TensorBoard data found — skipping loss curve")
        return

    # confirmed tag names from tfevents:
    # train_loss, val_loss, train_mae, val_mae, train_mse, val_mse,
    # train_r2, val_r2, lr-Adam, epoch

    def get_xy(src, tag):
        """Extract (x, y) arrays from df or scalars dict."""
        if src is df and df is not None:
            if tag in df.columns:
                sub = df[df[tag].notna()][["epoch", tag]].dropna()
                return sub["epoch"].values, sub[tag].values
        elif src is scalars and scalars is not None:
            if tag in scalars:
                steps, vals = zip(*scalars[tag])
                return np.array(steps), np.array(vals)
        return None, None

    src = df if df is not None else scalars

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- top left: loss curve ----
    ax = axes[0, 0]
    for tag, color, label, ls in [
        ("train_loss", "#185FA5", "train", "-"),
        ("val_loss",   "#D85A30", "val",   "--"),
    ]:
        x, y = get_xy(src, tag)
        if x is not None:
            ax.plot(x, y, color=color, linewidth=1.2, label=label, linestyle=ls)
    ax.set_xlabel("epoch"); ax.set_ylabel("scaled MSE loss")
    ax.set_title("Loss curve")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ---- top right: MAE curve ----
    ax = axes[0, 1]
    for tag, color, label, ls in [
        ("train_mae", "#185FA5", "train", "-"),
        ("val_mae",   "#D85A30", "val",   "--"),
    ]:
        x, y = get_xy(src, tag)
        if x is not None:
            ax.plot(x, y, color=color, linewidth=1.2, label=label, linestyle=ls)
    ax.set_xlabel("epoch"); ax.set_ylabel("MAE (scaled)")
    ax.set_title("MAE curve")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ---- bottom left: R² curve ----
    ax = axes[1, 0]
    for tag, color, label, ls in [
        ("train_r2", "#888780", "train", "--"),
        ("val_r2",   "#1D9E75", "val",   "-"),
    ]:
        x, y = get_xy(src, tag)
        if x is not None:
            ax.plot(x, y, color=color, linewidth=1.2, label=label, linestyle=ls)
    ax.set_xlabel("epoch"); ax.set_ylabel("R²")
    ax.set_title("R² curve")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    # ---- bottom right: learning rate ----
    ax = axes[1, 1]
    for tag, color, label in [
        ("lr-Adam", "#BA7517", "learning rate"),
    ]:
        x, y = get_xy(src, tag)
        if x is not None:
            ax.plot(x, y, color=color, linewidth=1.2, label=label)
    ax.set_xlabel("step"); ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule")
    ax.set_yscale("log")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Training curves — {model_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(eval_dir / "loss_curve.png")
    plt.close(fig)
    print(f"  ✓ Loss curve saved (loss + MAE + R² + LR)")


# ------------------------------------------------------------------ #
# Parity plots                                                        #
# ------------------------------------------------------------------ #
def plot_parity(preds, labels, eval_dir, model_name, split="test"):
    """Individual parity plots per target."""
    for i, t in enumerate(TARGETS):
        fig, ax = plt.subplots(figsize=(5, 5))
        p = preds[:, i]; l = labels[:, i]
        lo = min(p.min(), l.min()); hi = max(p.max(), l.max())
        margin = (hi - lo) * 0.05
        ax.scatter(l, p, alpha=0.3, s=8, color=COLORS[t], rasterized=True)
        ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin],
                "k--", linewidth=0.8, label="perfect")
        mae  = mean_absolute_error(l, p)
        r2   = r2_score(l, p)
        ax.text(0.05, 0.92, f"MAE={mae:.5f} Ha\nR²={r2:.4f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel(f"true {t} ({TARGET_UNITS})")
        ax.set_ylabel(f"predicted {t} ({TARGET_UNITS})")
        ax.set_title(f"{model_name} — {t} parity ({split})")
        ax.set_xlim(lo-margin, hi+margin)
        ax.set_ylim(lo-margin, hi+margin)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(eval_dir / f"parity_{t}_{split}.png")
        plt.close(fig)
    print(f"  ✓ Parity plots saved ({split})")


def plot_parity_combined(preds, labels, eval_dir, model_name):
    """All 3 targets in one figure side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (t, ax) in enumerate(zip(TARGETS, axes)):
        p = preds[:, i]; l = labels[:, i]
        lo = min(p.min(), l.min()); hi = max(p.max(), l.max())
        margin = (hi - lo) * 0.05
        ax.scatter(l, p, alpha=0.2, s=6, color=COLORS[t], rasterized=True)
        ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin],
                "k--", linewidth=0.8)
        mae = mean_absolute_error(l, p)
        r2  = r2_score(l, p)
        ax.text(0.05, 0.92, f"MAE={mae:.5f}\nR²={r2:.4f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel(f"true {t} ({TARGET_UNITS})", fontsize=10)
        ax.set_ylabel(f"predicted {t} ({TARGET_UNITS})", fontsize=10)
        ax.set_title(t, fontsize=11)
        ax.set_xlim(lo-margin, hi+margin)
        ax.set_ylim(lo-margin, hi+margin)
        ax.set_aspect("equal")
    fig.suptitle(f"Parity plots — {model_name} (test set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(eval_dir / "parity_combined.png")
    plt.close(fig)
    print(f"  ✓ Combined parity plot saved")


# ------------------------------------------------------------------ #
# Hexbin density plots                                                #
# ------------------------------------------------------------------ #
def plot_hexbin(preds, labels, eval_dir, model_name):
    """2D hexbin density plots — shows where predictions are dense/sparse."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (t, ax) in enumerate(zip(TARGETS, axes)):
        p = preds[:, i]; l = labels[:, i]
        lo = min(p.min(), l.min()); hi = max(p.max(), l.max())
        margin = (hi - lo) * 0.05
        hb = ax.hexbin(l, p, gridsize=50, cmap="Blues",
                       extent=[lo-margin, hi+margin, lo-margin, hi+margin],
                       mincnt=1)
        ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin],
                "r--", linewidth=0.8, label="perfect")
        plt.colorbar(hb, ax=ax, label="count")
        mae = mean_absolute_error(l, p)
        r2  = r2_score(l, p)
        ax.text(0.05, 0.92, f"MAE={mae:.5f}\nR²={r2:.4f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel(f"true {t} ({TARGET_UNITS})", fontsize=10)
        ax.set_ylabel(f"predicted {t} ({TARGET_UNITS})", fontsize=10)
        ax.set_title(t, fontsize=11)
        ax.set_aspect("equal")
    fig.suptitle(f"Density plots — {model_name} (test set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(eval_dir / "hexbin_combined.png")
    plt.close(fig)
    print(f"  ✓ Hexbin density plots saved")


# ------------------------------------------------------------------ #
# Residual plots                                                      #
# ------------------------------------------------------------------ #
def plot_residuals(preds, labels, eval_dir, model_name):
    """Residuals (predicted - true) vs true value."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (t, ax) in enumerate(zip(TARGETS, axes)):
        p = preds[:, i]; l = labels[:, i]
        residuals = p - l
        ax.scatter(l, residuals, alpha=0.2, s=6,
                   color=COLORS[t], rasterized=True)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        # running mean
        sort_idx = np.argsort(l)
        l_sorted = l[sort_idx]
        r_sorted = residuals[sort_idx]
        window = max(len(l)//50, 10)
        r_smooth = np.convolve(r_sorted, np.ones(window)/window, mode="valid")
        l_smooth = l_sorted[window//2: window//2 + len(r_smooth)]
        ax.plot(l_smooth, r_smooth, color="red", linewidth=1.2,
                label="running mean")
        ax.set_xlabel(f"true {t} ({TARGET_UNITS})", fontsize=10)
        ax.set_ylabel(f"residual ({TARGET_UNITS})", fontsize=10)
        ax.set_title(t, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        std = np.std(residuals)
        ax.text(0.05, 0.05, f"std={std:.5f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.suptitle(f"Residual plots — {model_name} (test set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(eval_dir / "residuals_combined.png")
    plt.close(fig)
    print(f"  ✓ Residual plots saved")


# ------------------------------------------------------------------ #
# Error distributions                                                 #
# ------------------------------------------------------------------ #
def plot_error_distributions(preds, labels, eval_dir, model_name):
    """Histogram of absolute errors per target."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (t, ax) in enumerate(zip(TARGETS, axes)):
        p = preds[:, i]; l = labels[:, i]
        abs_err = np.abs(p - l)
        ax.hist(abs_err, bins=60, color=COLORS[t], alpha=0.8,
                edgecolor="white", linewidth=0.3)
        mae = np.mean(abs_err)
        p50 = np.percentile(abs_err, 50)
        p95 = np.percentile(abs_err, 95)
        ax.axvline(mae, color="black",  linewidth=1.5, linestyle="--",
                   label=f"MAE={mae:.5f}")
        ax.axvline(p95, color="#D85A30", linewidth=1.2, linestyle=":",
                   label=f"p95={p95:.5f}")
        ax.set_xlabel(f"|error| ({TARGET_UNITS})", fontsize=10)
        ax.set_ylabel("count", fontsize=10)
        ax.set_title(t, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle(f"Absolute error distributions — {model_name} (test set)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(eval_dir / "error_distributions.png")
    plt.close(fig)
    print(f"  ✓ Error distributions saved")


# ------------------------------------------------------------------ #
# Molecule sample                                                     #
# ------------------------------------------------------------------ #
def save_molecule_sample(dm, config, eval_dir):
    """Print one molecule's graph structure to a text file."""
    try:
        # get first molecule from test set
        test_ds = dm.test_dataset if hasattr(dm, "test_dataset") else None
        if test_ds is None:
            return

        g, label = test_ds[0]
        lines = []
        lines.append("="*60)
        lines.append("SAMPLE MOLECULE — graph structure")
        lines.append("="*60)
        lines.append(f"Label (HOMO, LUMO, Gap): {label}")
        lines.append(f"\nNode types: {g.ntypes}")
        lines.append(f"Edge types: {g.etypes}")
        for nt in g.ntypes:
            feat = g.nodes[nt].data.get("feat")
            if feat is not None:
                lines.append(f"\n{nt} nodes: {feat.shape}")
                lines.append(f"  first node features: {feat[0].numpy().round(4).tolist()}")
        for et in g.etypes:
            feat = g.edges[et].data.get("feat")
            n_edges = g.num_edges(et)
            lines.append(f"\n{et} edges: {n_edges}")
            if feat is not None and len(feat) > 0:
                lines.append(f"  first edge features: {feat[0].numpy().round(4).tolist()}")
        lines.append("\nFeature names:")
        feature_names = config.get("dataset", {}).get("feature_names", {})
        for nt, names in feature_names.items():
            lines.append(f"  {nt}: {names}")
        lines.append("="*60)

        out = "\n".join(lines)
        with open(eval_dir / "molecule_sample.txt", "w") as f:
            f.write(out)
        print(f"  ✓ Molecule sample saved")
    except Exception as e:
        print(f"  ⚠  Could not save molecule sample: {e}")
