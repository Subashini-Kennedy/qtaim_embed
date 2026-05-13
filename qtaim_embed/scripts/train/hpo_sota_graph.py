"""
hpo_sota_graph.py
=================
Optuna hyperparameter optimisation for the SOTA atom-bond-global graph pipeline.
Runs HPO for one experiment mode at a time:
  --mode baseline  : topology only (atom=13, bond=7)
  --mode qtaim     : topology + QTAIM (atom=31, bond=26)
  --mode elf       : topology + ELF (atom=18, bond=12)

Searches over:
  lr, dropout, n_conv_layers, embedding_size, batch_size,
  weight_decay, resid_n_graph_convs, fc_layer_size

Best params saved to --outdir/best.json
All trials saved to --outdir/all_trials.csv

Usage:
  python hpo_sota_graph.py --mode baseline --outdir results/hpo_sota/baseline
  python hpo_sota_graph.py --mode qtaim    --outdir results/hpo_sota/qtaim
  python hpo_sota_graph.py --mode elf      --outdir results/hpo_sota/elf
"""

from __future__ import annotations
import argparse, copy, json, logging, os, sys, warnings
from pathlib import Path

import numpy as np
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from qtaim_embed.core.datamodule import QTAIMGraphTaskDataModule
from qtaim_embed.models.utils import load_graph_level_model_from_config

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

# ------------------------------------------------------------------ #
# Feature dimensions per mode                                         #
# ------------------------------------------------------------------ #
MODE_DIMS = {
    "baseline": {"atom": 13, "bond": 7,  "global": 3},
    "qtaim":    {"atom": 31, "bond": 26, "global": 3},
    "elf":      {"atom": 18, "bond": 12, "global": 3},
}

MODE_EXTRA_KEYS = {
    "baseline": {"atom": [], "bond": [], "global": ["homo","lumo","gap"]},
    "qtaim": {
        "atom": [
            "new_extra_feat_atom_Hamiltonian_K","new_extra_feat_atom_e_density",
            "new_extra_feat_atom_lap_e_density","new_extra_feat_atom_e_loc_func",
            "new_extra_feat_atom_ave_loc_ion_E","new_extra_feat_atom_delta_g_promolecular",
            "new_extra_feat_atom_delta_g_hirsh","new_extra_feat_atom_esp_nuc",
            "new_extra_feat_atom_esp_e","new_extra_feat_atom_esp_total",
            "new_extra_feat_atom_grad_norm","new_extra_feat_atom_lap_norm",
            "new_extra_feat_atom_eig_hess","new_extra_feat_atom_det_hessian",
            "new_extra_feat_atom_ellip_e_dens","new_extra_feat_atom_eta",
            "new_extra_feat_atom_energy_density","new_extra_feat_atom_lol",
        ],
        "bond": [
            "new_extra_feat_bond_Lagrangian_K","new_extra_feat_bond_Hamiltonian_K",
            "new_extra_feat_bond_e_density","new_extra_feat_bond_lap_e_density",
            "new_extra_feat_bond_e_loc_func","new_extra_feat_bond_ave_loc_ion_E",
            "new_extra_feat_bond_delta_g_promolecular","new_extra_feat_bond_delta_g_hirsh",
            "new_extra_feat_bond_esp_nuc","new_extra_feat_bond_esp_e",
            "new_extra_feat_bond_esp_total","new_extra_feat_bond_grad_norm",
            "new_extra_feat_bond_lap_norm","new_extra_feat_bond_eig_hess",
            "new_extra_feat_bond_det_hessian","new_extra_feat_bond_ellip_e_dens",
            "new_extra_feat_bond_eta","new_extra_feat_bond_energy_density",
            "new_extra_feat_bond_lol",
        ],
        "global": ["homo","lumo","gap"],
    },
    "elf": {
        "atom": ["elf_atom_group_value","elf_atom_volume","elf_atom_population",
                 "elf_atom_charge","elf_atom_value"],
        "bond": ["elf_bond_group_value","elf_bond_volume","elf_bond_population",
                 "elf_bond_charge","elf_bond_value"],
        "global": ["homo","lumo","gap"],
    },
}

MODE_PKL = {
    "baseline": {
        "train": "data_suba/filtered_qtaim_fullqm9_corrected/train_43k.pkl",
        "val":   "data_suba/filtered_qtaim_fullqm9_corrected/val_43k.pkl",
        "test":  "data_suba/filtered_qtaim_fullqm9_corrected/test_43k.pkl",
    },
    "qtaim": {
        "train": "data_suba/filtered_qtaim_fullqm9_corrected/train_43k.pkl",
        "val":   "data_suba/filtered_qtaim_fullqm9_corrected/val_43k.pkl",
        "test":  "data_suba/filtered_qtaim_fullqm9_corrected/test_43k.pkl",
    },
    "elf": {
        "train": "data_suba/filtered_qtaim_fullqm9_elf_corrected/train_43k_elf.pkl",
        "val":   "data_suba/filtered_qtaim_fullqm9_elf_corrected/val_43k_elf.pkl",
        "test":  "data_suba/filtered_qtaim_fullqm9_elf_corrected/test_43k_elf.pkl",
    },
}


# ------------------------------------------------------------------ #
# Build config for one trial                                          #
# ------------------------------------------------------------------ #
def build_config(trial, mode, sota_root):
    ROOT = Path(sota_root)
    pkls = MODE_PKL[mode]
    dims = MODE_DIMS[mode]

    lr               = trial.suggest_float("lr",               1e-4, 5e-2, log=True)
    dropout          = trial.suggest_float("dropout",          0.0,  0.3)
    n_conv_layers    = trial.suggest_int(  "n_conv_layers",    6,    12)
    embedding_size   = trial.suggest_categorical("embedding_size", [16, 32, 64])
    batch_size       = trial.suggest_categorical("batch_size", [128, 256])
    weight_decay     = trial.suggest_float("weight_decay",     1e-6, 1e-3, log=True)
    resid_n_graph_convs = trial.suggest_int("resid_n_graph_convs", 1, 3)
    fc_hidden        = trial.suggest_categorical("fc_hidden",  [512, 1024, 2048])

    return {
        "optim": {
            "precision": "bf16-mixed",
            "num_devices": 1, "num_nodes": 1,
            "accumulate_grad_batches": 1,
            "strategy": "auto",
            "gradient_clip_val": 10.0,
            "train_batch_size": batch_size,
            "val_batch_size":   batch_size,
            "test_batch_size":  batch_size,
            "num_workers": 2,
            "pin_memory": False,
            "persistent_workers": False,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 0,
            "log_every_n_steps": 50,
        },
        "dataset": {
            "train_dataset_loc": str(ROOT / pkls["train"]),
            "val_dataset_loc":   str(ROOT / pkls["val"]),
            "test_dataset_loc":  str(ROOT / pkls["test"]),
            "log_save_dir": "/tmp/hpo_sota_trial/",
            "verbose": False, "debug": False,
            "num_workers": 2,
            "train_batch_size": batch_size,
            "val_batch_size":   batch_size,
            "test_batch_size":  batch_size,
            "pin_memory": False,
            "persistent_workers": False,
            "allowed_ring_size": [3,4,5,6,7],
            "allowed_charges": None,
            "allowed_spins": None,
            "self_loop": True,
            "bond_key": "new_bonds",
            "map_key": "new_bond_indices",
            "per_atom": False,
            "target_list": ["homo","lumo","gap"],
            "element_set": ["C","F","H","N","O"],
            "val_prop": 0.0, "test_prop": 0.0,
            "seed": 42,
            "extra_dataset_info": {},
            "log_scale_features": False,
            "log_scale_targets": False,
            "standard_scale_features": True,
            "standard_scale_targets": True,
            "extra_keys": MODE_EXTRA_KEYS[mode],
        },
        "model": {
            "atom_feature_size":   dims["atom"],
            "bond_feature_size":   dims["bond"],
            "global_feature_size": dims["global"],
            "compile": False, "compiled": False,
            "conv_fn": "ResidualBlock",
            "initializer": "kaiming",
            "target_dict": {"global": ["homo","lumo","gap"]},
            "ntasks": 3,
            "dropout": dropout,
            "batch_norm_tf": True,
            "activation": "ReLU",
            "bias": True,
            "norm": "both",
            "aggregate": "sum",
            "n_conv_layers": n_conv_layers,
            "lr": lr,
            "weight_decay": weight_decay,
            "lr_plateau_patience": 15,
            "lr_scale_factor": 0.5,
            "scheduler_name": "reduce_on_plateau",
            "loss_fn": "mse",
            "resid_n_graph_convs": resid_n_graph_convs,
            "embedding_size": embedding_size,
            "fc_layer_size": [fc_hidden, fc_hidden],
            "shape_fc": "flat",
            "fc_dropout": 0.0,
            "fc_batch_norm": True,
            "n_fc_layers": 2,
            "global_pooling_fn": "GlobalAttentionPoolingThenCat",
            "ntypes_pool": ["atom","bond","global"],
            "ntypes_pool_direct_cat": ["global"],
            "lstm_iters": 5, "lstm_layers": 3,
            "num_heads": 3,
            "feat_drop": 0.1, "attn_drop": 0.0,
            "residual": False,
            "num_heads_gat": 3,
            "dropout_feat_gat": 0.2, "dropout_attn_gat": 0.1,
            "hidden_size": 30, "hidden_size_gat": 30,
            "residual_gat": True,
            "classifier": False,
            "batch_norm": True,
            "pooling_ntypes": ["atom","bond","global"],
            "pooling_ntypes_direct": ["global"],
            "fc_hidden_size_1": fc_hidden,
            "fc_num_layers": 2,
            "bond_key": "new_bonds",
            "map_key": "new_bond_indices",
            "restore": False,
            "max_epochs": 250,          # short for HPO
            "extra_stop_patience": 30,  #  early stopping for speed
        },
    }


# ------------------------------------------------------------------ #
# Single trial                                                        #
# ------------------------------------------------------------------ #
def run_trial(trial, mode, sota_root):
    config = build_config(trial, mode, sota_root)

    dm = QTAIMGraphTaskDataModule(config=config)
    config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]
    _, feature_size = dm.prepare_data(stage="fit")
    config["model"]["atom_feature_size"]   = feature_size["atom"]
    config["model"]["bond_feature_size"]   = feature_size["bond"]
    config["model"]["global_feature_size"] = feature_size["global"]

    model = load_graph_level_model_from_config(config["model"])

    early_stop = EarlyStopping(
        monitor="val_mae", patience=config["model"]["extra_stop_patience"],
        mode="min", verbose=False,
    )

    trainer = pl.Trainer(
        max_epochs=config["model"]["max_epochs"],
        accelerator="gpu",
        devices=1, num_nodes=1,
        gradient_clip_val=config["optim"]["gradient_clip_val"],
        accumulate_grad_batches=1,
        enable_progress_bar=False,
        callbacks=[early_stop],
        enable_checkpointing=False,
        strategy="auto",
        logger=False,
        precision=config["optim"]["precision"],
        enable_model_summary=False,
    )

    trainer.fit(model, dm)

    # return best val_mae as objective
    best_val = early_stop.best_score
    if best_val is None or torch.isnan(torch.tensor(best_val)):
        return float("inf")
    return float(best_val)


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser("hpo_sota_graph.py")
    p.add_argument("--mode",      required=True, choices=["baseline","qtaim","elf"])
    p.add_argument("--sota_root", required=True,
                   help="Root of qtaim_embed_private")
    p.add_argument("--outdir",    required=True)
    p.add_argument("--n_trials",  type=int, default=60)
    p.add_argument("--study_name",type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"hpo_sota_{args.mode}"

    print(f"\n{'='*60}")
    print(f"  HPO — SOTA graph — mode={args.mode}")
    print(f"  n_trials={args.n_trials}")
    print(f"  outdir={OUTDIR}")
    print(f"{'='*60}\n")

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    def objective(trial):
        try:
            val = run_trial(trial, args.mode, args.sota_root)
            print(f"  trial {trial.number:3d}  val_mae={val:.5f}  "
                  f"lr={trial.params['lr']:.5f}  "
                  f"n_conv={trial.params['n_conv_layers']}  "
                  f"emb={trial.params['embedding_size']}  "
                  f"bs={trial.params['batch_size']}")
            return val
        except Exception as e:
            print(f"  trial {trial.number:3d}  FAILED: {e}")
            return float("inf")

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  BEST TRIAL: {best.number}  val_mae={best.value:.5f}")
    print(f"  Params: {best.params}")
    print(f"{'='*60}")

    # save best.json
    best_json = {
        "best_params": best.params,
        "best_value":  best.value,
        "best_trial":  best.number,
        "run_config": {
            "mode": args.mode,
            "n_trials": args.n_trials,
        },
    }
    with open(OUTDIR / "best.json", "w") as f:
        json.dump(best_json, f, indent=2)

    # save all_trials.csv
    import pandas as pd
    rows = []
    for t in study.trials:
        row = {"trial": t.number, "value": t.value, "state": str(t.state)}
        row.update(t.params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTDIR / "all_trials.csv", index=False)

    print(f"\n✓ Best params: {OUTDIR}/best.json")
    print(f"✓ All trials:  {OUTDIR}/all_trials.csv")


if __name__ == "__main__":
    main()