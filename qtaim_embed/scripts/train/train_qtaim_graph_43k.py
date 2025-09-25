#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)

from qtaim_embed.core.datamodule import QTAIMGraphTaskDataModule, LMDBDataModule
from qtaim_embed.models.utils import LogParameters, load_graph_level_model_from_config
from qtaim_embed.utils.data import get_default_graph_level_config


# ----- PyTorch perf knobs -----
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


# ===========================
# Per-task metric helpers
# ===========================
def _target_names_from_cfg(cfg: dict):
    names = cfg.get("dataset", {}).get("target_list", [])
    if names:
        return list(names)
    td = cfg.get("model", {}).get("target_dict", {})
    if isinstance(td, dict) and td:
        return [n for _, arr in td.items() for n in arr]
    nt = int(cfg.get("model", {}).get("ntasks", 1))
    return [f"task{i}" for i in range(nt)]


def _save_per_task_df(df: pd.DataFrame, outdir: Path, tag: str):
    outdir.mkdir(parents=True, exist_ok=True)
    df.index.name = "target"
    (outdir / f"per_task_test_metrics_{tag}.json").write_text(
        df.to_json(orient="index", indent=2)
    )
    df.to_csv(outdir / f"per_task_test_metrics_{tag}.csv")
    print("\nPER-TASK TEST METRICS")
    print(df.to_string(float_format=lambda x: f"{x:.6f}"))
    for name, row in df.iterrows():
        print(
            f"[per-task] {name:>10}  MAE={row['mae']:.6f}  MSE={row['mse']:.6f}  R2={row['r2']:.6f}"
        )


def _invert_cols(arr: np.ndarray, scalers):
    if not scalers:
        return arr
    out = np.empty_like(arr, dtype=np.float64)
    T = arr.shape[1]
    for i in range(T):
        s = scalers[i] if i < len(scalers) else None
        if s is not None and hasattr(s, "inverse_transform"):
            try:
                out[:, i] = np.asarray(s.inverse_transform(arr[:, i : i + 1])).reshape(-1)
                continue
            except Exception:
                pass
        out[:, i] = arr[:, i]
    return out


def _per_task_eval_lmdb(cfg, dm, best_ckpt_path: str):
    """
    Robust per-task eval on CPU using the same DataModule pipeline.
    Avoids CUDA + heterograph feature edge cases.
    """
    device = torch.device("cpu")

    # Build a *fresh* plain model and load weights on CPU
    mcfg = dict(cfg.get("model", {}))
    model = load_graph_level_model_from_config(mcfg).to(device)
    state = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Ensure test loader exists (same as used by trainer.test)
    try:
        dm.prepare_data(stage="test")
    except Exception:
        pass
    try:
        dm.setup(stage="test")
    except Exception:
        pass
    test_loader = dm.test_dataloader()

    # Names + scalers
    names = _target_names_from_cfg(cfg)
    T = len(names)
    scalers = getattr(dm, "label_scalers", None)
    if scalers is None:
        tds = getattr(dm, "test_ds", None) or getattr(dm, "test_dataset", None)
        if tds is not None:
            scalers = getattr(tds, "label_scalers", None)

    # Try model's evaluate_manually first (on CPU)
    try:
        res = model.evaluate_manually(test_loader, scalers, False)

        def _vec(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return np.array(x, dtype=np.float64).reshape(-1)

        if isinstance(res, (list, tuple)) and len(res) >= 3:
            r2v, maev, msev = map(_vec, (res[0], res[1], res[2]))
        elif isinstance(res, dict):
            r2v = _vec(res["r2"]) if "r2" in res else None
            maev = _vec(res["mae"]) if "mae" in res else None
            msev = _vec(res["mse"]) if "mse" in res else None
        else:
            r2v = maev = msev = None

        if any(v is not None for v in (r2v, maev, msev)):
            if r2v is not None and r2v.size == 1 and T > 1:
                r2v = np.repeat(r2v, T)
            if maev is not None and maev.size == 1 and T > 1:
                maev = np.repeat(maev, T)
            if msev is not None and msev.size == 1 and T > 1:
                msev = np.repeat(msev, T)

            data = {}
            if maev is not None:
                data["mae"] = maev[: T]
            if msev is not None:
                data["mse"] = msev[: T]
            if r2v is not None:
                data["r2"] = r2v[: T]
            if data:
                return pd.DataFrame(data, index=names)
    except Exception as e:
        print(f"[warn] evaluate_manually failed in-train: {e}; falling back to streaming.")

    # Streaming fallback (CPU)
    N = np.zeros(T, dtype=np.float64)
    sum_y = np.zeros(T, dtype=np.float64)
    sum_y2 = np.zeros(T, dtype=np.float64)
    sae = np.zeros(T, dtype=np.float64)
    sse = np.zeros(T, dtype=np.float64)

    torch.set_grad_enabled(False)
    for batch in test_loader:
        # training collate usually yields (g, y) or (g, x, y)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            g = batch[0]
            y = batch[-1]  # works for (g,y) and (g,x,y)
            try:
                yp = model(g)  # many models infer features internally
            except TypeError:
                x = batch[1] if len(batch) == 3 else None
                yp = model(g, x) if x is not None else model(g)
        elif isinstance(batch, dict):
            g = (
                batch.get("graph")
                or batch.get("g")
                or batch.get("batched_graph")
                or batch.get("bg")
            )
            y = (
                batch.get("label")
                or batch.get("labels")
                or batch.get("y")
                or batch.get("target")
                or batch.get("targets")
            )
            try:
                yp = model(g)
            except TypeError:
                x = (
                    batch.get("feat")
                    or batch.get("node_feat")
                    or batch.get("atom_feat")
                    or batch.get("x")
                    or batch.get("node_features")
                )
                yp = model(g, x) if x is not None else model(g)
        else:
            raise RuntimeError(f"Unexpected batch type: {type(batch)}")

        if not isinstance(yp, torch.Tensor):
            yp = yp[0] if isinstance(yp, (list, tuple)) else yp
        if yp.ndim == 1:
            yp = yp.unsqueeze(1)
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu()
        if y.ndim == 1:
            y = y.unsqueeze(1)

        yp_np = yp.detach().cpu().numpy().astype(np.float64, copy=False)
        y_np = y.numpy().astype(np.float64, copy=False)

        yp_u = _invert_cols(yp_np, scalers)
        y_u = _invert_cols(y_np, scalers)

        err = yp_u - y_u
        N += y_u.shape[0]
        sum_y += y_u.sum(axis=0)
        sum_y2 += (y_u**2).sum(axis=0)
        sae += np.abs(err).sum(axis=0)
        sse += (err**2).sum(axis=0)

    N_safe = np.maximum(N, 1.0)
    mae = sae / N_safe
    mse = sse / N_safe
    mean_y = sum_y / N_safe
    ss_tot = sum_y2 - N_safe * (mean_y**2)
    r2 = 1.0 - (sse / (ss_tot + 1e-12))
    return pd.DataFrame({"mae": mae, "mse": mse, "r2": r2}, index=names)


# ===========================
# Main training script
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("-project_name", type=str, default="qtaim_embed_test")
    parser.add_argument("-dataset_loc", type=str, default=None)
    parser.add_argument("-dataset_test_loc", type=str, default=None)
    parser.add_argument("-log_save_dir", type=str, default="./test_logs/")
    parser.add_argument("-config", type=str, default=None)
    parser.add_argument("-wandb_entity", type=str, default="santi")
    parser.add_argument("--use_lmdb", default=False, action="store_true")
    parser.add_argument(
        "--per_task_eval",
        action="store_true",
        default=True,
        help="After fit, run per-task evaluation on test set and write JSON/CSV.",
    )

    args = parser.parse_args()

    debug = bool(args.debug)
    use_lmdb = bool(args.use_lmdb)
    project_name = args.project_name
    dataset_loc = args.dataset_loc
    dataset_test_loc = args.dataset_test_loc
    log_save_dir = args.log_save_dir
    wandb_entity = args.wandb_entity
    config_path = args.config

    if config_path is None:
        config = get_default_graph_level_config()
    else:
        config = json.load(open(config_path, "r"))

    # set log save dir (you stage this to logs_suba/<job>/logs in your batch script)
    config["dataset"]["log_save_dir"] = log_save_dir

    print(">" * 40 + "config_settings" + "<" * 40)

    if use_lmdb:
        print("using lmdbs!")
        dm = LMDBDataModule(config=config)
        # ensure target_dict matches dataset targets
        config.setdefault("model", {}).setdefault("target_dict", {})
        config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]
    else:
        if dataset_loc is not None:
            config["dataset"]["train_dataset_loc"] = dataset_loc

        if debug:
            config["dataset"]["debug"] = debug

        # normalize precision format if using "16"/"32"
        if config["optim"]["precision"] in ["16", "32"]:
            config["optim"]["precision"] = int(config["optim"]["precision"])

        dm = QTAIMGraphTaskDataModule(config=config)
        config.setdefault("model", {}).setdefault("target_dict", {})
        config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]

        if dataset_test_loc is not None:
            test_config = deepcopy(config)
            test_config["dataset"]["test_dataset_loc"] = dataset_test_loc
            dm_test = QTAIMGraphTaskDataModule(config=test_config)
            dm_test.prepare_data(stage="test")

    # Prepare data to discover feature sizes
    feature_names, feature_size = dm.prepare_data(stage="fit")
    print(feature_names, feature_size)
    config["model"]["atom_feature_size"] = feature_size["atom"]
    config["model"]["bond_feature_size"] = feature_size["bond"]
    config["model"]["global_feature_size"] = feature_size["global"]

    print(">" * 40 + "config_settings" + "<" * 40)
    for k, v in config.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))
    print(">" * 40 + "config_settings" + "<" * 40)

    model = load_graph_level_model_from_config(config["model"])
    print("model constructed!")

    with wandb.init(project=project_name) as run:
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(config["dataset"]["log_save_dir"], name="test_logs")
        logger_wb = WandbLogger(project=project_name, name="test_logs", entity=wandb_entity)
        lr_monitor = LearningRateMonitor(logging_interval="step")

        checkpoint_callback = ModelCheckpoint(
            dirpath=config["dataset"]["log_save_dir"],
            filename="model_lightning_{epoch:03d}-{val_loss:.4f}",
            monitor="val_mae",
            mode="min",
            auto_insert_metric_name=True,
            save_last=True,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=config["model"]["extra_stop_patience"],
            verbose=False,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=config["model"]["max_epochs"],
            accelerator="gpu",
            devices=config["optim"]["num_devices"],
            num_nodes=config["optim"]["num_nodes"],
            gradient_clip_val=config["optim"]["gradient_clip_val"],
            accumulate_grad_batches=config["optim"]["accumulate_grad_batches"],
            enable_progress_bar=True,
            callbacks=[early_stopping_callback, lr_monitor, log_parameters, checkpoint_callback],
            enable_checkpointing=True,
            strategy=config["optim"]["strategy"],
            default_root_dir=config["dataset"]["log_save_dir"],
            logger=[logger_tb, logger_wb],
            precision=config["optim"]["precision"],
            num_sanity_val_steps=config["optim"].get("num_sanity_val_steps", 0),
            log_every_n_steps=config["optim"].get("log_every_n_steps", 50),
        )

        run.config.update(config["dataset"])
        run.config.update(config["optim"])

        # ---- Train ----
        trainer.fit(model, dm)

        # ---- Standard test (Lightning) ----
        if use_lmdb and "test_lmdb" in config["dataset"]:
            results = trainer.test(model, dm)
            print(">" * 40 + "test_results" + "<" * 40)
            for i, res in enumerate(results):
                print(f"[test dataset {i}]")
                for k, v in res.items():
                    print(f"{k}: {v}")
            pd.to_pickle(results, Path(config["dataset"]["log_save_dir"]) / "test_results.pkl")

            # ---- Per-task metrics on CPU from best or last checkpoint ----
            log_dir = Path(config["dataset"]["log_save_dir"])  # .../<job>/logs
            run_dir = log_dir.parent                            # .../<job>/
            best_ckpt = None
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                    best_ckpt = cb.best_model_path
                    break
            if not best_ckpt:
                last = log_dir / "last.ckpt"
                best_ckpt = str(last) if last.exists() else None

            if args.per_task_eval and best_ckpt:
                tag = Path(best_ckpt).stem
                outdir = run_dir / "per_task" / tag
                df = _per_task_eval_lmdb(config, dm, best_ckpt)
                _save_per_task_df(df, outdir, tag)
                print(f"[per-task] written: {outdir}")
            else:
                print("[per-task] skipped (no checkpoint found or flag disabled).")

        else:
            # Non-LMDB branch (kept from original script)
            if config["dataset"]["test_prop"] > 0.0:
                trainer.test(model, dm)

            if dataset_test_loc is not None:
                batch_graph, batch_labels = next(iter(dm_test.test_dataloader()))
                scalers = dm.full_dataset.label_scalers

                if config["dataset"]["per_atom"] is True:
                    (
                        mean_mae_test,
                        mean_rmse_test,
                        ewt_prop_test,
                        preds_unscaled,
                        labels_unscaled,
                    ) = model.evaluate_manually(
                        batch_graph=batch_graph,
                        batch_label=batch_labels,
                        scaler_list=scalers,
                        per_atom=True,
                    )
                    print(">" * 40 + "test_results" + "<" * 40)
                    print("mean_mae_test: ", mean_mae_test.numpy())
                    print("mean_rmse_test: ", mean_rmse_test.numpy())
                    print("ewt_prop_test: ", ewt_prop_test.numpy())
                    results = {
                        "mean_mae_test": mean_mae_test.numpy(),
                        "mean_rmse_test": mean_rmse_test.numpy(),
                        "ewt_prop_test": ewt_prop_test.numpy(),
                        "preds_unscaled": preds_unscaled.numpy(),
                        "labels_unscaled": labels_unscaled.numpy(),
                    }
                else:
                    (r2_val, mae_val, mse_val, preds_unscaled, labels_unscaled,) = model.evaluate_manually(
                        batch_graph, batch_labels, scalers, per_atom=False
                    )
                    print(">" * 40 + "test_results" + "<" * 40)
                    print("r2_test: ", r2_val.numpy())
                    print("mae_test: ", mae_val.numpy())
                    print("mse_test: ", mse_val.numpy())
                    results = {
                        "r2_val": r2_val.numpy(),
                        "mae_val": mae_val.numpy(),
                        "mse_val": mse_val.numpy(),
                        "preds_unscaled": preds_unscaled.numpy(),
                        "labels_unscaled": labels_unscaled.numpy(),
                    }
                pd.to_pickle(results, Path(config["dataset"]["log_save_dir"]) / "test_results.pkl")

    wandb.finish()
