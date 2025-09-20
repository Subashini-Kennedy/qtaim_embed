import wandb, argparse, torch, json
from copy import deepcopy
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from qtaim_embed.core.datamodule import QTAIMGraphTaskDataModule, LMDBDataModule
from qtaim_embed.models.utils import LogParameters, load_graph_level_model_from_config
from qtaim_embed.utils.data import get_default_graph_level_config

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

# why: Lightning wants int for plain 16/32
def _normalize_precision(cfg):
    p = cfg["optim"].get("precision")
    if p in ("16","32"):
        cfg["optim"]["precision"] = int(p)

def _print_effective_io(cfg, use_lmdb, dataset_loc, dataset_test_loc):
    ds = cfg.get("dataset", {})
    print("\n==== Effective IO ====")
    print("use_lmdb:", use_lmdb)
    if use_lmdb:
        for k in ("train_lmdb","val_lmdb","test_lmdb"):
            print(f"{k}: {ds.get(k)}")
    else:
        print("train_dataset_loc   :", ds.get("train_dataset_loc") or dataset_loc)
        print("external test (CLI) :", dataset_test_loc or ds.get("test_dataset_loc"))
        print("val_prop/test_prop  :", ds.get("val_prop"), ds.get("test_prop"))
    print("log_save_dir:", ds.get("log_save_dir"))
    print("======================\n")

# compute per-task + macro average metrics
def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_names):
    eps = 1e-12
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    mse = np.mean(err**2, axis=0)
    rmse = np.sqrt(mse)
    # R^2 per task
    r2 = []
    for j in range(y_true.shape[1]):
        sse = np.sum((err[:, j])**2)
        yj = y_true[:, j]
        sst = np.sum((yj - np.mean(yj))**2)
        r2.append(1.0 - sse / (sst + eps))
    r2 = np.array(r2)
    # macro averages
    metrics_avg = {
        "mae_avg": float(np.mean(mae)),
        "mse_avg": float(np.mean(mse)),
        "rmse_avg": float(np.mean(rmse)),
        "r2_avg": float(np.mean(r2)),
    }
    # per-task dict
    metrics_per = {}
    for name, m_mae, m_mse, m_rmse, m_r2 in zip(task_names, mae, mse, rmse, r2):
        metrics_per[f"mae_{name}"] = float(m_mae)
        metrics_per[f"mse_{name}"] = float(m_mse)
        metrics_per[f"rmse_{name}"] = float(m_rmse)
        metrics_per[f"r2_{name}"] = float(m_r2)
    return metrics_avg, metrics_per

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
    args = parser.parse_args()

    debug = bool(args.debug)
    use_lmdb = bool(args.use_lmdb)
    project_name = args.project_name
    dataset_loc = args.dataset_loc
    dataset_test_loc = args.dataset_test_loc
    log_save_dir = args.log_save_dir
    wandb_entity = args.wandb_entity
    cfg = args.config

    if cfg is None:
        cfg = get_default_graph_level_config()
    else:
        cfg = json.load(open(cfg, "r"))

    # reproducible split for val
    pl.seed_everything(cfg.get("dataset", {}).get("seed", 42), workers=True)

    cfg["dataset"]["log_save_dir"] = log_save_dir
    _normalize_precision(cfg)

    if use_lmdb:
        print("using lmdbs!")
        dm = LMDBDataModule(config=cfg)
        cfg["model"]["target_dict"]["global"] = cfg["dataset"]["target_list"]
        dm_test = None
    else:
        if dataset_loc is not None:
            cfg["dataset"]["train_dataset_loc"] = dataset_loc
        if debug:
            cfg["dataset"]["debug"] = debug
        dm = QTAIMGraphTaskDataModule(config=cfg)
        cfg["model"]["target_dict"]["global"] = cfg["dataset"]["target_list"]

        dm_test = None
        # prefer external test datamodule when provided
        ext_test_path = dataset_test_loc or cfg["dataset"].get("test_dataset_loc")
        if ext_test_path:
            test_config = deepcopy(cfg)
            test_config["dataset"]["test_dataset_loc"] = ext_test_path
            dm_test = QTAIMGraphTaskDataModule(config=test_config)
            dm_test.prepare_data(stage="test")

    _print_effective_io(cfg, use_lmdb, dataset_loc, dataset_test_loc)

    feature_names, feature_size = dm.prepare_data(stage="fit")
    cfg["model"]["atom_feature_size"] = feature_size["atom"]
    cfg["model"]["bond_feature_size"] = feature_size["bond"]
    cfg["model"]["global_feature_size"] = feature_size["global"]

    print(">" * 40 + "config_settings" + "<" * 40)
    for k, v in cfg.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))
    print(">" * 40 + "config_settings" + "<" * 40)

    model = load_graph_level_model_from_config(cfg["model"])
    print("model constructed!")

    with wandb.init(project=project_name) as run:
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(cfg["dataset"]["log_save_dir"], name="tb")
        logger_wb = WandbLogger(project=project_name, name="train", entity=wandb_entity)
        lr_monitor = LearningRateMonitor(logging_interval="step")

        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg["dataset"]["log_save_dir"],
            filename="model_lightning_{epoch:03d}-{val_loss:.4f}",
            monitor="val_mae",
            mode="min",
            auto_insert_metric_name=True,
            save_last=True,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=cfg["model"]["extra_stop_patience"],
            verbose=False,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=cfg["model"]["max_epochs"],
            accelerator="gpu",
            devices=cfg["optim"]["num_devices"],
            num_nodes=cfg["optim"]["num_nodes"],
            gradient_clip_val=cfg["optim"]["gradient_clip_val"],
            accumulate_grad_batches=cfg["optim"]["accumulate_grad_batches"],
            enable_progress_bar=True,
            callbacks=[early_stopping_callback, lr_monitor, log_parameters, checkpoint_callback],
            enable_checkpointing=True,
            strategy=cfg["optim"]["strategy"],
            default_root_dir=cfg["dataset"]["log_save_dir"],
            logger=[logger_tb, logger_wb],
            precision=cfg["optim"]["precision"],
            check_val_every_n_epoch=cfg["optim"].get("check_val_every_n_epoch", 1),
            num_sanity_val_steps=cfg["optim"].get("num_sanity_val_steps", 0),
            log_every_n_steps=cfg["optim"].get("log_every_n_steps", 50),
        )

        run.config.update(cfg["dataset"])
        run.config.update(cfg["optim"])

        trainer.fit(model, dm)

        # --- Test selection: prefer external test datamodule when present ---
        if use_lmdb:
            if "test_lmdb" in cfg["dataset"]:
                trainer.test(model, dm)
        else:
            if dm_test is not None:
                trainer.test(model, dm_test)
            elif cfg["dataset"].get("test_prop", 0.0) > 0.0:
                trainer.test(model, dm)

        # --- Manual test metrics (per-task + average) on external test ---
        if not use_lmdb and dm_test is not None:
            # why: compute per-task metrics in unscaled space
            batch_graph, batch_labels = next(iter(dm_test.test_dataloader()))
            scalers = dm.full_dataset.label_scalers
            (
                r2_val,
                mae_val,
                mse_val,
                preds_unscaled,
                labels_unscaled,
            ) = model.evaluate_manually(batch_graph, batch_labels, scalers, per_atom=False)

            y_true = labels_unscaled.numpy()
            y_pred = preds_unscaled.numpy()
            task_names = cfg["model"]["target_dict"]["global"]

            metrics_avg, metrics_per = _regression_metrics(y_true, y_pred, task_names)

            print(">" * 40 + "test_results (external)" + "<" * 40)
            for k, v in {**metrics_avg, **metrics_per}.items():
                print(f"{k}: {v:.6f}")

            # log to W&B summary
            wandb.run.summary.update({**metrics_avg, **metrics_per})

            # save full artifact
            results = {
                "metrics_avg": metrics_avg,
                "metrics_per_task": metrics_per,
                "preds_unscaled": y_pred,
                "labels_unscaled": y_true,
                "task_names": task_names,
            }
            pd.to_pickle(results, cfg["dataset"]["log_save_dir"] + "test_results.pkl")

    wandb.finish()