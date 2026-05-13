import wandb, argparse, torch, json
from copy import deepcopy
import numpy as np
import pandas as pd

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


torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def evaluate_split(model, dataset, config, scalers, split_name):
    """Evaluate model on a dataset split, return preds and labels in real units."""
    from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask
    loader = DataLoaderMoleculeGraphTask(
        dataset=dataset,
        batch_size=config["dataset"]["test_batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    r2, mae, mse, preds, labels = model.evaluate_manually(
        loader, scalers, per_atom=False
    )
    preds  = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    r2     = r2.cpu().numpy()
    mae    = mae.cpu().numpy()
    mse    = mse.cpu().numpy()

    print(f"\n  {split_name.upper()} RESULTS:")
    print(f"  {'Target':<8} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print(f"  {'-'*38}")
    targets = ["HOMO", "LUMO", "Gap"]
    for i, t in enumerate(targets):
        rmse_val = float(np.sqrt(mse[i]))
        print(f"  {t:<8} {mae[i]:>10.5f} {rmse_val:>10.5f} {r2[i]:>8.4f}")
    avg_mae = mae.mean()
    print(f"  {'avg':<8} {avg_mae:>10.5f}")

    return preds, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",           default=False, action="store_true")
    parser.add_argument("-project_name",     type=str, default="qtaim_embed_test")
    parser.add_argument("-dataset_loc",      type=str, default=None)
    parser.add_argument("-dataset_test_loc", type=str, default=None)
    parser.add_argument("-log_save_dir",     type=str, default="./test_logs/")
    parser.add_argument("-config",           type=str, default=None)
    parser.add_argument("-wandb_entity",     type=str, default="santi")
    parser.add_argument("-model_name",       type=str, default="model",
                        help="short label for this model: baseline | qtaim | elf")
    parser.add_argument("--use_lmdb",        default=False, action="store_true")
    parser.add_argument("--skip_eval",       default=False, action="store_true",
                        help="skip eval_utils (use if matplotlib not available)")

    args = parser.parse_args()

    debug            = bool(args.debug)
    use_lmdb         = bool(args.use_lmdb)
    project_name     = args.project_name
    dataset_loc      = args.dataset_loc
    dataset_test_loc = args.dataset_test_loc
    log_save_dir     = args.log_save_dir
    wandb_entity     = args.wandb_entity
    model_name       = args.model_name
    config           = args.config

    if config is None:
        config = get_default_graph_level_config()
    else:
        config = json.load(open(config, "r"))

    config["dataset"]["log_save_dir"] = log_save_dir

    print(">" * 40 + "config_settings" + "<" * 40)

    if use_lmdb:
        print("using lmdbs!")
        dm = LMDBDataModule(config=config)
        config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]
    else:
        if dataset_loc is not None:
            config["dataset"]["train_dataset_loc"] = dataset_loc
        extra_keys = config["dataset"]["extra_keys"]

        if debug:
            config["dataset"]["debug"] = debug

        if config["optim"]["precision"] == "16" or config["optim"]["precision"] == "32":
            config["optim"]["precision"] = int(config["optim"]["precision"])

        dm = QTAIMGraphTaskDataModule(config=config)
        config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]

        if dataset_test_loc is not None:
            test_config = deepcopy(config)
            test_config["dataset"]["test_dataset_loc"] = dataset_test_loc
            dm_test = QTAIMGraphTaskDataModule(config=test_config)
            dm_test.prepare_data(stage="test")

    feature_names, feature_size = dm.prepare_data(stage="fit")
    print(feature_names, feature_size)
    config["model"]["atom_feature_size"]   = feature_size["atom"]
    config["model"]["bond_feature_size"]   = feature_size["bond"]
    config["model"]["global_feature_size"] = feature_size["global"]
    config["dataset"]["feature_names"]     = feature_names

    print(">" * 40 + "config_settings" + "<" * 40)
    for k, v in config.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))
    print(">" * 40 + "config_settings" + "<" * 40)

    model = load_graph_level_model_from_config(config["model"])
    print("model constructed!")

    with wandb.init(project=project_name) as run:
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(
            config["dataset"]["log_save_dir"], name="test_logs"
        )
        logger_wb = WandbLogger(
            project=project_name, name="test_logs", entity=wandb_entity
        )
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
            monitor="val_mae",
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
            callbacks=[
                early_stopping_callback,
                lr_monitor,
                log_parameters,
                checkpoint_callback,
            ],
            enable_checkpointing=True,
            strategy=config["optim"]["strategy"],
            default_root_dir=config["dataset"]["log_save_dir"],
            logger=[logger_tb, logger_wb],
            precision=config["optim"]["precision"],
        )

        run.config.update(config["dataset"])
        run.config.update(config["optim"])

        trainer.fit(model, dm)

        if use_lmdb:
            if "test_lmdb" in config["dataset"]:
                trainer.test(model, dm)
        else:
            if config["dataset"]["test_prop"] > 0.0:
                trainer.test(model, dm)

        # ---- evaluate all splits ----
        scalers = dm.full_dataset.label_scalers

        preds_val, labels_val     = None, None
        preds_train, labels_train = None, None
        preds_test, labels_test   = None, None

        # train set — use dm train dataset directly (already loaded)
        print("\n" + "="*50)
        print("EVALUATING ON TRAIN SET (first 5000 samples)")
        print("="*50)
        try:
            from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask
            import random as _random
            _random.seed(42)
            train_ds = dm.train_dataset
            n_sample = min(5000, len(train_ds))
            sample_idx = _random.sample(range(len(train_ds)), n_sample)
            # build a simple subset using the dataset items directly
            class _Subset:
                def __init__(self, ds, idx): self.ds=ds; self.idx=idx
                def __len__(self): return len(self.idx)
                def __getitem__(self, i): return self.ds[self.idx[i]]
            train_subset = _Subset(train_ds, sample_idx)
            train_loader = DataLoaderMoleculeGraphTask(
                dataset=train_subset,
                batch_size=config["dataset"]["test_batch_size"],
                shuffle=False,
                num_workers=0,
            )
            r2_tr, mae_tr, mse_tr, preds_train, labels_train = model.evaluate_manually(
                train_loader, scalers, per_atom=False
            )
            preds_train  = preds_train.cpu().numpy()
            labels_train = labels_train.cpu().numpy()
            targets_list = ["HOMO", "LUMO", "Gap"]
            print(f"\n  TRAIN (sample n={n_sample}) RESULTS:")
            print(f"  {'Target':<8} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
            print(f"  {'-'*38}")
            for i, t in enumerate(targets_list):
                rmse_v = float(np.sqrt(mse_tr.cpu().numpy()[i]))
                print(f"  {t:<8} {mae_tr.cpu().numpy()[i]:>10.5f} "
                      f"{rmse_v:>10.5f} {r2_tr.cpu().numpy()[i]:>8.4f}")
        except Exception as e:
            print(f"  ⚠  Train evaluation failed: {e}")
            import traceback; traceback.print_exc()

        # val set
        print("\n" + "="*50)
        print("EVALUATING ON VALIDATION SET")
        print("="*50)
        try:
            from qtaim_embed.core.dataset import HeteroGraphGraphLabelDataset
            val_dataset = HeteroGraphGraphLabelDataset(
                file=config["dataset"]["val_dataset_loc"],
                allowed_ring_size=config["dataset"]["allowed_ring_size"],
                allowed_charges=config["dataset"]["allowed_charges"],
                allowed_spins=config["dataset"]["allowed_spins"],
                self_loop=config["dataset"]["self_loop"],
                extra_keys=config["dataset"]["extra_keys"],
                element_set=config["dataset"]["element_set"],
                target_list=config["dataset"]["target_list"],
                extra_dataset_info=config["dataset"]["extra_dataset_info"],
                debug=False,
                log_scale_features=config["dataset"]["log_scale_features"],
                log_scale_targets=config["dataset"]["log_scale_targets"],
                bond_key=config["dataset"]["bond_key"],
                map_key=config["dataset"]["map_key"],
                standard_scale_features=config["dataset"]["standard_scale_features"],
                standard_scale_targets=config["dataset"]["standard_scale_targets"],
                verbose=False,
                fit_scalers=False,
                feature_scalers=dm.full_dataset.feature_scalers,
                label_scalers=scalers,
            )
            preds_val, labels_val = evaluate_split(
                model, val_dataset, config, scalers, "val"
            )
            # log val metrics to wandb
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            targets = ["HOMO", "LUMO", "Gap"]
            for i, t in enumerate(targets):
                mae  = mean_absolute_error(labels_val[:,i], preds_val[:,i])
                rmse = float(np.sqrt(mean_squared_error(labels_val[:,i], preds_val[:,i])))
                r2   = r2_score(labels_val[:,i], preds_val[:,i])
                wandb.log({
                    f"val_final/mae_{t}": mae,
                    f"val_final/rmse_{t}": rmse,
                    f"val_final/r2_{t}": r2,
                })
            wandb.log({
                "val_final/mae_avg": mean_absolute_error(
                    labels_val, preds_val,
                    multioutput="uniform_average"
                )
            })
        except Exception as e:
            print(f"  ⚠  Val evaluation failed: {e}")

        # test set
        if dataset_test_loc is not None:
            print("\n" + "="*50)
            print("EVALUATING ON TEST SET")
            print("="*50)
            from qtaim_embed.core.dataset import HeteroGraphGraphLabelDataset
            from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask

            test_dataset = HeteroGraphGraphLabelDataset(
                file=dataset_test_loc,
                allowed_ring_size=config["dataset"]["allowed_ring_size"],
                allowed_charges=config["dataset"]["allowed_charges"],
                allowed_spins=config["dataset"]["allowed_spins"],
                self_loop=config["dataset"]["self_loop"],
                extra_keys=config["dataset"]["extra_keys"],
                element_set=config["dataset"]["element_set"],
                target_list=config["dataset"]["target_list"],
                extra_dataset_info=config["dataset"]["extra_dataset_info"],
                debug=False,
                log_scale_features=config["dataset"]["log_scale_features"],
                log_scale_targets=config["dataset"]["log_scale_targets"],
                bond_key=config["dataset"]["bond_key"],
                map_key=config["dataset"]["map_key"],
                standard_scale_features=config["dataset"]["standard_scale_features"],
                standard_scale_targets=config["dataset"]["standard_scale_targets"],
                verbose=False,
                fit_scalers=False,
                feature_scalers=dm.full_dataset.feature_scalers,
                label_scalers=scalers,
            )
            preds_test, labels_test = evaluate_split(
                model, test_dataset, config, scalers, "test"
            )

            # save test results pkl (backward compatible)
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            targets = ["HOMO", "LUMO", "Gap"]
            r2_arr  = np.array([r2_score(labels_test[:,i], preds_test[:,i])
                                 for i in range(3)])
            mae_arr = np.array([mean_absolute_error(labels_test[:,i], preds_test[:,i])
                                 for i in range(3)])
            mse_arr = np.array([mean_squared_error(labels_test[:,i], preds_test[:,i])
                                 for i in range(3)])
            results = {
                "r2_val": r2_arr, "mae_val": mae_arr, "mse_val": mse_arr,
                "r2_mean": r2_arr.mean(), "mae_mean": mae_arr.mean(),
                "mse_mean": mse_arr.mean(),
                "preds_unscaled": preds_test,
                "labels_unscaled": labels_test,
            }
            pd.to_pickle(results, config["dataset"]["log_save_dir"] + "test_results.pkl")

            # print formatted table
            print(f"\n{'>'*40}test_results{'<'*40}")
            print(f"r2_mean : {r2_arr.mean():.4f}")
            print(f"mae_mean: {mae_arr.mean():.4f}")
            print(f"mse_mean: {mse_arr.mean():.4f}")
            print("\nPer-task Metrics:")
            for i, t in enumerate(targets):
                print(f"Task {i} ({t}): R2={r2_arr[i]:.5f}, "
                      f"MAE={mae_arr[i]:.5f}, RMSE={float(np.sqrt(mse_arr[i])):.5f}")

            # log to wandb
            wandb.log({
                "test/r2_mean": r2_arr.mean(),
                "test/mae_mean": mae_arr.mean(),
                "test/mse_mean": mse_arr.mean(),
            })
            for i, t in enumerate(targets):
                wandb.log({
                    f"test/r2_{t}": r2_arr[i],
                    f"test/mae_{t}": mae_arr[i],
                    f"test/mse_{t}": mse_arr[i],
                    f"test/rmse_{t}": float(np.sqrt(mse_arr[i])),
                })

        # ---- run eval_utils ----
        if not args.skip_eval and preds_test is not None:
            try:
                import sys, os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                from eval_utils import run_eval
                run_eval(
                    model=model,
                    trainer=trainer,
                    dm=dm,
                    config=config,
                    model_name=model_name,
                    preds_test=preds_test,
                    labels_test=labels_test,
                    preds_val=preds_val,
                    labels_val=labels_val,
                    preds_train=preds_train,
                    labels_train=labels_train,
                )
            except Exception as e:
                print(f"\n  ⚠  eval_utils failed: {e}")
                import traceback; traceback.print_exc()

    run.finish()
