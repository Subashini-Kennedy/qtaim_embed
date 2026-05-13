import json
from qtaim_embed.core.datamodule import QTAIMGraphTaskDataModule

config = json.load(open("settings_qm9_nonlmdb_43k_withval_baseline.json"))
config["dataset"]["debug"] = True  # only loads ~100 molecules
feature_names, feature_size = QTAIMGraphTaskDataModule(config=config).prepare_data(stage="fit")

print("Feature sizes:", feature_size)
print("Feature names:", feature_names)