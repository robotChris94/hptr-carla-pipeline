import torch
from omegaconf import DictConfig
from torch.serialization import safe_globals

ckpt_path = "hptr_core/models/model.ckpt"

# 允許載入含有 DictConfig 的 ckpt
with safe_globals({"omegaconf.dictconfig.DictConfig": DictConfig}):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# 印出超參數
hparams = checkpoint.get("hyper_parameters", {})
print("✅ hyper_parameters loaded from .ckpt:")
for k, v in hparams.items():
    print(f"{k}: {v}")
