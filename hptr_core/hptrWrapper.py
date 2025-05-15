# hptr_wrapper.py
# 功能：載入 HPTR 預訓練模型並進行線上預測

import sys
sys.path.append("..")
import torch
import pytorch_lightning as pl
from hptr_core.models.sc_relative import SceneCentricRelative # 根據你的 config 推出
from omegaconf import OmegaConf


class HPTRWrapper:
    def __init__(self, model_path, device='cpu'):
        self.device = device

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        #state_dict = ckpt['state_dict']
        state_dict = {k: v for k, v in ckpt['state_dict'].items()
              if not k.startswith("model.intra_class_encoder.fc_tl.")}

        # 重新命名 key：移除 "model." 前綴
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "")  # 如有需要可調整為更完整的匹配邏輯
            new_state_dict[new_key] = v

        # 用手動 config 初始化模型（這段直接複製你原本寫好的初始化參數）
        model = SceneCentricRelative(
            hidden_dim=256,
            agent_attr_dim=68,
            map_attr_dim=38,
            tl_attr_dim=38,
            pl_aggr=False,
            n_tgt_knn=36,
            rpe_mode='pe_xy_yaw',
            tf_cfg=OmegaConf.create({
                'n_head': 4,
                'dropout_p': 0.1,
                'norm_first': True,
                'apply_q_rpe': False,
                'bias': True,
            }),
            intra_class_encoder=OmegaConf.create({
                'n_layer_mlp': 3,
                'mlp_cfg': {
                    'end_layer_activation': True,
                    'use_layernorm': False,
                    'use_batchnorm': False,
                    'dropout_p': None,
                },
                'n_layer_tf_map': 6,
                'n_layer_tf_tl': -1,
                'n_layer_tf_agent': -1,
            }),
            decoder_remove_ego_agent=False,
            n_decoders=1,
            decoder=OmegaConf.create({
                '_target_': 'hptr_core.models.sc_relative.Decoder',
                'n_pred': 6,
                'tf_n_layer': 2,
                'k_reinforce_tl': 2,
                'k_reinforce_agent': 4,
                'k_reinforce_all': -1,
                'k_reinforce_anchor': 10,
                'n_latent_query': -1,
                'latent_query': {
                    'use_agent_type': False,
                    'mode_emb': 'linear',
                    'mode_init': 'xavier',
                    'scale': 1.0,
                },
                'latent_query_use_tf_decoder': False,
                'multi_modal_anchors': {
                    'use_agent_type': True,
                    'mode_emb': 'linear',
                    'mode_init': 'randn',
                    'scale': 5.0,
                },
                'anchor_self_attn': True,
                'mlp_head': {
                    'predictions': ['pos', 'cov3', 'spd', 'vel', 'yaw_bbox'],
                    'use_agent_type': False,
                    'multi_modal_ensemble': False,
                    'flatten_conf_head': False,
                    'out_mlp_layernorm': False,
                    'out_mlp_batchnorm': False,
                    'n_step_future': 60,
                },
                'use_attr_for_multi_modal': False,
                'use_multi_modal_mlp': False,
                'use_vmap': True,
            }),
            dist_limit_map=1500,
            dist_limit_tl=1000,
            dist_limit_agent=[1500, 500, 1000],
        )

        model.load_state_dict(new_state_dict, strict=False)

        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, batch: dict):
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(**batch)

            # 解包
            agent_score, conf, pred = output

            # debug info
            print("🧪 [Debug] output type:", type(output))
            print("🧪 [Debug] output length:", len(output))
            for i, x in enumerate(output):
                print(f" - output[{i}].shape: {x.shape}")

            return conf.cpu(), pred.cpu()

    

# 用法範例：
# hptr = HPTRWrapper("./models/hptr_model.pt")
# pred = hptr.predict(polyline_tensor)
