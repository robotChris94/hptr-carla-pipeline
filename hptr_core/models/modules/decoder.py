from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from omegaconf import DictConfig
from hptr_core.models.modules.transformer import TransformerBlock
from hptr_core.models.modules.multi_modal import MultiModalAnchors
from hptr_core.models.modules.mlp import MLP

class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        use_vmap: bool,
        d_rpe: int,
        n_pred: int,
        mlp_head: DictConfig,
        multi_modal_anchors: DictConfig,
        tf_n_layer: int,
        tf_cfg: DictConfig,
        agent_attr_dim: int,
        k_reinforce_tl: float,
        k_reinforce_agent: float,
        k_reinforce_anchor: float,
        k_reinforce_all: float,
        n_latent_query: float,
        latent_query_use_tf_decoder: bool,
        latent_query: DictConfig,
        use_attr_for_multi_modal: bool,
        anchor_self_attn: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_pred = n_pred
        self.k_reinforce_tl = k_reinforce_tl
        self.k_reinforce_agent = k_reinforce_agent
        self.k_reinforce_anchor = k_reinforce_anchor
        self.k_reinforce_all = k_reinforce_all
        self.n_latent_query = n_latent_query
        self.use_attr_for_multi_modal = use_attr_for_multi_modal
        self.anchor_self_attn = anchor_self_attn
        self.latent_query_use_tf_decoder = latent_query_use_tf_decoder

        if self.k_reinforce_tl > 0:
            self.tf_reinforce_tl = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                    for _ in range(tf_n_layer)
                ]
            )

        if self.k_reinforce_agent > 0:
            self.tf_reinforce_agent = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                    for _ in range(tf_n_layer)
                ]
            )

        if self.k_reinforce_all > 0:
            self.tf_reinforce_all = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        decoder_self_attn=False,
                        **tf_cfg,
                    )
                    for _ in range(tf_n_layer)
                ]
            )

        # anchor based approaches
        emb_dim = agent_attr_dim if self.use_attr_for_multi_modal else hidden_dim
        self.anchors = MultiModalAnchors(hidden_dim=hidden_dim, emb_dim=emb_dim, n_pred=n_pred, **multi_modal_anchors)
        output_dim = 60 * n_pred  # 假設你要預測 pos/vel/yaw 等
        self.mlp_head = MLP(
            fc_dims=[hidden_dim, 256, output_dim],
            dropout_p=mlp_head.get("dropout_p", None),
            end_layer_activation=mlp_head.get("end_layer_activation", True),
            use_layernorm=mlp_head.get("use_layernorm", False),
            use_batchnorm=mlp_head.get("use_batchnorm", False),
        )

        if self.k_reinforce_anchor > 0:
            if self.n_latent_query > 0:
                self.latent_query = MultiModalAnchors(
                    hidden_dim=hidden_dim, emb_dim=emb_dim, n_pred=self.n_latent_query, **latent_query
                )
                if self.latent_query_use_tf_decoder:
                    self.tf_latent_query = TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        n_layer=tf_n_layer,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                else:
                    self.tf_latent_cross = TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, d_rpe=d_rpe, n_layer=1, **tf_cfg
                    )
                    self.tf_latent_self = TransformerBlock(
                        d_model=hidden_dim, d_feedforward=hidden_dim * 4, n_layer=tf_n_layer, **tf_cfg
                    )

                self.tf_reinforce_anchor = TransformerBlock(
                    d_model=hidden_dim,
                    d_feedforward=hidden_dim * 4,
                    n_layer=tf_n_layer,
                    decoder_self_attn=anchor_self_attn,
                    **tf_cfg,
                )
            else:
                self.tf_reinforce_anchor = TransformerBlock(
                    d_model=hidden_dim,
                    d_feedforward=hidden_dim * 4,
                    d_rpe=d_rpe,
                    n_layer=tf_n_layer,
                    decoder_self_attn=anchor_self_attn,
                    **tf_cfg,
                )
        else:
            if self.anchor_self_attn:
                self.tf_anchor_self = TransformerBlock(
                    d_model=hidden_dim,
                    d_feedforward=hidden_dim * 4,
                    n_layer=tf_n_layer,
                    decoder_self_attn=False,
                    **tf_cfg,
                )

    def forward(
        self,
        agent_type: Tensor,
        agent_valid: Tensor,
        agent_attr: Tensor,
        agent_emb: Tensor,
        tl_valid: Tensor,
        tl_emb: Tensor,
        map_valid: Tensor,
        map_emb: Tensor,
        knn_idx_tl2self: Optional[Tensor],
        knn_invalid_tl2self: Optional[Tensor],
        knn_rpe_tl2self: Optional[Tensor],
        knn_idx_tl2map: Optional[Tensor],
        knn_invalid_tl2map: Optional[Tensor],
        knn_rpe_tl2map: Optional[Tensor],
        knn_idx_agent2self: Optional[Tensor],
        knn_invalid_agent2self: Optional[Tensor],
        knn_rpe_agent2self: Optional[Tensor],
        knn_idx_agent2maptl: Optional[Tensor],
        knn_invalid_agent2maptl: Optional[Tensor],
        knn_rpe_agent2maptl: Optional[Tensor],
        knn_idx_agent2all: Optional[Tensor],
        knn_invalid_agent2all: Optional[Tensor],
        knn_rpe_agent2all: Optional[Tensor],
        knn_idx_all2all: Optional[Tensor],
        knn_invalid_all2all: Optional[Tensor],
        knn_rpe_all2all: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            agent_type: [n_scene, n_agent, 3]
            agent_valid: [n_scene, n_agent]
            agent_attr: [n_scene, n_agent, agent_attr_dim], used if anchor_add_to_attr, assert pl_aggr
            agent_emb: [n_scene, n_agent, hidden_dim]
            tl_valid: [n_scene, n_tl]
            tl_emb: [n_scene, n_tl, hidden_dim]
            map_valid: [n_scene, n_map]
            map_emb: [n_scene, n_map, hidden_dim]

            if self.k_reinforce_tl > 0: # tl_emb
                knn_idx_tl2self: [n_scene, n_tl, n_knn_tl2self]
                knn_invalid_tl2self: [n_scene, n_tl, n_knn_tl2self]
                knn_rpe_tl2self: [n_scene, n_tl, n_knn_tl2self, d_rpe]
                knn_idx_tl2map: [n_scene, n_tl, n_knn_tl2map]
                knn_invalid_tl2map: [n_scene, n_tl, n_knn_tl2map]
                knn_rpe_tl2map: [n_scene, n_tl, n_knn_tl2map, d_rpe]

            if self.k_reinforce_agent > 0: # agent_emb
                knn_idx_agent2self: [n_scene, n_agent, n_knn_agent2self]
                knn_invalid_agent2self: [n_scene, n_agent, n_knn_agent2self]
                knn_rpe_agent2self: [n_scene, n_agent, n_knn_agent2self, d_rpe]
                knn_idx_agent2maptl: [n_scene, n_agent, n_knn_agent2maptl]
                knn_invalid_agent2maptl: [n_scene, n_agent, n_knn_agent2maptl]
                knn_rpe_agent2maptl: [n_scene, n_agent, n_knn_agent2maptl, d_rpe]

            if self.k_reinforce_anchor > 0: # anchor_emb
                knn_idx_agent2all: [n_scene, n_agent, n_knn_agent2all]
                knn_invalid_agent2all: [n_scene, n_agent, n_knn_agent2all]
                knn_rpe_agent2all: [n_scene, n_agent, n_knn_agent2all, d_rpe]

            if self.k_reinforce_all > 0:
                knn_idx_all2all: [n_scene, n_map+n_tl+n_agent, n_knn_all2all]
                knn_invalid_all2all: [n_scene, n_map+n_tl+n_agent, n_knn_all2all]
                knn_rpe_all2all: [n_scene, n_map+n_tl+n_agent, n_knn_all2all]

        Returns:
            conf: [n_decoders, n_scene, n_agent, n_pred]
            pred: [n_decoders, n_scene, n_agent, n_pred, n_step_future, pred_dim]
        """
        n_scene, n_agent = agent_valid.shape
        n_tl = tl_valid.shape[1]
        n_map = map_valid.shape[1]
        _idx_scene = torch.arange(n_scene)[:, None, None]  # [n_scene, 1, 1]
        _idx_agent = torch.arange(n_agent)[None, :, None]  # [1, n_agent, 1]

        # ! reinforce tl_emb by attending to map_emb
        if self.k_reinforce_tl > 0:
            _idx_tl = torch.arange(n_tl)[None, :, None]  # [1, n_tl, 1]
            tl_invalid = ~tl_valid
            _tgt = map_emb.unsqueeze(1).expand(-1, n_tl, -1, -1)
            if knn_idx_tl2map is not None:
                _tgt = _tgt[_idx_scene, _idx_tl, knn_idx_tl2map]
            for mod in self.tf_reinforce_tl:
                _decoder_tgt = tl_emb.unsqueeze(1).expand(-1, n_tl, -1, -1)
                if knn_idx_tl2self is not None:
                    _decoder_tgt = _decoder_tgt[_idx_scene, _idx_tl, knn_idx_tl2self]
                tl_emb, _ = mod(
                    src=tl_emb,  # [n_scene, n_tl, hidden_dim]
                    src_padding_mask=tl_invalid,  # [n_scene, n_tl]
                    tgt=_tgt,  # [n_scene, n_tl, n_knn_tl2map, hidden_dim]
                    tgt_padding_mask=knn_invalid_tl2map,  # [n_scene, n_tl, n_knn_tl2map]
                    rpe=knn_rpe_tl2map,  # [n_scene, n_tl, n_knn_tl2map, d_rpe]
                    decoder_tgt=_decoder_tgt,  # [n_scene, n_tl, n_knn_tl2self, hidden_dim]
                    decoder_tgt_padding_mask=knn_invalid_tl2self,  # [n_scene, n_tl, n_knn_tl2self]
                    decoder_rpe=knn_rpe_tl2self,  # [n_scene, n_tl, n_knn_tl2self, d_rpe]
                )

        # ! reinforce agent_emb by attending to map_emb and tl_emb
        if self.k_reinforce_agent > 0:
            agent_invalid = ~agent_valid
            _tgt = torch.cat([map_emb, tl_emb], dim=1).unsqueeze(1).expand(-1, n_agent, -1, -1)
            if knn_idx_agent2maptl is not None:
                _tgt = _tgt[_idx_scene, _idx_agent, knn_idx_agent2maptl]
            for mod in self.tf_reinforce_agent:
                _decoder_tgt = agent_emb.unsqueeze(1).expand(-1, n_agent, -1, -1)
                if knn_idx_agent2self is not None:
                    _decoder_tgt = _decoder_tgt[_idx_scene, _idx_agent, knn_idx_agent2self]
                agent_emb, _ = mod(
                    src=agent_emb,  # [n_scene, n_agent, hidden_dim]
                    src_padding_mask=agent_invalid,  # [n_scene, n_agent]
                    tgt=_tgt,  # [n_scene, n_agent, n_knn_agent2maptl, hidden_dim]
                    tgt_padding_mask=knn_invalid_agent2maptl,  # [n_scene, n_agent, n_knn_agent2maptl]
                    rpe=knn_rpe_agent2maptl,  # [n_scene, n_agent, n_knn_agent2maptl, d_rpe]
                    decoder_tgt=_decoder_tgt,  # [n_scene, n_agent, n_knn_agent2self, hidden_dim]
                    decoder_tgt_padding_mask=knn_invalid_agent2self,  # [n_scene, n_agent, n_knn_agent2self]
                    decoder_rpe=knn_rpe_agent2self,  # [n_scene, n_agent, n_knn_agent2self, d_rpe]
                )

        # ! all-to-all self attention
        if self.k_reinforce_all > 0:
            _emb = torch.cat([map_emb, tl_emb, agent_emb], dim=1)
            _emb_invalid = ~torch.cat([map_valid, tl_valid, agent_valid], dim=1)  # [n_scene, n_emb], bool
            n_emb = n_map + n_tl + n_agent
            _idx_all = torch.arange(n_emb)[None, :, None]  # [1, n_emb, 1]
            for mod in self.tf_reinforce_all:
                _tgt = _emb.unsqueeze(1).expand(-1, n_emb, -1, -1)
                if knn_idx_all2all is not None:
                    _tgt = _tgt[_idx_scene, _idx_all, knn_idx_all2all]
                _emb, _ = mod(
                    src=_emb,  # [n_scene, n_emb, hidden_dim]
                    src_padding_mask=_emb_invalid,  # [n_scene, n_emb]
                    tgt=_tgt,  # [n_scene, n_emb, n_knn_all2all, hidden_dim]
                    tgt_padding_mask=knn_invalid_all2all,  # [n_scene, n_emb, n_knn_all2all]
                    rpe=knn_rpe_all2all,  # [n_scene, n_emb, n_knn_all2all, d_rpe]
                )
            map_emb = _emb[:, :n_map]
            tl_emb = _emb[:, n_map : n_map + n_tl]
            agent_emb = _emb[:, -n_agent:]

        # ! decode to multiple futures
        # ! prepare multi-modal anchor_emb: [n_scene*n_pred, n_agent, hidden_dim]
        anchor_emb = agent_attr if self.use_attr_for_multi_modal else agent_emb
        # [n_scene*n_agent, n_pred, hidden_dim]
        anchor_emb = self.anchors(agent_valid.flatten(0, 1), anchor_emb.flatten(0, 1), agent_type.flatten(0, 1))
        # ! reinforce anchor_emb by attending to contexts, gather context for each agent
        if self.k_reinforce_anchor > 0:
            if self.n_latent_query > 0:  # latent query attention to reduce context dimension,
                ctx_emb = agent_attr if self.use_attr_for_multi_modal else agent_emb
                # [n_scene*n_agent, n_latent_query, hidden_dim]
                ctx_emb = self.latent_query(agent_valid.flatten(0, 1), ctx_emb.flatten(0, 1), agent_type.flatten(0, 1))
                # [n_scene, n_agent, n_emb, hidden_dim]
                _tgt = torch.cat([map_emb, tl_emb, agent_emb], dim=1).unsqueeze(1).expand(-1, n_agent, -1, -1)
                # [n_scene, n_agent, n_knn_agent2all, hidden_dim]
                if knn_idx_agent2all is not None:
                    _tgt = _tgt[_idx_scene, _idx_agent, knn_idx_agent2all]

                if self.latent_query_use_tf_decoder:
                    ctx_emb, _ = self.tf_latent_query(
                        src=ctx_emb,  # [n_scene*n_agent, n_latent_query, hidden_dim]
                        tgt=_tgt.flatten(0, 1).unsqueeze(1).expand(-1, self.n_latent_query, -1, -1),
                        tgt_padding_mask=knn_invalid_agent2all.flatten(0, 1)
                        .unsqueeze(1)
                        .expand(-1, self.n_latent_query, -1),
                        rpe=knn_rpe_agent2all.flatten(0, 1).unsqueeze(1).expand(-1, self.n_latent_query, -1, -1),
                    )
                else:
                    ctx_emb, _ = self.tf_latent_cross(
                        src=ctx_emb,  # [n_scene*n_agent, n_latent_query, hidden_dim]
                        tgt=_tgt.flatten(0, 1).unsqueeze(1).expand(-1, self.n_latent_query, -1, -1),
                        tgt_padding_mask=knn_invalid_agent2all.flatten(0, 1)
                        .unsqueeze(1)
                        .expand(-1, self.n_latent_query, -1),
                        rpe=knn_rpe_agent2all.flatten(0, 1).unsqueeze(1).expand(-1, self.n_latent_query, -1, -1),
                    )
                    ctx_emb, _ = self.tf_latent_self(src=ctx_emb, tgt=ctx_emb)

                anchor_emb, _ = self.tf_reinforce_anchor(src=anchor_emb, tgt=ctx_emb)
            else:  # no compression, ctx_emb: [n_scene*n_agent, n_knn_agent2all, hidden_dim]
                ctx_emb = torch.cat([map_emb, tl_emb, agent_emb], dim=1).unsqueeze(1).expand(-1, n_agent, -1, -1)
                # [n_scene, n_agent, n_knn_agent2all, hidden_dim]
                if knn_idx_agent2all is not None:
                    ctx_emb = ctx_emb[_idx_scene, _idx_agent, knn_idx_agent2all]
                anchor_emb, _ = self.tf_reinforce_anchor(
                    src=anchor_emb,  # [n_scene*n_agent, n_pred, hidden_dim]
                    tgt=ctx_emb.flatten(0, 1).unsqueeze(1).expand(-1, self.n_pred, -1, -1),
                    tgt_padding_mask=knn_invalid_agent2all.flatten(0, 1).unsqueeze(1).expand(-1, self.n_pred, -1),
                    rpe=knn_rpe_agent2all.flatten(0, 1).unsqueeze(1).expand(-1, self.n_pred, -1, -1),
                )
        else:  # ! no reinfoce by attending to context
            if self.anchor_self_attn:
                anchor_emb, _ = self.tf_anchor_self(src=anchor_emb, tgt=anchor_emb)

        #  generate output
        anchor_emb = anchor_emb.view(n_scene, n_agent, self.n_pred, self.hidden_dim)
        output = self.mlp_head(anchor_emb)  # [B, N, n_pred * 60]
        output = output.view(n_scene, n_agent, self.n_pred, 360)
        conf = output[..., 0]
        pred = output[..., 1:]
        return conf, pred