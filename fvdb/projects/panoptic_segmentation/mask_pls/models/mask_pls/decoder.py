# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Rodrigo Marcuzzi from https://github.com/facebookresearch/Mask2Former
from typing import Dict, Tuple

import torch

from .blocks import MLP, CrossAttentionLayer, FFNLayer, SelfAttentionLayer
from .positional_encoder import PositionalEncoder


class MaskedTransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        dataset_extent: Tuple[float, float, float],
        backbone_channels: Tuple[int, ...],
        num_queries: int = 100,
    ):
        super().__init__()
        self.backbone_channels = backbone_channels
        hidden_dim = 256
        self.num_queries = num_queries
        self.num_feature_levels = 1

        self.pe_layer = PositionalEncoder(hidden_dim, dataset_extent)

        self.num_layers = 1  # feature levels
        self.decoder_blocks = 6
        self.num_layers *= self.decoder_blocks

        self.num_heads = 8

        self.feature_level_encoder = False

        self.transformer_self_attention_layers = torch.nn.ModuleList()
        self.transformer_cross_attention_layers = torch.nn.ModuleList()
        self.transformer_ffn_layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=self.num_heads, dropout=0.0)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=self.num_heads, dropout=0.0)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=1024,
                    dropout=0.0,
                )
            )

        self.decoder_norm = torch.nn.LayerNorm(hidden_dim)
        self.query_feat = torch.nn.Embedding(self.num_queries, hidden_dim)
        self.query_embed = torch.nn.Embedding(self.num_queries, hidden_dim)
        if self.feature_level_encoder:
            self.level_embed = torch.nn.Embedding(self.num_feature_levels, hidden_dim)

        self.mask_feat_proj = torch.nn.Sequential()
        in_channels = self.backbone_channels  # backbone channels
        if in_channels[-1] != hidden_dim:
            self.mask_feat_proj = torch.nn.Linear(in_channels[-1], hidden_dim)

        in_channels = in_channels[:-1][-self.num_feature_levels :]

        self.input_proj = torch.nn.ModuleList()
        for ch in in_channels:
            if ch != hidden_dim:  # linear projection to hidden_dim
                self.input_proj.append(torch.nn.Linear(ch, hidden_dim))
            else:
                self.input_proj.append(torch.nn.Sequential())

        # output FFNs

        self.class_embed = torch.nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, [hidden_dim, hidden_dim], hidden_dim)

    def forward(self, feats, coors, pad_masks) -> Tuple[Dict, torch.Tensor]:
        last_coors = coors.pop()
        last_feat = feats.pop()

        mask_features = self.mask_feat_proj(last_feat) + self.pe_layer(last_coors)
        last_pad = pad_masks.pop()
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(feats[i].shape[1])
            pos.append(self.pe_layer(coors[i]))

            feat = self.input_proj[i](feats[i])

            src.append(feat)

        bs = src[0].shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        output = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)

        predictions_class = []
        predictions_class_sem = []
        predictions_mask = []
        predictions_sem_embed = []

        # predictions on learnable query features, first attn_mask
        pred_result = None
        pred_result = self.pred_heads(
            output,
            mask_features,
            pad_mask=last_pad,
        )

        predictions_class.append(pred_result["outputs_class"])
        predictions_class_sem.append(pred_result["outputs_class_sem"])
        predictions_mask.append(pred_result["outputs_mask"])
        predictions_sem_embed.append(pred_result["outputs_sem_embed"])

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            attn_mask = None
            if pred_result is not None:
                attn_mask = pred_result["attn_mask"]
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                attn_mask=attn_mask,
                padding_mask=pad_masks[level_index],
                pos=pos[level_index],
                query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[i](
                output, attn_mask=None, padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            # get predictions and attn mask for next feature level
            pred_result = self.pred_heads(
                output,
                mask_features,
                pad_mask=last_pad,
            )

            predictions_class.append(pred_result["outputs_class"])
            predictions_class_sem.append(pred_result["outputs_class_sem"])
            predictions_mask.append(pred_result["outputs_mask"])
            predictions_sem_embed.append(pred_result["outputs_sem_embed"])

        assert len(predictions_mask) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_logits_sem": predictions_class_sem[-1],
            "pred_masks": predictions_mask[-1],
            "pred_sem_embed": predictions_sem_embed[-1],
            "query_embeddings": output,
        }

        return out, last_pad

    def pred_heads(
        self,
        output,
        mask_features,
        pad_mask=None,
    ):
        decoder_output = self.decoder_norm(output)
        mask_embed = self.mask_embed(decoder_output)

        # The mask predictions outputs_mask are computed as a combination of mask_embed and mask_features
        # using the einsum function from PyTorch.
        # This function performs a batch-wise matrix multiplication between mask_embed and mask_features
        # and outputs a tensor of shape (batch_size, num_points, num_queries).
        # The result is a tensor that represents the mask prediction for each query and point in the batch.
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)

        attn_mask = (outputs_mask.sigmoid() < 0.5).detach().bool()
        attn_mask[pad_mask] = True
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1).permute(0, 2, 1)

        # The *outputs_class* tensor holds the "semantic information,"
        # essentially a distribution over possible classes for each of the queries.
        # Each slice along the num_queries dimension can be seen as the model's prediction of the class
        # of the object that each query represents.
        # Dim: (num_batch, num_classes, num_query)

        # The *outputs_mask* tensor provides spatial information, indicating which points
        # from the input point cloud are associated with each query.
        # Each slice along the num_queries dimension in this tensor can be seen as a mask over
        # the point cloud, highlighting the points that the model
        # believes belong to the object represented by the corresponding query.
        # Dim: (num_batch, num_points, num_query)

        result = {
            "outputs_mask": outputs_mask,
            "attn_mask": attn_mask,
        }

        result["outputs_class"] = self.class_embed(decoder_output)

        result["outputs_class_sem"] = None  # type: ignore

        result["outputs_sem_embed"] = None  # type: ignore

        return result
