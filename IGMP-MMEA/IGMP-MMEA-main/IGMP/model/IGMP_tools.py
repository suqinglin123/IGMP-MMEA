# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward

from .layers import ProjectionHead
from .Tool_model import GAT, GCN
import pdb


class MformerFusion(nn.Module):
    def __init__(self, args, modal_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])
        self.type_id = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).cuda()  # 更新类型 ID 张量
        
        # 是否使用自定义模态权重
        self.use_modal_weights = hasattr(args, 'use_modal_weights') and args.use_modal_weights == 1
        
        # 模态类型对应的权重
        if self.use_modal_weights:
            # 模态索引: 0=img, 1=attr, 2=rel, 3=gcn, 4=name, 5=char, 6=gat_img, 7=gat_attr, 8=gat_rel, 9=gat_name, 10=gat_char
            self.modal_weights = torch.tensor([
                args.img_weight,      # 图像模态权重
                args.attr_weight,     # 属性模态权重
                args.rel_weight,      # 关系模态权重
                args.gcn_weight,      # 图结构模态权重
                args.name_weight,     # 名称模态权重
                args.char_weight,     # 字符模态权重
                args.gat_img_weight,  # GAT图像模态权重
                args.gat_attr_weight, # GAT属性模态权重
                args.gat_rel_weight,  # GAT关系模态权重
                args.gat_name_weight, # GAT名称模态权重
                args.gat_char_weight  # GAT字符模态权重
            ]).cuda()
            self.attention_temp = args.attention_temp  # 注意力温度参数

    def forward(self, embs):
        # 过滤掉 None 值，仅保留非空嵌入
        valid_embs = []
        valid_indices = []
        for idx in range(len(embs)):
            if embs[idx] is not None:
                valid_embs.append(embs[idx])
                valid_indices.append(idx)
        
        # 计算有效模态数量
        modal_num = len(valid_embs)

        # 将非空嵌入堆叠到一起，形成一个新的张量，维度为 [batch_size, modal_num, hidden_size]
        hidden_states = torch.stack(valid_embs, dim=1)
        bs = hidden_states.shape[0]

        # 遍历每一层 BertLayer，对 hidden_states 进行处理
        for i, layer_module in enumerate(self.fusion_layer):
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]

        # 计算注意力权重
        attention_pro = torch.sum(layer_outputs[1], dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(modal_num * self.args.num_attention_heads)
        
        # 应用自定义模态权重
        if self.use_modal_weights:
            # 获取有效模态的权重
            valid_weights = torch.stack([self.modal_weights[idx] for idx in valid_indices])

            # 调整注意力分数
            attention_pro_comb = attention_pro_comb * valid_weights.unsqueeze(0)

            # 应用温度参数调整注意力分布的锐度
            attention_pro_comb = attention_pro_comb / self.attention_temp
        
        # 归一化注意力权重
        weight_norm = F.softmax(attention_pro_comb, dim=-1)

        # 对每个模态嵌入进行加权并归一化
        weighted_embs = [weight_norm[:, idx].unsqueeze(1) * F.normalize(valid_embs[idx]) for idx in range(modal_num)]
        # 将加权后的嵌入拼接起来，形成联合嵌入
        joint_emb = torch.cat(weighted_embs, dim=1)

        return joint_emb, hidden_states, weight_norm


class MultiModalEncoder(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units

    """

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False,
                 attr_input_dim=1000):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        #########################
        # Entity Embedding
        #########################
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        #########################
        # Modal Encoder
        #########################

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(768, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)
        # self.graph_fc = nn.Linear(self.input_dim, char_dim)

        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)

        # 定义 GAT 层
        self.gat_rel = GAT(n_units=[attr_dim, attr_dim], n_heads=[1, 1], dropout=args.dropout,
                           attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization,
                           diag=True)
        self.gat_att = GAT(n_units=[attr_dim, attr_dim], n_heads=[1, 1], dropout=args.dropout,
                           attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization,
                           diag=True)
        self.gat_img = GAT(n_units=[img_dim, img_dim], n_heads=[1, 1], dropout=args.dropout,
                           attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization,
                           diag=True)
        self.gat_name = GAT(n_units=[char_dim, char_dim], n_heads=[1, 1], dropout=args.dropout,
                            attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization,
                            diag=True)
        self.gat_char = GAT(n_units=[char_dim, char_dim], n_heads=[1, 1], dropout=args.dropout,
                            attn_dropout=args.attn_dropout, instance_normalization=args.instance_normalization,
                            diag=True)

        #########################
        # Fusion Encoder
        #########################
        self.fusion = MformerFusion(args, modal_num=self.args.inner_view_num,
                                    with_weight=self.args.with_weight)

    def forward(self,
                input_idx,
                adj,
                img_features=None,
                rel_features=None,
                att_features=None,
                name_features=None,
                char_features=None):

        # 是否使用自定义模态权重
        use_modal_weights = hasattr(self.args, 'use_modal_weights') and self.args.use_modal_weights == 1

        if self.args.w_gcn:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
            # 应用图结构权重
            if use_modal_weights:
                gph_emb = gph_emb * self.args.gcn_weight
        else:
            gph_emb = None
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
            # img_emb = None
            # 应用图像权重
            if use_modal_weights:
                img_emb = img_emb * self.args.img_weight
                # img_emb = None
            # 使用GAT增强图像特征
            if use_modal_weights and self.args.gat_img_weight > 0:
                gat_img_emb = None
            else:
                gat_img_emb = None
        else:
            img_emb = None
            gat_img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
            # 应用关系权重
            if use_modal_weights:
                rel_emb = rel_emb * self.args.rel_weight
            
            # 使用GAT增强关系特征
            if use_modal_weights and self.args.gat_rel_weight > 0:
                gat_rel_emb = None
            else:
                gat_rel_emb = None
        else:
            rel_emb = None
            gat_rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
            # gat_att_emb = self.gat_att(att_emb, adj)
            # 应用属性权重
            if use_modal_weights:
                att_emb = att_emb * self.args.attr_weight
            
            # 使用GAT增强属性特征
            if use_modal_weights and self.args.gat_attr_weight > 0:
                gat_att_emb = self.gat_att(att_emb, adj) * self.args.gat_attr_weight
            else:
                gat_att_emb = None
        else:
            att_emb = None
            gat_att_emb = None
        if self.args.w_name and name_features is not None:
            name_emb = self.name_fc(name_features)
            # 应用名称权重
            if use_modal_weights:
                name_emb = name_emb * self.args.name_weight
            
            # 使用GAT增强名称特征
            if use_modal_weights and self.args.gat_name_weight > 0:
                gat_name_emb = None
            else:
                gat_name_emb = None
        else:
            name_emb = None
            gat_name_emb = None
        if self.args.w_char and char_features is not None:
            char_emb = self.char_fc(char_features)
            # 应用字符权重
            if use_modal_weights:
                char_emb = char_emb * self.args.char_weight
            
            # 使用GAT增强字符特征
            if use_modal_weights and self.args.gat_char_weight > 0:
                gat_char_emb = None
            else:
                gat_char_emb = None
        else:
            char_emb = None
            gat_char_emb = None

        joint_emb, hidden_states, weight_norm = self.fusion(
            [img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb, gat_img_emb, gat_att_emb, gat_rel_emb,
             gat_name_emb, gat_char_emb])
        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, gat_img_emb, gat_att_emb, gat_rel_emb, gat_name_emb, gat_char_emb, joint_emb, hidden_states, weight_norm


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)
        
        # 是否使用自定义模态权重
        self.use_modal_weights = hasattr(config, 'use_modal_weights') and config.use_modal_weights == 1

        if self.use_modal_weights:
            self.modal_preference = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.attention_temp = config.attention_temp

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [8, 8, 3, 256]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        # return x
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)


        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        if self.config.use_intermediate:
            self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states,
            output_attentions=output_attentions,
        )
        if not self.config.use_intermediate:
            return (self_attention_outputs[0], self_attention_outputs[1])

        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output, outputs)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        # return attention_output


