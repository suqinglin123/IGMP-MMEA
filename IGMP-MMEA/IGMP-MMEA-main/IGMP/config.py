import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict
import argparse


class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        # change
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', '..', 'data', ''))

    def get_args(self):
        parser = argparse.ArgumentParser()
        # base
        parser.add_argument('--gpu', default=0, type=int)
        parser.add_argument('--batch_size', default=3500, type=int)
        parser.add_argument('--epoch', default=250, type=int)
        parser.add_argument("--save_model", default=0, type=int, choices=[0, 1])
        parser.add_argument("--only_test", default=0, type=int, choices=[0, 1])

        # torthlight
        parser.add_argument("--no_tensorboard", default=False, action="store_true")
        parser.add_argument("--exp_name", default="EA_exp", type=str, help="Experiment name")
        parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        parser.add_argument("--exp_id", default="001", type=str, help="Experiment ID")
        parser.add_argument("--random_seed", default=42, type=int)
        parser.add_argument("--data_path", default="mmkg", type=str, help="Experiment path")

        # --------- EA -----------
        parser.add_argument("--data_choice", default="OEA_EN_FR_15K_V1", type=str, choices=["DBP15K", "DWY", "FBYG15K", "FBDB15K","OEA_D_W_15K_V1","OEA_EN_FR_15K_V1","OEA_EN_DE_15K_V1","OEA_D_W_15K_V2"], help="Experiment path")
        parser.add_argument("--data_rate", type=float, default=0.2, help="training set rate")


        # TODO: add some dynamic variable
        parser.add_argument("--model_name", default="IGMP", type=str, choices=["EVA", "MCLEA", "MSNEA", "IGMP"], help="model name")
        parser.add_argument("--model_name_save", default="", type=str, help="model name for model load")

        parser.add_argument('--workers', type=int, default=12)
        parser.add_argument('--accumulation_steps', type=int, default=1)
        parser.add_argument("--scheduler", default="cos", type=str, choices=["linear", "cos", "fixed"])
        parser.add_argument("--optim", default="adamw", type=str, choices=["adamw", "adam"])
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument('--eval_epoch', default=1, type=int, help='evaluate each n epoch')
        parser.add_argument("--enable_sota", action="store_true", default=True)

        parser.add_argument('--margin', default=1, type=float, help='The fixed margin in loss function. ')
        parser.add_argument('--emb_dim', default=1000, type=int, help='The embedding dimension in KGE model.')
        parser.add_argument('--adv_temp', default=1.0, type=float, help='The temperature of sampling in self-adversarial negative sampling.')
        parser.add_argument("--contrastive_loss", default=0, type=int, choices=[0, 1])
        parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        parser.add_argument("--ratio", type=str, default="0.4", help="which image adapt",
                            choices=["0.05", "0.1", "0.15", "0.2", "0.3", "0.4",
                                     "0.45", "0.5", "0.55", "0.6", "0.7", "0.75", "0.8", "0.9", "0.95", "1.0"])

        # --------- EVA -----------
        parser.add_argument("--data_split", default="norm", type=str, help="Experiment split", choices=["dbp_wd_15k_V2", "dbp_wd_15k_V1", "zh_en", "ja_en", "fr_en", "norm","dense"])
        parser.add_argument("--hidden_units", type=str, default="300,300,300", help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
        parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
        parser.add_argument("--distance", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')", choices=[1, 2])
        parser.add_argument("--csls", action="store_true", default=True, help="use CSLS for inference")
        parser.add_argument("--csls_k", type=int, default=3, help="top k for csls")
        parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
        parser.add_argument("--semi_learn_step", type=int, default=10, help="If IL, what's the update step?")
        parser.add_argument("--il_start", type=int, default=0, help="If Il, when to start?")
        parser.add_argument("--unsup", action="store_true", default=False)
        parser.add_argument("--unsup_k", type=int, default=1000, help="|visual seed|")
        parser.add_argument("--update", action="store_true", default=True, help="update image")
        parser.add_argument("--update_start", type=int, default=1, help="If update, when to start")


        # --------- MCLEA -----------
        parser.add_argument("--unsup_mode", type=str, default="img", help="unsup mode", choices=["img", "name", "char"])
        parser.add_argument("--tau", type=float, default=0.1, help="the temperature factor of contrastive loss")
        parser.add_argument("--alpha", type=float, default=0.2, help="the margin of InfoMaxNCE loss")
        parser.add_argument("--with_weight", type=int, default=1, help="Whether to weight the fusion of different ")
        parser.add_argument("--structure_encoder", type=str, default="gat", help="the encoder of structure view", choices=["gat", "gcn"])
        parser.add_argument("--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss")

        parser.add_argument("--projection", action="store_true", default=False, help="add projection for model")
        parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
        parser.add_argument("--instance_normalization", action="store_true", default=False, help="enable instance normalization")
        parser.add_argument("--attr_dim", type=int, default=300, help="the hidden size of attr and rel features")
        parser.add_argument("--img_dim", type=int, default=300, help="the hidden size of img feature")
        parser.add_argument("--name_dim", type=int, default=300, help="the hidden size of name feature")
        parser.add_argument("--char_dim", type=int, default=300, help="the hidden size of char feature")

        parser.add_argument("--w_gcn", action="store_false", default=True, help="with gcn features")
        parser.add_argument("--w_rel", action="store_false", default=True, help="with rel features")
        parser.add_argument("--w_attr", action="store_false", default=True, help="with attr features")
        parser.add_argument("--w_name", action="store_false", default=False, help="with name features")
        parser.add_argument("--w_char", action="store_false", default=False, help="with char features")
        parser.add_argument("--w_img", action="store_false", default=True, help="with img features")
        parser.add_argument("--use_surface", type=int, default=0, help="whether to use the surface")

        # --------- 模态权重调整参数 -----------
        parser.add_argument("--use_modal_weights", type=int, default=1, help="是否使用自定义模态权重")
        parser.add_argument("--img_weight", type=float, default=0.4, help="图像模态的权重")
        parser.add_argument("--attr_weight", type=float, default=1.2, help="属性模态的权重")
        parser.add_argument("--rel_weight", type=float, default=1.0, help="关系模态的权重")
        parser.add_argument("--gcn_weight", type=float, default=1.2, help="图结构模态的权重")
        parser.add_argument("--name_weight", type=float, default=0, help="名称模态的权重")
        parser.add_argument("--char_weight", type=float, default=0, help="字符模态的权重")
        parser.add_argument("--gat_img_weight", type=float, default=0, help="GAT图像模态的权重")
        parser.add_argument("--gat_attr"
                            "_weight", type=float, default=1.2, help="GAT属性模态的权重")
        parser.add_argument("--gat_rel_weight", type=float, default=0, help="GAT关系模态的权重")
        parser.add_argument("--gat_name_weight", type=float, default=0, help="GAT名称模态的权重")
        parser.add_argument("--gat_char_weight", type=float, default=0, help="GAT字符模态的权重")
        parser.add_argument("--attention_temp", type=float, default=1.0, help="注意力温度参数，控制注意力分布的锐度")

        parser.add_argument("--inner_view_num", type=int, default=6, help="the number of inner view")
        parser.add_argument("--word_embedding", type=str, default="bert", help="the type of word embedding, [glove|fasttext]", choices=["glove", "bert"])
        # projection head
        parser.add_argument("--use_project_head", action="store_true", default=False, help="use projection head")
        parser.add_argument("--zoom", type=float, default=0.1, help="narrow the range of losses")
        parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]", choices=["sum", "mean"])

        # --------- MEAformer -----------
        parser.add_argument("--hidden_size", type=int, default=300, help="the hidden size of MEAformer")
        parser.add_argument("--intermediate_size", type=int, default=400, help="the hidden size of MEAformer")
        parser.add_argument("--num_attention_heads", type=int, default=1, help="the number of attention_heads of MEAformer")
        parser.add_argument("--num_hidden_layers", type=int, default=1, help="the number of hidden_layers of MEAformer")
        parser.add_argument("--position_embedding_type", default="absolute", type=str)
        parser.add_argument("--use_intermediate", type=int, default=1, help="whether to use_intermediate")
        parser.add_argument("--replay", type=int, default=0, help="whether to use replay strategy")
        parser.add_argument("--neg_cross_kg", type=int, default=0, help="whether to force the negative samples in the opposite KG")

        # --------- MSNEA -----------
        parser.add_argument("--dim", type=int, default=100, help="the hidden size of MSNEA")
        parser.add_argument("--neg_triple_num", type=int, default=1, help="neg triple num")
        parser.add_argument("--use_bert", type=int, default=0)
        parser.add_argument("--use_attr_value", type=int, default=0)


        # ------------ Para ------------
        parser.add_argument('--rank', type=int, default=0, help='rank to dist')
        parser.add_argument('--dist', type=int, default=0, help='whether to dist')
        parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--world-size', default=3, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
        parser.add_argument("--local_rank", default=-1, type=int)

        self.cfg = parser.parse_args()

    def update_train_configs(self):
        assert not (self.cfg.save_model and self.cfg.only_test)

        self.cfg.data_root = self.data_root

        if self.cfg.use_surface:
            self.cfg.w_name = True
            self.cfg.w_char = True
        else:
            self.cfg.w_name = False
            self.cfg.w_char = False

        if self.cfg.data_choice in ["FBYG15K", "FBDB15K"]:
            self.cfg.use_intermediate = 0
            self.cfg.data_split = "norm"
            self.cfg.inner_view_num = 4
            # assert self.cfg.data_rate in [0.2, 0.5, 0.8]
            self.cfg.w_name = False
            self.cfg.w_char = False
            self.cfg.use_surface = 0
            data_split_name = f"{self.cfg.data_rate}_"
        else:
            data_split_name = f"{self.cfg.data_split}_"
            if self.cfg.w_name and self.cfg.w_char:
                data_split_name = f"{data_split_name}with_surface_"

        self.cfg.exp_id = f"{self.cfg.model_name}_{self.cfg.data_choice}_{data_split_name}{self.cfg.exp_id}"
        self.cfg.data_path = osp.join(self.data_root, self.cfg.data_path)
        self.cfg.dump_path = osp.join(self.cfg.data_path, self.cfg.dump_path)
        if self.cfg.only_test == 1:
            self.save_model = 0
            self.dist = 0

        # --------- MSNEA -----------
        self.cfg.dim = self.cfg.attr_dim

        # --------- MEAformer -----------
        self.cfg.max_position_embeddings = self.cfg.inner_view_num + 1
        assert self.cfg.hidden_size == self.cfg.attr_dim


        return self.cfg
