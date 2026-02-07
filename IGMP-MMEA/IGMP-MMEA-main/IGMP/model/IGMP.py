import types
import torch
import transformers
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
import time  # 添加time模块用于性能计时

from .Tool_model import AutomaticWeightedLoss
from .IGMP_tools import MultiModalEncoder
from .IGMP_loss import CustomMultiLossLayer, icl_loss
from src.utils import set_optim, Loss_log, pairwise_distances, csls_sim
from src.utils import pairwise_distances
import os.path as osp
import json


class IGMP(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.kgs = kgs
        self.args = args
        self.ent_wo_img = kgs["ent_wo_img"]
        self.ent_w_img = kgs["ent_w_img"]

        self.img_features1 = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        self.rel_features = torch.Tensor(kgs["rel_features"]).cuda()
        self.att_features = torch.Tensor(kgs["att_features"]).cuda()
        self.name_features = None
        self.char_features = None
        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].cuda()
            self.char_features = kgs["char_features"].cuda()

        img_dim = self._get_img_dim(kgs)

        char_dim = kgs["char_features"].shape[1] if self.char_features is not None else 100
        self.logger = args.logger if hasattr(args, 'logger') else None
        if self.logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)

        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    ent_num=kgs["ent_num"],
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head,
                                                    attr_input_dim=kgs["att_features"].shape[1])

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=11)
        self.criterion_cl = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_cl_joint = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2,
                                           replay=self.args.replay, neg_cross_kg=self.args.neg_cross_kg)

        tmp = -1 * torch.ones(self.input_idx.shape[0], dtype=torch.int64).cuda()
        self.replay_matrix = torch.stack([self.input_idx, tmp], dim=1).cuda()
        self.replay_ready = 0
        self.idx_one = torch.ones(self.args.batch_size, dtype=torch.int64).cuda()
        self.idx_double = torch.cat([self.idx_one, self.idx_one]).cuda()
        self.last_num = 1000000000000
        # self.idx_one = np.ones(self.args.batch_size, dtype=np.int64)

    def forward(self, batch):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, gat_img_emb, gat_att_emb, gat_rel_emb, gat_name_emb, gat_char_emb, joint_emb, hidden_states = self.joint_emb_generat(
            only_joint=False)

        gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid, char_emb_hid, gat_rel_emb_hid, gat_att_emb_hid, gat_img_emb_hid, gat_name_emb_hid, gat_char_emb_hid, joint_emb_hid = self.generate_hidden_emb(
            hidden_states)

        if self.args.replay:
            batch = torch.tensor(batch, dtype=torch.int64).cuda()
            all_ent_batch = torch.cat([batch[:, 0], batch[:, 1]])
            if not self.replay_ready:
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch)
            else:
                neg_l = self.replay_matrix[batch[:, 0], self.idx_one[:batch.shape[0]]]
                neg_r = self.replay_matrix[batch[:, 1], self.idx_one[:batch.shape[0]]]
                neg_l_set = set(neg_l.tolist())
                neg_r_set = set(neg_r.tolist())
                all_ent_set = set(all_ent_batch.tolist())
                neg_l_list = list(neg_l_set - all_ent_set)
                neg_r_list = list(neg_r_set - all_ent_set)
                neg_l_ipt = torch.tensor(neg_l_list, dtype=torch.int64).cuda()
                neg_r_ipt = torch.tensor(neg_r_list, dtype=torch.int64).cuda()
                loss_joi, l_neg, r_neg = self.criterion_cl_joint(joint_emb, batch, neg_l_ipt, neg_r_ipt)

            index = (
                all_ent_batch,
                self.idx_double[:batch.shape[0] * 2],
            )
            new_value = torch.cat([l_neg, r_neg]).cuda()

            self.replay_matrix = self.replay_matrix.index_put(index, new_value)
            if self.replay_ready == 0:
                num = torch.sum(self.replay_matrix < 0)
                if num == self.last_num:
                    self.replay_ready = 1
                    print("-----------------------------------------")
                    print("begin replay!")
                    print("-----------------------------------------")
                else:
                    self.last_num = num
        else:
            loss_joi = self.criterion_cl_joint(joint_emb, batch)

        in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, gat_rel_emb, gat_att_emb,
                                       gat_img_emb, gat_name_emb, gat_char_emb, batch)
        out_loss = self.inner_view_loss(gph_emb_hid, rel_emb_hid, att_emb_hid, img_emb_hid, name_emb_hid,
                                        gat_rel_emb_hid, gat_att_emb_hid, gat_img_emb_hid, gat_name_emb_hid,
                                        gat_char_emb_hid, char_emb_hid, batch)

        loss_all = loss_joi + in_loss + out_loss
        # loss_all = loss_joi + in_loss
        loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item()}
        output = {"loss_dic": loss_dic, "emb": joint_emb}
        return loss_all, output

    def generate_hidden_emb(self, hidden):

        gph_emb = F.normalize(hidden[:, 0, :].squeeze(1))
        rel_emb = F.normalize(hidden[:, 1, :].squeeze(1))
        att_emb = F.normalize(hidden[:, 2, :].squeeze(1))
        img_emb = F.normalize(hidden[:, 3, :].squeeze(1))
        # gat_rel_emb = F.normalize(hidden[:, 4, :].squeeze(1))
        # gat_att_emb = F.normalize(hidden[:, 5, :].squeeze(1))
        # gat_img_emb = F.normalize(hidden[:, 6, :].squeeze(1))
        gat_rel_emb, gat_att_emb, gat_img_emb = None, None, None
        # gat_img_emb =  None
        if hidden.shape[1] >= 99:
            name_emb = F.normalize(hidden[:, 7, :].squeeze(1))
            char_emb = F.normalize(hidden[:, 8, :].squeeze(1))

            gat_name_emb = F.normalize(hidden[:, 9, :].squeeze(1))
            gat_char_emb = F.normalize(hidden[:, 10, :].squeeze(1))
            joint_emb = torch.cat(
                [gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, gat_rel_emb, gat_att_emb, gat_img_emb,
                 gat_name_emb, gat_char_emb], dim=1)
        else:
            name_emb, char_emb = None, None
            gat_name_emb, gat_char_emb = None, None
            loss_name, loss_char = None, None
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb], dim=1)

        return gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, gat_rel_emb, gat_att_emb, gat_img_emb, gat_name_emb, gat_char_emb, joint_emb

    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, gat_rel_emb, gat_att_emb,
                        gat_img_emb, gat_name_emb, gat_char_emb, train_ill):
        # pdb.set_trace()
        loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
        loss_name = self.criterion_cl(name_emb, train_ill) if name_emb is not None else 0
        loss_char = self.criterion_cl(char_emb, train_ill) if char_emb is not None else 0

        # 计算新增的损失
        loss_gat_rel = self.criterion_cl(gat_rel_emb, train_ill) if gat_rel_emb is not None else 0
        loss_gat_att = self.criterion_cl(gat_att_emb, train_ill) if gat_att_emb is not None else 0
        loss_gat_img = self.criterion_cl(gat_img_emb, train_ill) if gat_img_emb is not None else 0
        loss_gat_name = self.criterion_cl(gat_name_emb, train_ill) if gat_name_emb is not None else 0
        loss_gat_char = self.criterion_cl(gat_char_emb, train_ill) if gat_char_emb is not None else 0

        total_loss = self.multi_loss_layer([
            loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char,
            loss_gat_rel, loss_gat_att, loss_gat_img, loss_gat_name, loss_gat_char
        ])

        return total_loss

    # --------- necessary ---------------

    def joint_emb_generat(self, only_joint=True):
        gph_emb, img_emb, rel_emb, att_emb, \
        name_emb, char_emb, gat_img_emb, gat_att_emb, gat_rel_emb, gat_name_emb, gat_char_emb, joint_emb, hidden_states, weight_norm \
            = self.multimodal_encoder(self.input_idx,
                                      self.adj,
                                      self.img_features1,
                                      self.rel_features,
                                      self.att_features,
                                      self.name_features,
                                      self.char_features)

        if only_joint:
            return joint_emb, weight_norm
        else:
            return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, gat_img_emb, gat_att_emb, gat_rel_emb, gat_name_emb, gat_char_emb, joint_emb, hidden_states

    def compute_csls_sim(self, sim_matrix, k=10):

        source_avg = torch.mean(torch.topk(sim_matrix, min(k, sim_matrix.shape[1]), dim=1)[0], dim=1)
        target_avg = torch.mean(torch.topk(sim_matrix.t(), min(k, sim_matrix.shape[0]), dim=1)[0], dim=1)
        csls_sim = 2 * sim_matrix - source_avg.unsqueeze(1) - target_avg.unsqueeze(0)

        return csls_sim

    # --------- share ---------------

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links
        distance_list = []
        for i in np.arange(0, len(left_non_train), 1000):
            d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        if (epoch + 1) % (self.args.semi_learn_step * 5) == self.args.semi_learn_step:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
        else:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if
                         (preds_r[p] == i) and ((left_non_train[i], right_non_train[p]) in new_links)]

        return new_links

    def data_refresh(self, logger, train_ill, test_ill_, left_non_train, right_non_train, new_links=[]):
        if len(new_links) != 0 and (len(left_non_train) != 0 and len(right_non_train) != 0):
            new_links_select = new_links
            train_ill = np.vstack((train_ill, np.array(new_links_select)))
            num_true = len([nl for nl in new_links_select if nl in test_ill_])
            for nl in new_links_select:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])

            logger.info(f"#new_links_select:{len(new_links_select)}")
            logger.info(f"train_ill.shape:{train_ill.shape}")
            logger.info(f"#true_links: {num_true}")
            logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
            logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links

    def update_images(self, epoch, train_ill, final_emb):

        self.logger.info(f"Epoch {epoch}: Updating image features")
        with torch.no_grad():
            if isinstance(train_ill, torch.utils.data.Dataset):
                train_pairs = train_ill.data
            else:
                train_pairs = train_ill

            left_train = [pair[0] for pair in train_pairs]
            right_train = [pair[1] for pair in train_pairs]
            left_wo_img = [e for e in left_train if e in self.ent_wo_img]
            right_wo_img = [e for e in right_train if e in self.ent_wo_img]

            distance_list = []

            # 批处理计算距离以提高效率
            batch_size = 128

            if left_wo_img and right_train:
                for i in range(0, len(left_wo_img), batch_size):
                    batch = left_wo_img[i:i + batch_size]

                    d = pairwise_distances(final_emb[batch], final_emb[right_train])

                    distance_list.append((batch, right_train, d))

            if right_wo_img and left_train:
                for i in range(0, len(right_wo_img), batch_size):
                    batch = right_wo_img[i:i + batch_size]
                    d = pairwise_distances(final_emb[batch], final_emb[left_train])
                    distance_list.append((batch, left_train, d))

            updates = []

            # 处理所有距离
            for ents_wo_img, ents_w_img_candidates, distances in distance_list:
                closest_indices = torch.argmin(distances, dim=1).cpu().numpy()

                for i, idx in enumerate(closest_indices):
                    entity_wo_img = ents_wo_img[i]
                    closest_entity = ents_w_img_candidates[idx]

                    if closest_entity in self.ent_w_img:
                        updates.append((entity_wo_img, closest_entity))


            # 应用更新
            if updates:
                img_features = self.img_features1
                updated_count = 0

                for entity_wo_img, donor_entity in updates:
                    if entity_wo_img in self.ent_wo_img and donor_entity in self.ent_w_img:
                        img_features[entity_wo_img] = img_features[donor_entity]
                        self.ent_w_img.append(entity_wo_img)
                        updated_count += 1


                if updated_count > 0:
                    self.img_features1 = F.normalize(img_features)
                    self.logger.info(f"Updated {updated_count} image features using only training entities")
                    self.logger.info(f"Remaining entities without images: {len(self.ent_wo_img)}")
                    self.logger.info(f"Entities with images: {len(self.ent_w_img)}")
                else:
                    self.logger.info("No actual updates performed")

            return

    def update_images_from_neighbors(self, epoch, train_ill, final_emb=None, sim_threshold=0.0,
                                     first_order_weight=0.5, second_order_weight=0.5):
        if not hasattr(self, 'logger') or self.logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)

        # 归一化权重
        total_weight = first_order_weight + second_order_weight
        if total_weight != 1.0:
            first_order_weight = first_order_weight / total_weight
            second_order_weight = second_order_weight / total_weight

        similarity_emb = final_emb

        try:

            if isinstance(train_ill, torch.utils.data.Dataset):
                train_pairs = train_ill.data
            else:
                train_pairs = train_ill


            left_train = [pair[0] for pair in train_pairs]
            right_train = [pair[1] for pair in train_pairs]
            train_entities = set(left_train + right_train)


            train_wo_img = [e for e in train_entities if e in self.ent_wo_img]


            # 获取图像特征
            img_features = self.img_features1.cpu()

            try:
                if hasattr(self.adj, '_indices') and hasattr(self.adj, '_values'):
                    # 如果adj已经是稀疏格式
                    indices = self.adj._indices().cpu()
                    values = self.adj._values().cpu()
                else:
                    # 尝试转换为稀疏格式
                    adj_coo = self.adj.cpu().to_sparse_coo()
                    indices = adj_coo.indices()
                    values = adj_coo.values()
            except Exception as e:
                self.logger.error(f"转换邻接矩阵失败: {str(e)}")
                try:
                    adj_numpy = self.adj.cpu().numpy()
                    # 使用NumPy找到非零元素
                    src_indices, tgt_indices = np.nonzero(adj_numpy)
                    indices = torch.from_numpy(np.vstack((src_indices, tgt_indices)))
                    values = torch.ones(indices.shape[1])
                    self.logger.info(f"使用NumPy处理邻接矩阵，共有 {indices.shape[1]} 条边")
                except Exception as e2:
                    self.logger.error(f"NumPy处理也失败: {str(e2)}")
                    return 0
            start_time = time.time()
            entity_neighbors = {}
            processed_edges = 0

            # 遍历所有边，构建邻居关系
            for i in range(indices.shape[1]):
                src, tgt = indices[0, i].item(), indices[1, i].item()
                processed_edges += 1
                if src not in train_entities or tgt not in train_entities:
                    continue
                if src not in entity_neighbors:
                    entity_neighbors[src] = []
                entity_neighbors[src].append(tgt)

                if src != tgt:
                    if tgt not in entity_neighbors:
                        entity_neighbors[tgt] = []
                    entity_neighbors[tgt].append(src)


            updates = []

            batch_size = 1000
            total_batches = (len(train_wo_img) + batch_size - 1) // batch_size
            start_time = time.time()

            # 批处理更新收集过程
            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(train_wo_img))
                batch_entities = train_wo_img[batch_start:batch_end]

                batch_updates = []

                for entity in batch_entities:
                    first_order = entity_neighbors.get(entity, [])

                    if not first_order:
                        continue

                    first_order = list(set(first_order))

                    if entity in first_order:
                        first_order.remove(entity)

                    if not first_order:
                        continue

                    first_order = [n for n in first_order if n in train_entities]

                    if not first_order:
                        continue

                    first_order_with_img = [n for n in first_order if n in self.ent_w_img]

                    second_order = []
                    for neighbor in first_order:

                        neighbors_of_neighbor = entity_neighbors.get(neighbor, [])
                        second_order.extend(neighbors_of_neighbor)

                    second_order = list(set(second_order))

                    second_order = [n for n in second_order if n not in first_order and n != entity]

                    second_order = [n for n in second_order if n in train_entities]

                    # 过滤掉没有图像的二阶邻居
                    second_order_with_img = [n for n in second_order if n in self.ent_w_img]

                    if not first_order_with_img and not second_order_with_img:
                        continue

                    if similarity_emb is not None:
                        first_order_weights = None
                        if first_order_with_img:
                            entity_emb = similarity_emb[entity].cpu()
                            neighbor_embs = similarity_emb[first_order_with_img].cpu()

                            sim_scores = torch.mm(entity_emb.unsqueeze(0), neighbor_embs.t()).squeeze()

                            # 处理sim_scores为标量的情况
                            if sim_scores.dim() == 0:
                                sim_scores = sim_scores.unsqueeze(0)

                            # 过滤掉相似度低于阈值的邻居
                            valid_mask = sim_scores > sim_threshold
                            valid_indices = torch.nonzero(valid_mask).squeeze(1)


                            if valid_indices.numel() > 0:
                                valid_indices = valid_indices.tolist()
                                if not isinstance(valid_indices, list):
                                    valid_indices = [valid_indices]

                                valid_first_order = [first_order_with_img[i] for i in valid_indices]
                                valid_scores = sim_scores[valid_indices]

                                first_order_weights = torch.softmax(valid_scores, dim=0)
                                first_order_with_img = valid_first_order

                        second_order_weights = None
                        if second_order_with_img:
                            # 计算实体与其二阶邻居的相似度
                            entity_emb = similarity_emb[entity].cpu()
                            neighbor_embs = similarity_emb[second_order_with_img].cpu()

                            # 计算余弦相似度
                            sim_scores = torch.mm(entity_emb.unsqueeze(0), neighbor_embs.t()).squeeze()

                            if sim_scores.dim() == 0:
                                sim_scores = sim_scores.unsqueeze(0)

                            # 过滤掉相似度低于阈值的邻居
                            valid_mask = sim_scores > sim_threshold
                            valid_indices = torch.nonzero(valid_mask).squeeze(1)

                            if valid_indices.numel() > 0:
                                valid_indices = valid_indices.tolist()
                                if not isinstance(valid_indices, list):
                                    valid_indices = [valid_indices]


                                valid_second_order = [second_order_with_img[i] for i in valid_indices]
                                valid_scores = sim_scores[valid_indices]

                                # 应用softmax得到权重
                                second_order_weights = torch.softmax(valid_scores, dim=0)
                                second_order_with_img = valid_second_order

                        # 如果有有效的一阶或二阶邻居，添加到更新列表中
                        if (first_order_with_img and first_order_weights is not None) or \
                                (second_order_with_img and second_order_weights is not None):
                            batch_updates.append((
                                entity,
                                first_order_with_img if first_order_with_img else [],
                                first_order_weights if first_order_weights is not None else None,
                                second_order_with_img if second_order_with_img else [],
                                second_order_weights if second_order_weights is not None else None
                            ))
                    else:
                        if first_order_with_img or second_order_with_img:
                            first_order_weights = torch.ones(len(first_order_with_img)) / len(
                                first_order_with_img) if first_order_with_img else None
                            second_order_weights = torch.ones(len(second_order_with_img)) / len(
                                second_order_with_img) if second_order_with_img else None

                            batch_updates.append((
                                entity,
                                first_order_with_img,
                                first_order_weights,
                                second_order_with_img,
                                second_order_weights
                            ))

                updates.extend(batch_updates)

                if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
                    progress = (batch_idx + 1) / total_batches * 100
                    elapsed = time.time() - start_time
                    entities_per_sec = (batch_idx + 1) * batch_size / elapsed if elapsed > 0 else 0

            # 应用更新
            updated_count = 0
            first_time_count = 0

            with torch.no_grad():
                for update_data in updates:
                    entity = update_data[0]
                    first_order_neighbors = update_data[1]
                    first_order_weights = update_data[2]
                    second_order_neighbors = update_data[3]
                    second_order_weights = update_data[4]

                    new_feature = None


                    if first_order_neighbors and first_order_weights is not None:
                        # 获取一阶邻居的图像特征
                        first_order_features = img_features[first_order_neighbors]

                        # 确保weights是一个张量
                        first_weights_tensor = first_order_weights if isinstance(first_order_weights,
                                                                                 torch.Tensor) else torch.tensor(
                            first_order_weights)

                        # 计算一阶邻居的加权平均特征
                        first_order_feature = torch.sum(first_order_features * first_weights_tensor.unsqueeze(1), dim=0)
                        first_order_feature = F.normalize(first_order_feature, dim=0)

                        new_feature = first_order_feature * first_order_weight

                    # 处理二阶邻居
                    if second_order_neighbors and second_order_weights is not None:

                        second_order_features = img_features[second_order_neighbors]


                        second_weights_tensor = second_order_weights if isinstance(second_order_weights,
                                                                                   torch.Tensor) else torch.tensor(
                            second_order_weights)

                        # 计算二阶邻居的加权平均特征
                        second_order_feature = torch.sum(second_order_features * second_weights_tensor.unsqueeze(1),
                                                         dim=0)
                        second_order_feature = F.normalize(second_order_feature, dim=0)

                        if new_feature is None:
                            new_feature = second_order_feature * second_order_weight
                        else:
                            new_feature += second_order_feature * second_order_weight


                    if new_feature is not None:

                        new_feature = F.normalize(new_feature, dim=0)

                        # 更新特征
                        img_features[entity] = new_feature
                        updated_count += 1

                        # 如果实体是第一次被更新，记录一下
                        if entity not in self.ent_w_img:
                            self.ent_w_img.append(entity)
                            first_time_count += 1

            # 更新图像特征
            if updated_count > 0:
                self.img_features1 = F.normalize(img_features).cuda()
                if first_time_count > 0:
                    self.logger.info(f"First-time updates: {first_time_count}")
            else:
                self.logger.info("No actual updates performed")

            return updated_count
        except Exception as e:
            self.logger.error(f"更新图像特征时发生错误: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0


    def Iter_new_image_features(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):

        train_data = None

        if 'train_ill' in self.kgs:
            train_data = self.kgs['train_ill']
        else:
            self.logger.warning("训练集数据无法获取，无法执行图像更新")
            return new_links

        self.logger.info("执行图像特征更新")

        start_time = time.time()

        neighbor_update_count = self.update_images_from_neighbors(
            epoch=epoch,
            train_ill=train_data,
            final_emb=final_emb,
            sim_threshold=0.3
        )

        neighbor_time = time.time() - start_time
        self.logger.info(f"邻居更新完成，用时 {neighbor_time:.2f}秒，更新了 {neighbor_update_count} 个实体")

        align_start_time = time.time()

        # 使用基于对齐的方法更新图像特征
        self.update_images(
            epoch=epoch,
            train_ill=train_data,
            final_emb=final_emb
        )

        align_time = time.time() - align_start_time
        self.logger.info(f"对齐更新完成，用时 {align_time:.2f}秒")

        total_time = time.time() - start_time
        self.logger.info(f"图像特征更新完成，总用时 {total_time:.2f}秒")

        # 保持接口不变，返回原始new_links
        return new_links