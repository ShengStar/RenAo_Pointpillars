import torch
import torch.nn as nn
import torch.nn.functional as F


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        # # topk_6144
        # self.topk_6144 = nn.Linear(64, 64, bias=True)
        # self.nm_6144 = nn.BatchNorm1d(64)
        # self.rl_6144 = nn.ReLU()
        # self.topk_score_6144 = nn.Linear(64, 1, bias=True)
        # self.nm_score_6144 = nn.BatchNorm1d(1)
        # self.rl_score_6144 = nn.Sigmoid()
        # # self.rl_score_6144 = nn.Sigmoid()
        # # topk_6144

        # topk_4096
        self.topk_4096 = nn.Linear(64, 64, bias=True)
        self.nm_4096 = nn.BatchNorm1d(64)
        self.rl_4096 = nn.Sigmoid()
        self.topk_score_4096 = nn.Linear(64, 1, bias=True)
        self.nm_score_4096 = nn.BatchNorm1d(1)
        # self.rl_score_4096 = nn.Sigmoid()
        # topk_4096

        # # topk_2048
        # self.topk_2048 = nn.Linear(64, 64, bias=True)
        # self.nm_2048 = nn.BatchNorm1d(64)
        # self.rl_2048 = nn.ReLU()
        # self.topk_score_2048 = nn.Linear(64, 1, bias=True)
        # self.nm_score_2048 = nn.BatchNorm1d(1)
        # self.rl_score_2048 = nn.Sigmoid()
        # # topk_2018



    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        reduce_pre = []
        reduce_cls = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :] # 坐标
            # flag_mask = this_coords[:,4] != -1
            # pillars_cls = this_coords[:,4]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] # torch.Size([8032])
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]

            # if pillars.shape[0] > 6144:
            #     pillars = self.topk_6144(pillars)
            #     pillars = self.nm_6144(pillars)
            #     pillars = self.rl_6144(pillars)
            #     score = self.topk_score_6144(pillars)
            #     score = self.nm_score_6144(score)
            #     score = self.rl_score_6144(score).squeeze()
            #     # print(score.shape[0])
            #     top_score,index= torch.topk(score,6144,largest=True)
            #     indices = indices[index]
            #     pillars = pillars[index]

            if pillars.shape[0] > 4096:
                pillars = self.topk_4096(pillars)
                pillars = self.nm_4096(pillars)
                pillars = self.rl_4096(pillars)
                score = self.topk_score_4096(pillars)
                score = self.nm_score_4096(score)
                # score = self.rl_score_4096(score).squeeze()
                score = F.gumbel_softmax(score, hard=True).squeeze()
                top_score,index= torch.topk(score,4096,largest=True)
                top_score1 = top_score
                index1 = index
                indices = indices[index]
                pillars = pillars[index]

                # mask = score == score
                # mask[index1] = False
                # score = score[mask]
                # # print(mask.shape)
                # # print(top_score1.shape)
                # # top_score1 = top_score1[mask]
                # print(torch.mean(score))



            # if pillars.shape[0] > 2048:

            #     pillars = self.topk_2048(pillars)
            #     pillars = self.nm_2048(pillars)
            #     pillars = self.rl_2048(pillars)
            #     score = self.topk_score_2048(pillars)
            #     score = self.nm_score_2048(score)
            #     score = self.rl_score_2048(score).squeeze()
            #     top_score,index= torch.topk(score,2048,largest=True)
            #     indices = indices[index]
            #     pillars = pillars[index]



            # mask = indices == indices
            # mask[index] = False
            # indices = indices[mask]
            pillars = pillars.t()
            # pillars = pillars[:,mask]
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
            # pillars_cls = pillars_cls[index]
            # reduce_pre.append(top_score)
            # reduce_cls.append(pillars_cls)

            # print(mask)

            # indices[index] = 
            # print(indices.shape)
            # pillars = pillars.t()
            # indices = indices[flag_mask]
            # pillars = pillars[:,flag_mask] # 可能出错
            # spatial_feature[:, indices] = pillars
            # batch_spatial_features.append(spatial_feature)
        
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # batch_reduce_pre = torch.stack(reduce_pre, 0)
        # batch_reduce_cls = torch.stack(reduce_cls, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        # batch_dict.update({'batch_reduce_pre':batch_reduce_pre})
        # batch_dict.update({'batch_reduce_cls':batch_reduce_cls})


        return batch_dict