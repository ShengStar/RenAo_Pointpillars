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

        # topk_4096
        self.topk_4096 = nn.Linear(64, 64, bias=True)
        self.nm_4096 = nn.BatchNorm1d(64)
        self.rl_4096 = nn.Sigmoid()
        self.topk_score_4096 = nn.Linear(64, 1, bias=True)
        self.nm_score_4096 = nn.BatchNorm1d(1)
        self.rl_score_4096 = nn.Sigmoid()
        # topk_4096

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        score3 = []
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :] # 坐标
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] # torch.Size([8032])
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            score2 = 0
            if pillars.shape[0] > 4096:
                pillars = self.topk_4096(pillars)
                pillars = self.nm_4096(pillars)
                pillars = self.rl_4096(pillars)
                score = self.topk_score_4096(pillars)
                score = self.nm_score_4096(score).squeeze()
                score = self.rl_score_4096(score)
                # score = F.gumbel_softmax(score, hard=True).squeeze()
                top_score,index= torch.topk(score,4096,largest=True)
                top_score1 = top_score
                index1 = index
                indices = indices[index]
                pillars = pillars[index]
                mask = score == score
                mask[index1] = False
                score = torch.mean(score[mask])
                score2 = score.item()
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
            score3.append(score2)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict.update({'score3':score3})
        return batch_dict