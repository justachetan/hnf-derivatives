import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, cfg):
        
        super().__init__()
        res = cfg.get("res", 128)
        emb_dim = cfg.get("emb_dim", 32)
        emb_init_scale = cfg.get("emb_init_scale", 0.001)
        self.embeddings = nn.ParameterList([
            nn.Parameter(
                torch.randn(1, emb_dim, res, res) * emb_init_scale)
            for _ in range(3)])

        # Use this if you want a PE
        hidden_dim = cfg.get("hidden_dim", 128)
        output_dim = cfg.get("output_dim", 1)
        net_lst = []
        num_layers = cfg.get("num_layers", 3)
        curr_dim = emb_dim
        for _ in range(num_layers):
            net_lst.append(nn.Linear(curr_dim, hidden_dim))
            net_lst.append(nn.ReLU(inplace=True))
            curr_dim = hidden_dim
        net_lst.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*net_lst)

    def sample_plane(self, coords2d, plane):
        """
        Args:
            [coords2d]
            [plane]
        """
        sampled_features = torch.nn.functional.grid_sample(
            plane,
            coords2d.reshape(1, coords2d.shape[0], -1, coords2d.shape[-1]),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(
            N, C, H * W).permute(0, 2, 1)
        return sampled_features

    def forward(self, coordinates):
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[2])
        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0)
        return self.net(features)

    def tvreg(self):
        l = 0
        for embed in self.embeddings:
            l += ((embed[:, :, 1:] - embed[:, :, :-1]) ** 2).sum() ** 0.5
            l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1]) ** 2).sum() ** 0.5
        return l