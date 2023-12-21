import torch
import torch.nn as nn
import tinycudann as tcnn


class TCNNInstantNGP(nn.Module):

    """Instant NGP written with TCNN"""
    def __init__(self, cfg):
        super().__init__()

        self.num_layers = cfg.get("num_layers", 3)
        self.hidden_dim = cfg.get("hidden_dim", 64)
        self.clip_sdf = cfg.get("clip_sdf", None)
        # accepted activations are here: https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#activation-functions
        self.act = cfg.get("act", "ReLU") 
        self.output_dims = cfg.get("output_dims", 1)

        self.network_type = cfg.get("network_type", "fused") # "fused" or "torch"

        n_input_dims = cfg.get("input_dims", 3)
        
        max_res = cfg.enc.get("max_res", 2048)
        base_res = cfg.enc.get("base_res", 16)
        n_levels = cfg.enc.get("n_levels", 16)
        if n_levels > 1:
            per_level_scale = (
                float(max_res) / float(base_res)) ** (1. / (n_levels - 1.))
        else:
            # NOTE: use max resolution
            per_level_scale = 1.
            base_res = max_res
        n_features_per_level = cfg.enc.get("n_features_per_level", 2)
        self.encoder = tcnn.Encoding(
            n_input_dims=n_input_dims,
            encoding_config={
                "otype": cfg.enc.get("otype", "HashGrid"),
                "type" : cfg.enc.get("type", "Hash"),
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": cfg.enc.get("log2_hashmap_size", 19),
                "base_resolution": base_res,
                "per_level_scale": per_level_scale,
                "interpolation": cfg.enc.get("interpolation", "Linear")
            },
            dtype=torch.float
        )
        
        self.backbone = None

        if self.network_type == "fused":
            self.backbone = tcnn.Network(
                n_input_dims=n_levels * n_features_per_level,
                n_output_dims=self.output_dims,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": self.act,
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1
                }
            )
        else:

            layers = list()
            in_dim = n_levels * n_features_per_level
            out_dim = self.hidden_dim
            for i in range(self.num_layers - 1):
                
                layers.append(
                    nn.Linear(in_dim, out_dim, bias=False)
                )


                if i != self.num_layers - 2:
                    if self.act == "ReLU":
                        layers.append(nn.ReLU())
                    elif self.act == "Sigmoid":
                        layers.append(nn.Sigmoid())

                in_dim = out_dim
                if i == self.num_layers - 3:
                    out_dim = self.output_dims
            

            self.backbone = nn.Sequential(*layers)

        print(self)


    def forward(self, x):
        # x: [B, 3]

        x = (x + 1) / 2.  # to [0, 1]
        x = self.encoder(x)
        h = self.backbone(x)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h


# Wrapper :  interacting interface will be
#   __init__(self, all_cfg, model_cfg)
#   All model will call Net(all_cfg, model_cfg)
class Net(TCNNInstantNGP):
    def __init__(self, all_cfg, cfg):
        super().__init__(cfg)
