import numpy as np

from torch.utils import data
from torch.utils.data import Dataset

import trimesh
import pysdf

# SDF dataset
class SDFDataset(Dataset):
    def __init__(self, path, net=None, size=100, num_samples=2**18, clip_sdf=None,
                 cache_size=-1,
                 sample_vertices=False,
                 num_sup_vertices=None,
                 normalize_mesh=True,
                 use_sphere_gt_sdf=False,
                 bound=1.0,
                 sample_from_net=False,
                 sigma=None,
                 n_walks=None,
                 reproj_mode="ad"):
        super().__init__()
        self.path = path
        self.net = net
        self.sample_from_net = sample_from_net
        self.sample_vertices = sample_vertices
        self.num_sup_vertices = num_sup_vertices
        self.vertices_cache = None
        self.bound = bound

        # for network-based point sampling
        self.sigma = sigma
        self.n_walks = n_walks
        self.reproj_mode = reproj_mode

        # load obj
        self.mesh = trimesh.load(path, force='mesh')

        self.normalize_mesh = normalize_mesh
        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        vs = self.mesh.vertices
        if normalize_mesh:
            vmin = vs.min(0)
            vmax = vs.max(0)
            v_center = (vmin + vmax) / 2
            v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
            vs = (vs - v_center[None, :]) * v_scale
        self.mesh.vertices = vs

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")
        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)

        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size
        self.cache_size = cache_size
        if self.cache_size > 0:
            self.points_surface = None
            self.points_uniform = None
            self.sdfs_surface = None
            self.sdfs_uniform = None

        self.radius = np.mean(np.linalg.norm(self.mesh.vertices, axis=-1))
        self.use_sphere_gt_sdf = use_sphere_gt_sdf

        self.surface_cache = None
        self.surf_batch_idx = 0
        
        
    def __len__(self):
        return self.size

    def sample_online(self, num_samples):
        # online sampling
        sdfs = np.zeros((num_samples, 1))
        # surface (randomly sample or just use the vertices)
        if not self.sample_vertices: # then we sample surface
            points_surface = None
            if (self.sample_from_net) and (self.net is not None):
                sidx = self.surf_batch_idx * num_samples * 7 // 8
                eidx = (self.surf_batch_idx + 1) * num_samples * 7 // 8
                points_surface = self.surface_cache[sidx:eidx]
                self.surf_batch_idx += 1
                # if self.surf_batch_idx >= self.size:
                #     self.surf_batch_idx = 0
            else:
                points_surface = self.mesh.sample(num_samples * 7 // 8)
        else: # sample only vertices, not the triangle
            if self.vertices_cache is None:
                if self.num_sup_vertices is None:
                    self.vertices_cache = self.mesh.vertices
                else:
                    idx = np.random.choice(
                        self.mesh.vertices.shape[0], size=int(self.num_sup_vertices))
                    self.vertices_cache = self.mesh.vertices[idx]
            idx = np.random.choice(
                self.vertices_cache.shape[0], size=(num_samples * 7 // 8))
            points_surface = self.vertices_cache[idx]

        # perturb surface
        points_surface[num_samples // 2:] += 0.01 * np.random.randn(num_samples * 3 // 8, 3)
        # random uniform
        points_uniform = np.random.rand(num_samples // 8, 3) * (2 * self.bound) - self.bound

        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)
        if not self.use_sphere_gt_sdf:
            sdfs[num_samples // 2:] = -self.sdf_fn(points[num_samples // 2:])[:,None].astype(np.float32)
        else:
            # get SDF of all points using ground truth SDF for sphere
            sdfs = (np.linalg.norm(points, axis=-1) - self.radius)[:,None].astype(np.float32)

        if self.cache_size > 0:
            n_sfn = points_surface.shape[0]
            if self.points_surface is None:
                self.points_surface = points[:n_sfn]
                self.sdfs_surface = sdfs[:n_sfn]
            elif self.points_surface.shape[0] < self.cache_size * 7 // 8:
                self.points_surface = np.concatenate([self.points_surface, points[:n_sfn]], axis=0)
                self.sdfs_surface = np.concatenate([self.sdfs_surface, sdfs[:n_sfn]], axis=0)

            if self.points_uniform is None:
                self.points_uniform = points[n_sfn:]
                self.sdfs_uniform = sdfs[n_sfn:]
            elif self.points_uniform.shape[0] < self.cache_size // 8:
                self.points_uniform = np.concatenate([self.points_uniform, points[n_sfn:]], axis=0)
                self.sdfs_uniform = np.concatenate([self.sdfs_uniform, sdfs[n_sfn:]], axis=0)
        return points, sdfs

    def sample_points(self, num_samples):
        if (self.cache_size > 0 and
                self.points_surface is not None and
                self.points_surface.shape[0] >= self.cache_size * 7 // 8 and
                self.points_uniform is not None and
                self.points_uniform.shape[0] >= self.cache_size // 8):
            idx_surface = np.random.choice(
                self.points_surface.shape[0], size=(num_samples * 7 // 8))
            points_surface = self.points_surface[idx_surface]
            sdfs_surface = self.sdfs_surface[idx_surface]

            idx_uniform = np.random.choice(
                self.points_uniform.shape[0], size=(num_samples // 8))
            points_uniform = self.points_uniform[idx_uniform]
            sdfs_uniform = self.sdfs_uniform[idx_uniform]
            points = np.concatenate([points_surface, points_uniform], axis=0) # (n, 3)
            sdfs = np.concatenate([sdfs_surface, sdfs_uniform], axis=0)
            return points.astype(np.float32), sdfs.astype(np.float32)
        else:
            return self.sample_online(num_samples)

    def __getitem__(self, _):
        points, sdfs = self.sample_points(self.num_samples)

        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        return {
            'sdf': sdfs,
            'xyz': points,
        }


# Abstraction - turning cfg into arguments
# NOTE: in trainer we should have something simular to that
def get_data_loaders(cfg, net=None):
    cfg_tr = cfg.train
    tr_dataset = SDFDataset(
        cfg_tr.path,
        net=net,
        size=cfg_tr.size,
        num_samples=cfg_tr.get("num_samples", 2**18),
        clip_sdf=None,
        cache_size=cfg_tr.get("cache_size", -1),
        sample_vertices=cfg_tr.get("sample_vertices", False),
        num_sup_vertices=cfg_tr.get("num_sup_vertices", None),
        normalize_mesh=cfg_tr.get("normalize_mesh", True),
        use_sphere_gt_sdf=cfg_tr.get("use_sphere_gt_sdf", False),
        sample_from_net=cfg_tr.get("sample_from_net", False),
        sigma=cfg_tr.get("sigma", 2e-2),
        n_walks=cfg_tr.get("n_walks", 256),
        reproj_mode=cfg_tr.get("reproj_mode", "ad")
    )
    cfg_te = cfg.val
    te_dataset = SDFDataset(
        cfg_te.path,
        net=net,
        size=cfg_te.size,
        num_samples=cfg_te.get("num_samples", 2**18),
        clip_sdf=None,
        cache_size=cfg_te.get("cache_size", -1),
        sample_vertices=cfg_te.get("sample_vertices", False),
        num_sup_vertices=cfg_te.get("num_sup_vertices", None),
        normalize_mesh=cfg_tr.get("normalize_mesh", True),
        use_sphere_gt_sdf=cfg_te.get("use_sphere_gt_sdf", False),
        sample_from_net=cfg_tr.get("sample_from_net", False),
        sigma=cfg_tr.get("sigma", 2e-2),
        n_walks=cfg_tr.get("n_walks", 256),
        reproj_mode=cfg_tr.get("reproj_mode", "ad")
    )
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.get("num_workers", cfg.num_workers),
        drop_last=True)
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=cfg.val.batch_size, shuffle=False,
        num_workers=cfg.val.get("num_workers", cfg.num_workers),
        drop_last=False)

    return {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
