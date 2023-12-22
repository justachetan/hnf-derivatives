import os
import time
import random
import logging
import importlib
from functools import partial

import jax

import torch
import skimage
from torch.autograd import grad

from torch.utils.data import TensorDataset

import numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R


class ConfigNameSpace(object):
    def __init__(self, config_dict: dict = None):
        for k in config_dict:
            setattr(self, k, config_dict[k])

def get_encoder(encoding, input_dim=3, 
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False,
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim
    
    elif encoding == 'frequency':
        #encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)
        from models.freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'sphere_harmonics':
        from models.shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        from models.gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)
    
    elif encoding == 'tiledgrid':
        from models.gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners)
    
    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim

def get_pylogger(name=__name__):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, getattr(logger, level))

    return logger

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def shape_jacobian(y, x):
    """
    Jacobian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    ret: shape (meta_batch_size, num_observations, channels, dim)
    """
    meta_batch_size, num_observations = y.shape[:2]
    print("meta_batch_size", meta_batch_size, num_observations)
    # (meta_batch_size*num_points, 2, 2)
    jac = torch.zeros(
        meta_batch_size, num_observations,
        y.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        # print(y_flat.shape, y_flat.requires_grad, x.shape, x.requires_grad)
        jac[:, :, i, :] = grad(
            y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status

def jacobian(y, x):
    """
    Jacobian of y wrt x
    y: shape (num_observations, channels)
    x: shape (num_observations, dim)
    ret: shape (num_observations, channels, dim)
    """
    num_observations = y.shape[0]
    # print("meta_batch_size", num_observations)
    # (meta_batch_size*num_points, 2, 2)
    jac = torch.zeros( num_observations,
        y.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, i, :] = grad(
            y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(
            y[..., i], x, torch.ones_like(y[..., i]),
            create_graph=True)[0][..., i:i+1]
    return div

def laplace(y, x, normalize=False, eps=0., return_grad=False):
    grad = gradient(y, x)
    if normalize:
        grad = grad / (grad.norm(dim=-1, keepdim=True) + eps)
    div = divergence(grad, x)

    if return_grad:
        return div, grad
    return div

def get_zero_pts(
    model, 
    mesh_res: int = 175, 
    offset: float = 0, 
    batch_size: int = 8192, 
    fp16: bool = False,
    time_milli: bool = False,
    time_prec: int = 3,
    level: float = 0.0,
):

        mc_fwd_time = 0
        mc_time = 0
        fact = 1 if not time_milli else 1000

        res = mesh_res + np.random.randint(low=-3, high=3) # trick borrowed from NIE
        offset = offset
        bound = 1.
        batch_size = batch_size

        xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
        grid = np.concatenate([
            ys[..., np.newaxis],
            xs[..., np.newaxis],
            zs[..., np.newaxis]
        ], axis=-1).astype(float)
        grid = (grid / float(res - 1) - 0.5) * 2 * bound
        grid = grid.reshape(-1, 3)
        voxel_size = 2.0 / (res - 1)
        voxel_origin = -1 * bound

        dists_lst = np.zeros(grid.shape[0])
        pbar = range(0, grid.shape[0], batch_size)
        print("GRID", grid.min(), grid.max())
        mc_fwd_start = time.time()
        model.cuda()
        for i in pbar:
            sidx, eidx = i, i + batch_size
            eidx = min(grid.shape[0], eidx)

            with torch.no_grad():
                xyz = torch.from_numpy(
                    grid[sidx:eidx, :]).float().cuda().view(-1, 3)
                distances = model.forward(xyz-offset)
                distances = distances.cpu().numpy()
            dists_lst[sidx:eidx] = distances.reshape(-1)
        
        mc_fwd_time += round((time.time() - mc_fwd_start) * fact, time_prec)

        dists = dists_lst.reshape(-1)
        field = dists.reshape(res, res, res)
        mc_start = time.time()
        vert, face, _, _ = skimage.measure.marching_cubes(
            field, level=level, spacing=[voxel_size]*3, method='lorensen')
        mc_time += round((time.time() - mc_start) * fact, time_prec)
        vert += voxel_origin
        vert -= offset
        # print("zero_pts", vert.shape, face.shape)
        return vert, face, (mc_time, mc_fwd_time)

def get_onsurf_pts(
    model: torch.nn.Module,
    grad_model: torch.nn.Module = None,
    num_onsurf_pts: int= 2**18,
    bs: int= 2**18,
    bound: float = 1.0,
    dim: int = 3,
    num_workers: int = 8,
    **kwargs
):
    """Generate on surface points for an SDF

    Args:
        model (torch.nn.module): Neural SDF model, assumed to be on GPU
        grad_model (torch.nn.module, optional): Gradient Field model, assumed to be on GPU
        num_onsurf_pts (int, optional): Number of on-surface points to sample. Defaults to 2**18.
        bs (int, optional): batch size to use for inernal forward pass. Defaults to 2**18.
        bound (float, optional): domain of SDF. assumed to be from [-bound, bound]. Defaults to 1.0.
        dim (int, optional): input point dimension. Defaults to 3.
        num_workers (int, optional): num_workers for internal forward pass over data. Defaults to 8.

    Returns:
        torch.Tensor: #num_onsurf_pts x dim
    """
    cand_pts = (torch.rand(size=(num_onsurf_pts, dim)) * 2 * bound) - bound
    all_grad_sdfs = torch.zeros((num_onsurf_pts, dim))
    all_sdfs = torch.zeros(num_onsurf_pts)
    dataloader = torch.utils.data.DataLoader(TensorDataset(cand_pts), \
        shuffle=False, batch_size=bs, num_workers=num_workers, drop_last=False, **kwargs)
    
    for (i, data) in enumerate(dataloader):

        sidx = i*bs
        eidx = min(num_onsurf_pts, (i+1)*bs)

        xyz = data[0].cuda()
        if grad_model is None:
            xyz.requires_grad_()
            # print(xyz.shape)
            sdfs = model(xyz).float().reshape(-1)
            grad_sdfs = gradient(sdfs, xyz)
        else:
            with torch.no_grad():
                sdfs = model(xyz).float().reshape(-1)
                grad_sdfs = grad_model(xyz).float().reshape(-1, xyz.shape[-1])

        xyz = xyz.detach().cpu()
        sdfs = sdfs.detach().cpu()
        grad_sdfs = grad_sdfs.detach().cpu()
        all_grad_sdfs[sidx:eidx] = grad_sdfs
        all_sdfs[sidx:eidx] = sdfs

    all_grad_sdfs /= (torch.linalg.norm(all_grad_sdfs, ord=2, dim=-1).reshape(-1, 1) + 1e-6)
    on_surf_pts = cand_pts - (sdfs.reshape(-1, 1) * all_grad_sdfs)

    return on_surf_pts

def instantiate(target, **kwargs):
    module_name = ".".join(target.split(".")[:-1])
    cls_name = target.split(".")[-1]

    module = importlib.import_module(module_name)
    cls_obj = getattr(module, cls_name)
    return cls_obj(**kwargs)

def prepare_data(data, device="cuda"):
    if isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, np.ndarray):
                data[i] = torch.from_numpy(v).to(device, non_blocking=True)
            if torch.is_tensor(v):
                data[i] = v.to(device, non_blocking=True)
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(device, non_blocking=True)
            if torch.is_tensor(v):
                data[k] = v.to(device, non_blocking=True)
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(device, non_blocking=True)
    else: # is_tensor, or other similar objects that has `to`
        data = data.to(device, non_blocking=True)

    return data


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------

def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0, n/-x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)

def random_rotation():
    r = R.random().as_matrix()
    r = np.hstack([r, np.zeros((3, 1))])
    r = np.vstack([r, np.zeros((1, 4))])
    r[3, 3] = 1
    return r

def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m = np.identity(3)
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m

@partial(jax.jit, static_argnums=3)
def interp3d(feat, query, eps=1e-8, kernel="linear"):
    """Trilinear interpolation of [feat] using query points [query].
    
    Args:
        [feat]:  shape=(B, W, H, D), 
                 feat[0, 0] -> coordinate (-1, -1), feat[-1, -1] -> coordinate=(1, 1)
        [query]: shape=(..., 3), range [-1, 1].

    Returns:
        [out]: shape=(..., D), same batch dimension as [query].

    TODO: implement RBF kernel
    """
    B, W, H, D = feat.shape
    
    # Transform to local coordinate
    query = jnp.clip(query, -1 + eps, 1 - eps)
    size = jnp.array([B- 1, W - 1, H - 1]).reshape(*([1] * (len(query.shape) - 1) + [3]))
    local_query = 0.5 * (query + 1) * size # range (0, H or W)

    # Index
    local_i = jnp.floor(local_query).astype(np.int32) # [0, H-2 or W - 2]
    local_j = local_i + 1 # [1, H-1 or W - 1]
    
    # Ratio for each of the index
    local_p = local_query - local_i # (corresponding to local_i index, 0, 0)
    local_q = 1 - local_p # (corresponding to local_j index, 1, 1)
    
    # Query the feature, shape=(..., D), same as batch dimension
    feat_000 = feat[local_i[..., 0], local_i[..., 1], local_i[..., 2]]
    feat_001 = feat[local_i[..., 0], local_i[..., 1], local_j[..., 2]]
    feat_010 = feat[local_i[..., 0], local_j[..., 1], local_i[..., 2]]
    feat_011 = feat[local_i[..., 0], local_j[..., 1], local_j[..., 2]]
    feat_100 = feat[local_j[..., 0], local_i[..., 1], local_i[..., 2]]
    feat_101 = feat[local_j[..., 0], local_i[..., 1], local_j[..., 2]]
    feat_110 = feat[local_j[..., 0], local_j[..., 1], local_i[..., 2]]
    feat_111 = feat[local_j[..., 0], local_j[..., 1], local_j[..., 2]]
    
    # Compute the weights
    if kernel == "linear":
        w_000 = (local_q[..., 0] * local_q[..., 1] * local_q[..., 2])[..., None]
        w_001 = (local_q[..., 0] * local_q[..., 1] * local_p[..., 2])[..., None]
        w_010 = (local_q[..., 0] * local_p[..., 1] * local_q[..., 2])[..., None]
        w_011 = (local_q[..., 0] * local_p[..., 1] * local_p[..., 2])[..., None]
        w_100 = (local_p[..., 0] * local_q[..., 1] * local_q[..., 2])[..., None]
        w_101 = (local_p[..., 0] * local_q[..., 1] * local_p[..., 2])[..., None]
        w_110 = (local_p[..., 0] * local_p[..., 1] * local_q[..., 2])[..., None]
        w_111 = (local_p[..., 0] * local_p[..., 1] * local_p[..., 2])[..., None]
        
        # print("weight sum:", w_000 + w_001 + w_010 + w_011 + w_100 + w_101 + w_110 + w_111)
        # print(w_00 + w_01 + w_10 + w_11) # should equal to 1
    elif kernel == "rbf":
        raise NotImplementedError("RBF kernel is not implemented")
        # # beta = 1e1 # = (2 \sigma)^{-1}
        # beta = 1e0 # = (2 \sigma)^{-1}
        # def f(x, y):
        #     return jnp.exp(- beta * (x**2 + y**2)**0.5)[..., None]
        # w_00 = f(local_p[..., 0], local_p[..., 1])
        # w_01 = f(local_p[..., 0], local_q[..., 1])
        # w_10 = f(local_q[..., 0], local_p[..., 1])
        # w_11 = f(local_q[..., 0], local_q[..., 1])
        # ttlw = w_00 + w_01 + w_10 + w_11
        # w_00 = w_00 / ttlw
        # w_10 = w_10 / ttlw
        # w_01 = w_01 / ttlw
        # w_11 = w_11 / ttlw
    else:
        raise ValueError
    
    return w_000 * feat_000 + w_001 * feat_001 + w_010 * feat_010 + w_011 * feat_011 + \
        w_100 * feat_100 + w_101 * feat_101 + w_110 * feat_110 + w_111 * feat_111