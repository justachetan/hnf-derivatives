from typing_extensions import Literal
from typing import Tuple 

import trimesh

import igl

import jax
import jax.numpy as jnp
import jax.dlpack as jdp

import numpy as np

import torch

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.dlpack import from_dlpack, to_dlpack

from utils.utils import jacobian, gradient


def get_on_surf_pts(
    gtmesh: trimesh.Trimesh,
    npnts: int = 2**18,
) -> Tuple[np.array, np.array]:

    v, f = gtmesh.vertices, gtmesh.faces
    on_surf_pts, faces = trimesh.sample.sample_surface(gtmesh, npnts)
    on_surf_pts = np.array(on_surf_pts)

    on_surf_bary = trimesh.triangles.points_to_barycentric(gtmesh.triangles[faces], on_surf_pts)

    gt_sfn_v = igl.per_vertex_normals(v, f)
    gt_sfn = np.sum(on_surf_bary.reshape(-1, 3, 1) * gt_sfn_v[f[faces]], axis=-2)
    gt_sfn = gt_sfn / (np.linalg.norm(gt_sfn, axis=-1, ord=2)[..., None] + 1e-6)

    if np.isnan(gt_sfn).any():
        on_surf_pts = on_surf_pts[~np.isnan(gt_sfn).any(axis=1)]
        gt_sfn = gt_sfn[~np.isnan(gt_sfn).any(axis=1)]

    return on_surf_pts, gt_sfn

def make_points(
    npnts: int = 2**20,
    bound: float = 1.0,
    dim: int = 3
):
    """
    sample points uniformly from [-bound, bound]
    returns:
        grid (np.array): H x W x D x d
    """

    xy = (torch.rand(size=(npnts, dim)) * 2 * bound) - bound
    return xy

## Samplers

def gauss_sample(
    sigma: float,
    n_samples: int, 
    n_walks: int,
    dim: int=3,
):
    return torch.cuda.FloatTensor(n_samples, n_walks, dim).normal_() * sigma 

def rej_sample(
    sigma: float,
    n_samples: int,
    n_walks: int,
    xq: torch.Tensor,
    rej_tol: float = 0.9,
    dim: int=3,
):

    tmp = torch.cuda.FloatTensor(n_samples * n_walks, dim)
    q = xq.repeat_interleave(n_walks, dim=0)
    filled = torch.zeros(n_samples * n_walks).bool().cuda()

    while not (filled.all()):

        noise = torch.cuda.FloatTensor((n_samples * n_walks), dim).normal_() * sigma 
        b = q + noise
        valid = (torch.abs(b) < rej_tol).all(dim=-1)

        tmp[(~filled) & valid] = noise[(~filled) & valid].float()

        
        old_filled = filled.clone()

        filled[(~old_filled) & valid] = True

    return tmp.reshape(n_samples, n_walks, dim)

def unif_ball_sample(
    sigma: float,
    n_samples: int,
    n_walks: int,
    dim: int=3
):
    """
    source: https://en.wikipedia.org/wiki/N-sphere#Uniformly_at_random_within_the_n-ball
    """
    radius = sigma / 2.
    points = torch.normal(mean=0, std=1, size=(n_samples, n_walks, dim))
    points /= torch.linalg.norm(points, dim=-1, ord=2).reshape(n_samples, n_walks, 1)
    points *= torch.pow(torch.rand(size=(n_samples, n_walks, 1)), 1 / dim) * radius

    return points

def unif_box_sample(
    sigma: float, 
    n_samples: int,
    n_walks: int,
    dim: int=3
):
    bound = sigma / 2.
    points = torch.rand(size=(n_samples, n_walks, dim)) * bound - (bound / 2)

    return points


SAMPLERS = {
    "gauss": gauss_sample,
    "uniball": unif_ball_sample,
    "unibox": unif_box_sample,
    "rej": rej_sample,
}

## Aggregation


def mean_agg(
    Xs: torch.Tensor, # (num_walks * bs) x dim
    sdfs: torch.Tensor, # (num_walks * bs) x 1
    n_walks: int,
    **kwargs
):

    grad_Xs = gradient(sdfs, Xs)

    Xs = Xs.detach().cpu()
    sdfs = sdfs.detach().cpu()
    grad_Xs = grad_Xs.detach().cpu()

    dim = Xs.shape[-1]

    grad_Xs = grad_Xs.reshape(-1, n_walks, dim)
    est_gradX = torch.sum(grad_Xs, dim=1).reshape(-1, dim) / n_walks

    return est_gradX

def mean_agg_curv(
    Xs: torch.Tensor, # (num_walks * bs) x dim
    sdfs: torch.Tensor, # (num_walks * bs) x 1
    n_walks: int,
    **kwargs,
):
    dim = Xs.shape[-1]

    ad_normals = gradient(sdfs, Xs)
    norm_ad_normals = ad_normals / (ad_normals.norm(dim=-1, keepdim=True) + 1e-6)

    ad_jac = jacobian(norm_ad_normals, Xs)[0].detach() # num_walks * bs x dim x dim
    ad_jac = ad_jac.reshape(-1, n_walks, dim, dim)
    
    ad_jac = ad_jac.detach().cpu()
    ad_normals = ad_normals.detach().cpu()
    norm_ad_normals = norm_ad_normals.detach().cpu()
    Xs = Xs.detach()
    sdfs = sdfs.detach()

    ad_curv = 0.5 * torch.einsum("bijj->b", ad_jac).reshape(-1) / n_walks

    

    return ad_curv

def plane_fit_agg(
    Xs: torch.Tensor, 
    sdfs: torch.Tensor, 
    n_walks: int,
    return_all: bool = False
):
    
    dim = Xs.shape[-1]
    sdf_dim = sdfs.shape[-1]

    bias_Xs = torch.ones(Xs.shape[0], dim+1).float()
    bias_Xs[:, :dim] = Xs


    bias_Xs = bias_Xs.reshape(-1, n_walks, dim+1).cuda()
    if sdf_dim > 1:
        sdfs = sdfs.reshape(-1, n_walks, sdf_dim).float()
    else:
        sdfs = sdfs.reshape(-1, n_walks).float()

    # bs x n_walks x 4, bs x n_walks x 3

    sol = torch.linalg.lstsq(bias_Xs, sdfs, driver="gels").solution

    if sdf_dim == 1:
        est_gradX = sol[:, :dim]
    elif sdf_dim > 1:
        est_gradX = sol[:, :dim, :]

    if return_all:
        return est_gradX, sol
    return est_gradX

@jax.jit
def jax_quad_fit_3d(
    Xs: torch.Tensor, # n_walks x dim
    ys: torch.Tensor, # n_walks x 1
):
    
    n = Xs.shape[0]

    px, py, pz = Xs[:, 0:1], Xs[:, 1:2], Xs[:, 2:]

    g = jnp.concatenate([
        jnp.ones((n, 1)),
        px,
        py,
        pz,
        px * py,
        py * pz,
        px * pz,
        px ** 2,
        py ** 2, 
        pz ** 2
    ], axis=-1)

    b = (ys * g).sum(axis=0).reshape(10) 

    col_a = g[..., None] # n x 10 x 1
    col_b = (g * px)[..., None] # n x 10 x 1
    col_c = (g * py)[..., None] # n x 10 x 1
    col_d = (g * pz)[..., None] # n x 10 x 1
    col_e = (g * px * py)[..., None] # n x 10 x 1
    col_f = (g * py * pz)[..., None] # n x 10 x 1
    col_g = (g * px * pz)[..., None] # n x 10 x 1
    col_h = (g * px ** 2)[..., None] # n x 10 x 1
    col_i = (g * py ** 2)[..., None] # n x 10 x 1
    col_j = (g * pz ** 2)[..., None] # n x 10 x 1

    A = jnp.concatenate([
        col_a,
        col_b,
        col_c,
        col_d,
        col_e,
        col_f,
        col_g,
        col_h,
        col_i,
        col_j
    ], axis=-1).sum(axis=0) # 10 x 10

    fit, _, _, _ = jnp.linalg.lstsq(A, b)

    return fit

jax_quad_fit_3d_vmap = jax.jit(
    jax.vmap(
        lambda x, y: jax_quad_fit_3d(x, y)
    )
)


@jax.jit
def jax_quad_fit_2d(
    Xs: torch.Tensor, # n_walks x dim
    ys: torch.Tensor, # n_walks x 1
):
    
    n = Xs.shape[0]

    px, py = Xs[:, 0:1], Xs[:, 1:]

    g = jnp.concatenate([
        jnp.ones((n, 1)),
        px, 
        py, 
        px * px, 
        px * py,
        py * py
    ], axis=-1) # (n, 6)

    b = (ys * g).sum(axis=0).reshape(6) 

    col_a = g[..., None] # n x 6 x 1
    col_b = (g * px)[..., None] # n x 6 x 1
    col_c = (g * py)[..., None] # n x 6 x 1
    col_d = (g * px ** 2)[..., None] # n x 6 x 1
    col_e = (g * px * py)[..., None] # n x 6 x 1
    col_f = (g * py ** 2)[..., None] # n x 6 x 1

    A = jnp.concatenate([
        col_a,
        col_b,
        col_c,
        col_d,
        col_e,
        col_f,
    ], axis=-1).sum(axis=0) # 6 x 6

    fit, _, _, _ = jnp.linalg.lstsq(A, b)

    return fit

jax_quad_fit_2d_vmap = jax.jit(
    jax.vmap(
        lambda x, y: jax_quad_fit_2d(x, y)
    )
)

jax_quad_fit_3d_vmap = jax.jit(
    jax.vmap(
        lambda x, y: jax_quad_fit_3d(x, y)
    )
)


AGG = {
    "pfit": plane_fit_agg,
    "mean": mean_agg,
    "quadfit": {
        "jax": jax_quad_fit_3d_vmap,
        "jax2d": jax_quad_fit_2d_vmap,
        "mean": mean_agg_curv,
    }
}

def pfit_grad_est(
    x: torch.Tensor, # points at which to compute polynomial fitting Grad Estimator
    imf: torch.nn.Module,
    n_walks: int = 64,
    bs: int = 8192,
    sigma: float = 0.01,
    clip: bool = False,
    bound: float = 1.0,
    dim: int = 3,
    agg_type: Literal["pfit", "mean"] = "pfit",
    spl_type: Literal["gauss", "uniball", "unibox"] = "gauss",
    norm: bool = True,
    doall: bool = True,
    norm_y: bool = False,
    return_all: bool = False,
    mean_corr: bool = True,
    eps: float = 1e-15, # constant for normalization
    *kwargs
):
    


    def pfit_grad_est_batch(
        xs, imf
    ):

        dim = xs.shape[-1]
        batch_size = xs.shape[0]

        noise = SAMPLERS[spl_type](
            sigma=sigma, 
            n_samples=batch_size, 
            n_walks=n_walks, 
            dim=dim
        )
        if mean_corr:
            n_mean = torch.mean(noise, dim=1)[:, None, ...]
            noise -= n_mean
        
        if not xs.is_cuda:
            xs = xs.cuda()
        
        xs = xs.reshape(batch_size, 1, dim) + noise
        xs = xs.reshape(-1, dim)
        if clip:
            xs = torch.clip(xs, min=-1 * bound, max=bound)
        
        
        ys = None
        if agg_type == "mean": 
            xs.requires_grad_()
            ys = imf(xs)
        else:
            with torch.no_grad():
                ys = imf(xs)

        if norm_y:
            ys = ys / (torch.norm(ys, dim=-1, p=2).reshape(-1, 1) + eps)

        # print(torch.norm(ys, dim=-1, p=2))
        if return_all:
            grad_xs, sol = AGG[agg_type](xs, ys, n_walks, return_all=return_all)
            
            # grad_xs = grad_xs.reshape(bs, n_walks, dim)
            # sum_grad_x_noise[idx*bs:(idx+1)*bs] = torch.sum(grad_xs, axis=1).reshape(-1, dim)
            return grad_xs, sol
        
        grad_xs = AGG[agg_type](xs, ys, n_walks, return_all=return_all)
        return grad_xs





    if n_walks * x.shape[0] > 2**25: 
        doall = False
        # bs = bs # TODO: setting this just for now. (10th May)
        if torch.is_tensor(x) and x.is_cuda:
            x = x.detach().cpu()

    if doall: # compute gradient estimator over the whole data at once
        sol = None
        pfit_grad_x = None
        if return_all:
            pfit_grad_x, sol = pfit_grad_est_batch(x, imf)
        else:
            pfit_grad_x = pfit_grad_est_batch(x, imf)
        if norm:
            pfit_grad_x = pfit_grad_x / (torch.norm(pfit_grad_x, dim=-1, p=2, keepdim=True) + eps)
        
        if return_all:
            return pfit_grad_x, sol
        return pfit_grad_x


    pfit_grad_x = list()
    pfit_sols = list()

    num_workers = 4 * torch.cuda.device_count()
    dataloader = DataLoader(TensorDataset(x), shuffle=False, batch_size=bs, num_workers=num_workers, drop_last=False)

    for idx, xs in enumerate(dataloader):
        # print(xs, xs[0].shape)
        xs = xs[0]
        
        grd = None
        sol = None
        if return_all:
            grd, sol = pfit_grad_est_batch(xs, imf)
            pfit_sols.append(sol)
        else:
            grd = pfit_grad_est_batch(xs, imf)
        pfit_grad_x.append(grd)
    
    pfit_grad_x = torch.cat(pfit_grad_x, dim=0)
    if return_all:
        pfit_sols = torch.cat(pfit_sols, dim=0)
    
    if norm:
        pfit_grad_x = pfit_grad_x / (torch.norm(pfit_grad_x, dim=-1, p=2).reshape(-1, 1) + 1e-6)
    if return_all:
        return pfit_grad_x, pfit_sols
    return pfit_grad_x


### Second-order operators

def vmap_wrapper(f):
    def _f_(x):
        shape = x.shape
        out = jax.vmap(f)(x.reshape(-1, x.shape[-1]))
        return out.reshape(*shape[:-1], *out.shape[1:])
    return jax.jit(_f_)

def make_grad_fn(f):
    def grad_fn(x):
        return jax.grad(f)(x)
    return vmap_wrapper(grad_fn)

def make_sfn_fn(f):
    def gtr_sfn(x):
        g = jax.grad(f)(x)
        return g / jnp.linalg.norm(g, axis=-1, keepdims=True)
    return vmap_wrapper(gtr_sfn)

def make_curv_fn(f):
    def gtr_curv(x):
        dim = x.shape[0]
        return jnp.trace(jax.jacfwd(make_sfn_fn(f))(x)) / (dim - 1)
    return vmap_wrapper(gtr_curv)

def make_lapl_fn(f):
    def gtr_lapl(x):
        dim = x.shape[0]
        return jnp.trace(jax.jacfwd(make_grad_fn(f))(x)) / (dim - 1)
    return vmap_wrapper(gtr_lapl)

@jax.jit
def get_curv_s(
    Xs: jnp.array,
    ys: jnp.array,
    Xq: jnp.array
):
    """divergence of normalized gradient / curvature"""
    fit = jax_quad_fit_3d(Xs, ys)
    # print("fit:", fit.shape)
    # print("Xs", Xs.shape, "ys", ys.shape, "Xq", Xq.shape)

    a, b, c, d, e, f, g, h, i, j = fit

    def fn(X):
        x, y, z = X[0], X[1], X[2]
        return a + b*x + c*y + d*z + e*x*y + f*y*z + g*x*z + h*x**2 + i*y**2 + j*z**2

    curv_fn = make_curv_fn(fn)
    curv = curv_fn(Xq)

    return curv

@jax.jit
def get_curv_2d_s(
    Xs: jnp.array,
    ys: jnp.array,
    Xq: jnp.array
):
    """divergence of normalized gradient / curvature"""
    fit = jax_quad_fit_2d(Xs, ys)
    # print("fit:", fit.shape)
    # print("Xs", Xs.shape, "ys", ys.shape, "Xq", Xq.shape)

    a, b, c, d, e, f = fit

    def fn(X):
        x, y = X[0], X[1]
        return a + b * x + c * y + d * x * x + e * x * y + f * y * y

    curv_fn = make_curv_fn(fn)
    curv = curv_fn(Xq)

    return curv

get_curv = jax.jit(
    jax.vmap(
        lambda Xs, ys, Xq: get_curv_s(Xs, ys, Xq)
    )
)

get_curv_2d = jax.jit(
    jax.vmap(
        lambda Xs, ys, Xq: get_curv_2d_s(Xs, ys, Xq)
    )
)

def get_hess_s(
    Xs: jnp.array,
    ys: jnp.array,
    Xq: jnp.array
):
    fit = jax_quad_fit_3d(Xs, ys)
    # print("fit:", fit.shape)
    # print("Xs", Xs.shape, "ys", ys.shape, "Xq", Xq.shape)

    a, b, c, d, e, f, g, h, i, j = fit

    return 2 * jnp.array([[
        h,      e/2,    f/2,
        e/2,    i,      g/2,
        f/2,    g/2,    j 
    ]])
    

get_hess = jax.jit(
    jax.vmap(
        lambda Xs, ys, Xq: get_hess_s(Xs, ys, Xq)
    )
)

def get_lapl_s(
    Xs: jnp.array,
    ys: jnp.array,
    Xq: jnp.array
):
    
    fit = jax_quad_fit_3d(Xs, ys)
    # print("fit:", fit.shape)
    # print("Xs", Xs.shape, "ys", ys.shape, "Xq", Xq.shape)

    a, b, c, d, e, f, g, h, i, j = fit

    return 2 * (h + i + j)

def get_lapl_2d_s(
    Xs: jnp.array,
    ys: jnp.array,
    Xq: jnp.array
):


    fit = jax_quad_fit_2d(Xs, ys)
    # print("fit:", fit.shape)
    # print("Xs", Xs.shape, "ys", ys.shape, "Xq", Xq.shape)

    a, b, c, d, e, f = fit

    def fn(X):
        x, y = X[0], X[1]
        return a + b * x + c * y + d * x * x + e * x * y + f * y * y

    lapl_fn = make_lapl_fn(fn)
    lapl = lapl_fn(Xq)

    return lapl

get_lapl = jax.jit(
    jax.vmap(
        lambda Xs, ys, Xq: get_lapl_s(Xs, ys, Xq)
    )
)

get_lapl_2d = jax.jit(
    jax.vmap(
        lambda Xs, ys, Xq: get_lapl_2d_s(Xs, ys, Xq)
    )
)


# No Div or Curl as we are considering scalar fields for now
SO_OPS = {
    "curv": get_curv,
    "hess": get_hess,
    "lapl": get_lapl,
    "lapl2D": get_lapl_2d,
    "curv2D": get_curv_2d,
}


def pfit_quad_est(
    x: torch.Tensor,
    imf: torch.nn.Module,
    sigma: float,
    n_walks: int,
    spl_type: Literal["gauss", "uniball", "uniform", "rej"] = "gauss",
    doall: bool = True,
    bs: int = 2048,
    quad_fit_type: Literal["jax", "mean"] = "jax",
    return_op: Literal["hess", "lapl", "div", "curv", "curv2d", "lapl2d"] = "curv",
    mean_corr: bool = True,
    clip: bool = False,
    bound: float = 1.0,
):
    def pfit_quad_est_batch(
        x: torch.Tensor,
    ):

        # print("sigma", sigma)

        num_q = x.shape[0]
        dim = x.shape[1]

        if spl_type == "rej":
            noise =  SAMPLERS[spl_type](
                sigma=sigma,
                n_samples=num_q,
                n_walks=n_walks,
                dim=dim,
                xq=x,
            )
        else:
            noise =  SAMPLERS[spl_type](
                sigma=sigma,
                n_samples=num_q,
                n_walks=n_walks,
                dim=dim,
            )

        if mean_corr:
            n_mean = torch.mean(noise, dim=1)[:, None, ...]
            noise -= n_mean

        xs = x.reshape(num_q, 1, dim) + noise.cuda()

        xs = xs.reshape(-1, dim)

        
        if clip:
            xs = torch.clip(xs, min=-1 * bound, max=bound)
        
        
        ys = None
        if quad_fit_type != "mean":
            with torch.no_grad():
                ys = imf(xs)
        else:
            xs = xs.requires_grad_()
            ys = imf(xs)
            

        if quad_fit_type == "jax":

            xs = xs.reshape(num_q, n_walks, dim)
            ys = ys.reshape(num_q, n_walks, 1)

            xs_jnp = jdp.from_dlpack(to_dlpack(xs.clone()))
            ys_jnp = jdp.from_dlpack(to_dlpack(ys.clone()))

            x_jnp = jdp.from_dlpack(to_dlpack(x.clone()))
            

            so_op = SO_OPS[return_op](xs_jnp, ys_jnp, x_jnp)
            so_op_dlp = jdp.to_dlpack(so_op)
            so_op_pt = from_dlpack(so_op_dlp)

            return so_op_pt

        elif quad_fit_type == "mean":

            if return_op == "curv":
                magg_curv = AGG["quadfit"][quad_fit_type](xs, ys, n_walks=n_walks)
                return magg_curv
            else:
                raise RuntimeError("only curv is supported for mean quadfit")
            

        else:
            raise RuntimeError("quad_fit_type must be jax or mean")

        
    if doall:
        so_op_pt = pfit_quad_est_batch(x)
        return so_op_pt

    else:

        num_q = x.shape[0]
        so_op_pt = None

        for idx in range(0, num_q, bs):
            
            sidx = idx
            eidx = idx + bs
            x_b = x[sidx:eidx]

            if so_op_pt is None:
                out_size = []
                if return_op == "hess":
                    out_size = (3, 3)
                so_op_pt = torch.zeros(num_q, *out_size)
            

            so_op_pt_b = pfit_quad_est_batch(x_b)
            so_op_pt[sidx:eidx] = so_op_pt_b
    
        return so_op_pt



    