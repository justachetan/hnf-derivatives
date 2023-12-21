import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import time
from typing import Callable, List, Tuple
from typing_extensions import Literal

import igl
import trimesh

import numpy as np
import scipy as sp
import torch
import torch.nn as nn

from utils.utils import gradient, jacobian

def get_on_surf_pts(
    gtmesh: trimesh.Trimesh,
    npnts: int = 2**18,
    mode: Literal["sfn", "curv"] = "sfn",
) -> Tuple[np.array, np.array]:

    v, f = gtmesh.vertices, gtmesh.faces
    on_surf_pts, faces = trimesh.sample.sample_surface(gtmesh, npnts)
    on_surf_pts = np.array(on_surf_pts)

    on_surf_bary = trimesh.triangles.points_to_barycentric(gtmesh.triangles[faces], on_surf_pts)

    if mode == "sfn":
        gt_sfn_v = igl.per_vertex_normals(v, f)
        gt_sfn = np.sum(on_surf_bary.reshape(-1, 3, 1) * gt_sfn_v[f[faces]], axis=-2)
        gt_sfn = gt_sfn / (np.linalg.norm(gt_sfn, axis=-1, ord=2)[..., None] + 1e-6)

        return on_surf_pts, gt_sfn
    elif mode == "curv":
        l = igl.cotmatrix(v, f)
        m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)

        minv = sp.sparse.diags(1 / m.diagonal())

        hn = -minv.dot(l.dot(v))
        gt_curv_v = np.linalg.norm(hn, axis=1).reshape(-1, 1) / 2


        gt_curv = np.sum(on_surf_bary.reshape(-1, 3, 1) * gt_curv_v[f[faces]], axis=-2)
        gt_curv = gt_curv.reshape(-1)

        return on_surf_pts, gt_curv
    
def l2_err_sfn(pred_sfn, gt_sfn):
    return np.mean(np.linalg.norm(pred_sfn - gt_sfn, axis=-1, ord=2))
    
def ang_err_sfn(pred_sfn, gt_sfn, reduce="mean"):
    ang_err_all = np.abs(np.arccos(np.clip((pred_sfn * gt_sfn).sum(axis=-1), -1, 1)) / np.pi * 180.)
    if reduce is None:
        return ang_err_all
    elif reduce == "mean":
        return np.mean(ang_err_all)


def perc_ang_acc_below_k(pred_sfn, gt_sfn, k=1):
    """percentage of points having angular error less than k degrees

    Args:
        pred_sfn (np.array): predicted surface normal
        gt_sfn (np.array): ground-truth surface normal
        k (int, optional): threshold for angle error. Defaults to 1.

    Returns:
        float: angular accuracy at threshold k
    """
    ang_err_dist = np.abs(np.arccos(np.clip((pred_sfn * gt_sfn).sum(axis=-1), -1, 1)) / np.pi * 180.)
    return np.sum(ang_err_dist <= k) / ang_err_dist.shape[0]



def rec_rel_error(pred_curv, gt_curv, reduce: Literal["mean", "rmse", None] = "rmse"):
    """
    Rectified relative error
    Source: PCP-Net. https://arxiv.org/pdf/1710.04954.pdf
    """
    err_vals = np.abs((pred_curv - gt_curv) / np.maximum(np.abs(gt_curv), 1))
    if reduce == "mean":
        return np.mean(err_vals)
    elif reduce == "rmse":
        return np.linalg.norm(err_vals, ord=2) / np.sqrt(err_vals.shape[0])
    return err_vals

### first-order AD error analysis ###

def get_ad_stats_fo(
    on_surf_pts: torch.cuda.FloatTensor,
    model: nn.Module,
    err_funcs: List[Callable],
    gt_normals: np.array,
    return_time: bool = False,
):
    
    ad_grad_time_start = time.time()

    on_surf_pts.requires_grad_()
    sdfs = model(on_surf_pts)
    ad_normals = gradient(sdfs, on_surf_pts)
    ad_normals /= (ad_normals.norm(dim=-1, keepdim=True) + 1e-6)
    ad_normals = ad_normals.detach().cpu().numpy()
    ad_grad_time = time.time() - ad_grad_time_start

    on_surf_pts = on_surf_pts.detach()

    stats = [err_func(ad_normals, gt_normals) for err_func in err_funcs]

    if return_time:
        return ad_normals, stats, ad_grad_time

    return ad_normals, stats


### second-order AD error analysis ###

def get_ad_stats_so(
    on_surf_pts: torch.cuda.FloatTensor,
    model: nn.Module,
    err_funcs: List[Callable],
    gt_curv: np.array,
    return_time: bool = False,
    batched: bool = True,
):
    
    ad_curv = None
    ad_curv_time_start = time.time()
    bs = 8192
    
    if batched:
        ad_curv = np.zeros(on_surf_pts.shape[0])
        for i in range(0, on_surf_pts.shape[0], bs):
            sidx = i
            eidx = min(i + bs, on_surf_pts.shape[0])
            batch_pts = on_surf_pts[sidx:eidx]
            batch_pts.requires_grad_()
            sdfs = model(batch_pts)
            ad_normals = gradient(sdfs, batch_pts)
            norm_ad_normals = ad_normals / (ad_normals.norm(dim=-1, keepdim=True) + 1e-6)
            ad_curv[sidx:eidx] = 0.5 * torch.einsum("bii", \
                jacobian(norm_ad_normals, batch_pts)[0]).detach().cpu().numpy().reshape(-1)
            batch_pts = batch_pts.detach()
    else:
        on_surf_pts.requires_grad_()
        sdfs = model(on_surf_pts)
        ad_normals = gradient(sdfs, on_surf_pts)
        norm_ad_normals = ad_normals / (ad_normals.norm(dim=-1, keepdim=True) + 1e-6)
        ad_curv = 0.5 * torch.einsum("bii", jacobian(norm_ad_normals, on_surf_pts)[0])
        ad_curv = ad_curv.detach().cpu().numpy().reshape(-1)
        on_surf_pts = on_surf_pts.detach()

    ad_curv_time = time.time() - ad_curv_time_start

    stats = [err_func(ad_curv, gt_curv) for err_func in err_funcs]

    if return_time:
        return ad_curv, stats, ad_curv_time

    return ad_curv, stats
    