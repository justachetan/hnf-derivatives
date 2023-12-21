from typing import Callable

import numpy as np
import torch


def fd_stencil_fwd(x, delx, model, normalize=False):

    stencil = torch.Tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    nbhrs_x = x[:, None, ...] + delx * stencil.cuda()

    num_samples, stencil_size, dim = nbhrs_x.shape
    nbhrs_x = nbhrs_x.reshape(-1, 3)

    with torch.no_grad():
        sdf_nbhrs_x = model(nbhrs_x)

    sdf_nbhrs_x = sdf_nbhrs_x.reshape(num_samples, stencil_size, 1)

    with torch.no_grad():
        sdf_x = model(x)

    
    grad = (sdf_nbhrs_x - sdf_x[:, None,...]).reshape(-1, 3) / delx

    if normalize:
        grad = grad / (grad.norm(p=2, dim=-1, keepdim=True) + 1e-6)

    return grad

def fd_stencil_cen(x, delx, model, normalize=False):

    dim = x.shape[-1]
    
    fwd_stencil = torch.Tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    bkwd_stencil = torch.Tensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]) 
    
    if dim == 2:
        fwd_stencil = torch.Tensor([
            [1, 0],
            [0, 1]
        ])

        bkwd_stencil = torch.Tensor([
            [-1, 0],
            [0, -1]
        ]) 

    fwd_nbhrs_x = x[:, None, ...] + delx * fwd_stencil.cuda()
    bkwd_nbhrs_x = x[:, None, ...] + delx * bkwd_stencil.cuda()

    num_samples, stencil_size, dim = fwd_nbhrs_x.shape
    fwd_nbhrs_x = fwd_nbhrs_x.reshape(-1, dim)
    bkwd_nbhrs_x = bkwd_nbhrs_x.reshape(-1, dim)

    with torch.no_grad():
        fwd_sdf_nbhrs_x = model(fwd_nbhrs_x)
        bkwd_sdf_nbhrs_x = model(bkwd_nbhrs_x)

    fwd_sdf_nbhrs_x = fwd_sdf_nbhrs_x.reshape(num_samples, stencil_size, 1)
    bkwd_sdf_nbhrs_x = bkwd_sdf_nbhrs_x.reshape(num_samples, stencil_size, 1)
    
    grad = (fwd_sdf_nbhrs_x - bkwd_sdf_nbhrs_x).reshape(-1, dim) / (2 * delx)

    if normalize:
        grad = grad / (grad.norm(p=2, dim=-1, keepdim=True) + 1e-6)

    return grad

def fd_curv_dng_est(
    x, 
    delx,
    model
):

    dim = x.shape[-1]
    
    fwd_stencil = torch.Tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    bkwd_stencil = torch.Tensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]) 
    
    if dim == 2:
        fwd_stencil = torch.Tensor([
            [1, 0],
            [0, 1]
        ])

        bkwd_stencil = torch.Tensor([
            [-1, 0],
            [0, -1]
        ]) 

    fwd_nbhrs_x = x[:, None, ...] + (delx / 2) * fwd_stencil.cuda()
    bkwd_nbhrs_x = x[:, None, ...] + (delx / 2) * bkwd_stencil.cuda()

    num_samples, stencil_size, dim = fwd_nbhrs_x.shape
    fwd_nbhrs_x = fwd_nbhrs_x.reshape(-1, dim)
    bkwd_nbhrs_x = bkwd_nbhrs_x.reshape(-1, dim)
    
    fwd_grad_x = fd_stencil_cen(fwd_nbhrs_x, delx / 2, model)
    norm_fwd_grad_x = fwd_grad_x / (fwd_grad_x.norm(dim=-1, p=2, keepdim=True) + 1e-6)
    norm_fwd_grad_x = norm_fwd_grad_x.reshape(num_samples, stencil_size, dim) # num_samples x 3 x 3
    
    bkwd_grad_x = fd_stencil_cen(bkwd_nbhrs_x, delx / 2, model)
    norm_bkwd_grad_x = bkwd_grad_x / (bkwd_grad_x.norm(dim=-1, p=2, keepdim=True) + 1e-6)
    norm_bkwd_grad_x = norm_bkwd_grad_x.reshape(num_samples, stencil_size, dim) # num_samples x 3 x 3

    hess_diag_norm_grad = (norm_fwd_grad_x - norm_bkwd_grad_x) / delx

    div_norm_grad = 0.5 * torch.einsum("bii", hess_diag_norm_grad)

    return div_norm_grad
    

def fd_curv_dg_est(
    x, 
    delx,
    model
):

    dim = x.shape[-1]
    
    fwd_stencil = torch.Tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    bkwd_stencil = torch.Tensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]) 
    
    if dim == 2:
        fwd_stencil = torch.Tensor([
            [1, 0],
            [0, 1]
        ])

        bkwd_stencil = torch.Tensor([
            [-1, 0],
            [0, -1]
        ]) 

    fwd_nbhrs_x = x[:, None, ...] + (delx / 2) * fwd_stencil.cuda()
    bkwd_nbhrs_x = x[:, None, ...] + (delx / 2) * bkwd_stencil.cuda()

    num_samples, stencil_size, dim = fwd_nbhrs_x.shape
    fwd_nbhrs_x = fwd_nbhrs_x.reshape(-1, dim)
    bkwd_nbhrs_x = bkwd_nbhrs_x.reshape(-1, dim)

    with torch.no_grad():
        fwd_sdf_nbhrs_x = model(fwd_nbhrs_x)
        bkwd_sdf_nbhrs_x = model(bkwd_nbhrs_x)
        sdf_x = model(x)

    fwd_sdf_nbhrs_x = fwd_sdf_nbhrs_x.reshape(num_samples, stencil_size, 1)
    bkwd_sdf_nbhrs_x = bkwd_sdf_nbhrs_x.reshape(num_samples, stencil_size, 1)
    sdf_x = sdf_x.reshape(num_samples, 1, 1)

    div_grad_x = (fwd_sdf_nbhrs_x + bkwd_sdf_nbhrs_x - 2 * sdf_x) / (delx**2 / 4)
    div_grad_x = div_grad_x.sum(dim=1)

    curv = 0.5 * div_grad_x

    return curv
    


def fd_stencil_batch(
    fd_stencil_func: Callable,
    model: torch.nn.Module,
    input_pts: torch.cuda.FloatTensor,
    delx: float,
    dim: int = 3,
    bs: int = 8192,
    normalize: bool = False,
):
    
    num_pnts = input_pts.shape[0]
    fd_grad = np.zeros((num_pnts, dim))
    for i in range(0, num_pnts, bs):
        sidx = i
        eidx = min(i + bs, num_pnts)
        fd_grad[sidx:eidx] = fd_stencil_func(
            input_pts[sidx:eidx], 
            delx, 
            model).detach().cpu().numpy().reshape(-1, dim)

    if normalize:
        fd_grad = fd_grad / (np.linalg.norm(fd_grad, axis=-1, ord=2, keepdims=True) + 1e-6)

    return fd_grad