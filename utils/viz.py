from typing import List

import torch
import torchvision

import skimage
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import trimesh


## debugging visualization functions
def plot_slice(
    model: torch.nn.Module, 
    mesh_res: int = 256, 
    plot_fn: str = None,
    other_fields: List[np.array] = None,
    levels: List[int] = [0.0],
    return_fig: bool = False,
    slice_axis: int=0,
    plt_title: str = None,
    **kwargs,
) -> np.array: # mesh_res x mesh_res

    res = mesh_res

    xx, yy = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    if slice_axis == 0:
        xy = np.concatenate(
            [
                np.ones((res, res, 1)) * (res / 2),
                xx[..., None],
                yy[..., None]
            ],
            axis=-1
        )  # (res, res, 3)
    elif slice_axis == 1:
        xy = np.concatenate(
            [
                xx[..., None],
                np.ones((res, res, 1)) * (res / 2),
                yy[..., None]
            ],
            axis=-1
        )  # (res, res, 3)
    elif slice_axis == 2:
        xy = np.concatenate(
            [
                xx[..., None],
                yy[..., None],
                np.ones((res, res, 1)) * (res / 2),
            ],
            axis=-1
        )  # (res, res, 3)
    else:
        raise ValueError
    points = xy.reshape(-1, 3)
    points = ((points / mesh_res) - 0.5) * 2
    points = torch.from_numpy(points).float().cuda()

    model = model.cuda()
    with torch.no_grad():
        distances = model.forward(points)
    distances_np = distances.detach().cpu().numpy().reshape(-1)
    distances_np = distances_np.reshape(res, res)

    # for plotting
    points = points.detach().cpu().numpy()
    slice_xy = np.delete(points, slice_axis, axis=1).reshape(res, res, 2)

    plt.figure(figsize=(5, 4))
    plt.contourf(slice_xy[..., 0], slice_xy[..., 1], distances_np)
    plt.colorbar()
    plt.contour(slice_xy[..., 0], slice_xy[..., 1], distances_np, levels, colors='red')

    if other_fields is not None:
        for i in range(len(other_fields)):
            plt.contour(slice_xy[..., 0], slice_xy[..., 1], other_fields[i], [0.], colors="yellow")

    if plt_title is not None:
        plt.title(plt_title)

    if plot_fn is not None:
        plt.savefig(plot_fn)
    if return_fig:
        return (distances_np, points), plt.gcf()
    else:
        plt.show()
        return distances_np

def plot_slices(models, mesh_res=256, plot_every=2, plot_fn=None):
    for i in range(len(models)):
        model = models[i]
        res = mesh_res

        xx, yy = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
        xy = np.concatenate([
            np.ones(
                (res, res, 1)) * (res / 2),
                 xx[..., None],
                 yy[..., None]
            ], axis=-1) # (res, res, 3)
        points = xy.reshape(-1, 3)
        points = ((points / mesh_res) - 0.5) * 2

        points = torch.from_numpy(points).float()

        model = model.cuda()
        sidx = 0
        batch_size = 8192
        
        distances_np = np.zeros((points.shape[0], 1))

        with torch.no_grad():
            eidx = min(sidx + batch_size, points.shape[1])
            points_subset = points[sidx:eidx].cuda()

            with torch.cuda.amp.autocast(enabled=True):
                distances_subset = model.forward(points_subset)

            points_subset = points_subset.cpu()

            distances_np[sidx:eidx] = distances_subset.detach().cpu().numpy()

        model = model.cpu()
        distances_np = distances_np.reshape(res, res)

        if i%plot_every == 0: 
            cp = plt.contour(distances_np, [0.], s=0.08, width=0.08)
            plt.clabel(cp, inline=True, fontsize=10, fmt={x:f"t={i}" for x in cp.levels})

    if plot_fn is not None: plt.savefig(plot_fn)
    plt.show()


def imf2mesh(imf, res=256, threshold=0.0, batch_size=10000, verbose=True,
             use_double=False,
             norm_type='res', normalize=None, # deprecated
             return_stats=False, bound=1.):
    xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
    grid = np.concatenate([
        ys[..., np.newaxis],
        xs[..., np.newaxis],
        zs[..., np.newaxis]
    ], axis=-1).astype(np.float)
    grid = (grid / float(res - 1.) - 0.5) * 2 * bound
    grid = grid.reshape(-1, 3)

    dists_lst = []
    pbar = range(0, grid.shape[0], batch_size)
    if verbose:
        pbar = tqdm.tqdm(pbar)
    for i in pbar:
        sidx, eidx = i, i + batch_size
        eidx = min(grid.shape[0], eidx)
        with torch.no_grad():
            xyz = torch.from_numpy(
                grid[sidx:eidx, :]).float().cuda().view(-1, 3)
            if use_double:
                xyz = xyz.double()
            distances = imf(xyz)
            distances = distances.cpu().numpy()
        dists_lst.append(distances.reshape(-1))

    dists = np.concatenate(
        [x.reshape(-1, 1) for x in dists_lst], axis=0).reshape(-1)
    field = dists.reshape(res, res, res)
    try:
        sp = 2. * bound / float(res - 1)
        vert, face, _, _ = skimage.measure.marching_cubes(
            field, level=threshold, spacing=(sp, sp, sp))
        vert -= bound
        print("SPACING", sp)
        new_mesh = trimesh.Trimesh(vertices=vert, faces=face)
    except ValueError as e:
        print(field.max(), field.min())
        print(e)
        new_mesh = None
    except RuntimeError as e:
        print(field.max(), field.min())
        print(e)
        new_mesh = None

    if return_stats:
        if new_mesh is not None:
            area = new_mesh.area
            vol = (field < threshold).astype(np.float).mean() * (2 * bound) ** 3
        else:
            area = 0
            vol = 0
        return new_mesh, {
            'vol': vol,
            'area': area
        }

    return new_mesh

def make_img_grid(imgs, idx=None):
    if idx is None:
        idx = [-5, -4, -3, -2, -1]
    grid_img = torchvision.utils.make_grid(imgs, nrow=5)
    return grid_img
