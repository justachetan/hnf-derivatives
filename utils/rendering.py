import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import json
import yaml
import importlib
import argparse
from typing import Callable

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

import mitsuba as mi 

from utils.utils import gradient
from utils.polyfit_utils import pfit_grad_est
from utils.fd_utils import fd_stencil_cen

from utils.sphere_tracer import Camera, sphere_trace, shade, shade_normal, shade_diffuse, shade_specular

from models.tcnn_ingp import TCNNInstantNGP
from models.triplane import Net

import matplotlib
font = {'family': 'sans-serif',
        'size': 18}
matplotlib.rc('font', **font)

mi.set_variant("cuda_ad_rgb")

shader_dict = {
    "normal": shade_normal,
    "diffuse": shade_diffuse,
    "specular": shade_specular,
    "mirror": shade
}

def load_model_ckpt(
    logdir: str,
):
    """
    load model from a log directory
    """
    
    init_ckpt_fn = os.path.join(logdir, "latest.pt")
    init_cfg_fn = os.path.join(logdir, "config", "config.yaml")

    

    with open(init_cfg_fn, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    sn_lib = importlib.import_module(cfg.models.net.type)
    model = sn_lib.Net(cfg, cfg.models.net)
    model.cuda()

    # Resume pretrained model
    init_path = init_ckpt_fn
    model.load_state_dict(torch.load(init_path)['net'], strict=True)

    return model

def get_ray_trace_img(
    scene_file: str,
    camera: Camera,
    shade_fn: Callable,
    rot: bool = True,
    invert_normal: bool = False
):
    """get ray-traced image for a the given scene file and camera

    Args:
        scene_file (str): XML file in mitsuba format
        camera (Camera): Camera object
        shade_fn (Callable): shader function
        rot (bool, optional): whether to rotate the image. Defaults to True.
        invert_normal (bool, optional): invert normals for appropriate colors. Defaults to False.

    Returns:
        _type_: np.array
    """
    

    scene = mi.load_file(scene_file)

    directions = camera.get_rays()
    dir_np = directions.numpy()
    dir_mi = mi.Vector3f(dir_np[:,0], dir_np[:,1], dir_np[:,2])

    origin = camera.origin.tolist()

    ray = mi.Ray3f(o=origin, d=dir_mi)

    itx = scene.ray_intersect(ray)

    p = itx.p
    p = torch.tensor(p).cpu()

    n = itx.n
    n = torch.tensor(n).cpu()

    if invert_normal:
        n_copy = n.clone()
        n_copy[:, 0] = 1 * n[:, 1]
        n_copy[:, 1] = -1 * n[:, 0]
        n_copy[:, 2] = -1 * n[:, 2]
        n = n_copy

    mask = itx.is_valid()
    mask = torch.tensor(mask).unsqueeze(1)
    img = shade_fn(mask, p, n, -directions)

    mask = mask.reshape((camera.h, camera.w, 1))
    img = img.reshape((camera.h, camera.w, 3))

    mask = mask.reshape((camera.h, camera.w, 1))
    img = img.reshape((camera.h, camera.w, 3))
    img = img.float()

    img = img.numpy()
    if rot: 
        img = np.rot90(img, k=-1) 

    return img


def get_sphere_tracer_img(
    model: torch.nn.Module,
    camera: Camera,
    normal_func: Callable,
    shader_fn: Callable,
    rot: bool = True,
    invert_normal: bool = False,
):

    rays = camera.get_rays()

    itx, mask = sphere_trace(lambda x: model(x), camera.origin, rays)

    itx = itx.reshape((camera.h * camera.w, 3))

    grads = normal_func(itx, model)
    itx = itx.cpu()

    if invert_normal:
        grads_copy = grads.clone()
        grads_copy[:, 0] = 1 * grads[:, 1]
        grads_copy[:, 1] = -1 * grads[:, 0]
        grads_copy[:, 2] = -1 * grads[:, 2]
        grads = grads_copy

    img = shader_fn(mask.unsqueeze(1), itx, grads, -rays)

    img = img.reshape((camera.h, camera.w, 3))
    img = img.float()

    img = img.numpy()
    if rot: img = np.rot90(img, k=-1)
    

    return img



def get_ad_gradients(
    itx: torch.Tensor,
    model: torch.nn.Module, 
):

    dset = TensorDataset(itx.detach().cpu())
    bs = 8192
    dl = DataLoader(dset, batch_size=bs, num_workers=0, shuffle=False)
    out = torch.zeros(itx.shape[0], 1)
    grads = torch.zeros(itx.shape[0], 3)

    for i, batch in enumerate(dl):

        xs = batch[0].cuda()
        xs.requires_grad_()

        ys = model(xs)

        grad_xs = gradient(ys, xs)

        sidx = i*bs
        eidx = min(itx.shape[0], (i+1)* bs)
        xs = xs.cpu()
        out[sidx:eidx] = ys.detach().cpu()
        grads[sidx:eidx] = grad_xs.detach().cpu()

    grads = grads / (grads.norm(dim=-1, p=2, keepdim=True) + 1e-6)

    return grads

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, help="Path to config")
    parser.add_argument("--out", type=str, help="Path to output directory", default=None)
    parser.add_argument("--outimg", type=str, help="Name of output image file", default="test_image.pdf")
    parser.add_argument("--shader", type=str, default=None, help="Shader to use for rendering")
    parser.add_argument("--sphere", action="store_true", help="Use sphere tracer instead of ray tracer for ground truth")

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        render_cfg = json.load(f)
    
    logdir = render_cfg["logdir"]
    init_ckpt_fn = os.path.join(logdir, "latest.pt")
    init_cfg_fn = os.path.join(logdir, "config", "config.yaml")

    

    with open(init_cfg_fn, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    sn_lib = importlib.import_module(cfg.models.net.type)
    model = sn_lib.Net(cfg, cfg.models.net)
    model.cuda()

    # Resume pretrained model
    init_path = init_ckpt_fn
    model.load_state_dict(torch.load(init_path)['net'], strict=True)

    # Load camera config
    camera_cfg = render_cfg["camera"]
    h, w = camera_cfg["res_h"], camera_cfg["res_w"]

    camera = Camera(
        origin = np.array(camera_cfg["origin"]), # -0.43 to zoom in, -0.5 to zoom out
        focal_length = camera_cfg["focal_length"], # 2.5 to zoom in, 0.5 to zoom out
        # [0, 0.15, -0.43] and 2 for head
        resolution = (h, w)
    )

    shader_type = render_cfg["shader"] if (args.shader is None) and ("shader" in render_cfg) else args.shader
    if shader_type is None: 
        print("shader not specified, setting to 'normal'")
        shader_type = "normal"

    invert_normals = False
    if "fish" in logdir and shader_type == "normal":
        invert_normals = True

    rot = not args.sphere

    # Ray Tracer Image
    scene_xml_fn = render_cfg["scene_xml"]
    rt_image = get_ray_trace_img(scene_xml_fn, camera, shader_dict[shader_type], rot, invert_normals)

    """ SPHERE TRACER """

    # AD image
    ad_img = get_sphere_tracer_img(
        model,
        camera,
        get_ad_gradients,
        shader_dict[shader_type],
        rot,
        invert_normals
    )

    # polynomial-fitting image

    pfit_sigma =  render_cfg["pfit"]["sigma"]
    pfit_n_walks = render_cfg["pfit"]["n_walks"]
    # NOTE: If encountering memory issues, set doall=False and play with batch_size
    pfit_est = lambda x, model: pfit_grad_est(x, model, n_walks=pfit_n_walks, sigma=pfit_sigma, doall=True, bs=8192, norm=True).detach().cpu()

    pfit_img = get_sphere_tracer_img(
        model,
        camera,
        pfit_est,
        shader_dict[shader_type],
        rot,
        invert_normals
    )

    # FD image
    delx = render_cfg["fd"]["delx"]
    fd_est = lambda x, model: fd_stencil_cen(x, delx, model, normalize=True).detach().cpu()  

    fd_img = get_sphere_tracer_img(
        model,
        camera,
        fd_est,
        shader_dict[shader_type],
        rot,
        invert_normals
    )  



    f, axarr = plt.subplots(1, 4, figsize=(24, 6))
    axarr[0].imshow(rt_image)
    axarr[0].axis("off")
    axarr[0].set_title("Mesh Normals" if not args.sphere else "Ground Truth")

    axarr[1].imshow(ad_img)
    axarr[1].axis("off")
    axarr[1].set_title("Autodiff Gradients")

    axarr[2].imshow(fd_img)
    axarr[2].axis("off")
    axarr[2].set_title("FD Gradients")


    axarr[3].imshow(pfit_img)
    axarr[3].axis("off")
    axarr[3].set_title("Ours")


    plt.tight_layout()
    if args.out is None:
        out_dir = os.path.abspath(os.path.join(args.cfg, os.pardir))
    else:
        out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, args.outimg))

    np.savez(os.path.join(out_dir, "image_arrays.npz"), rt=rt_image, ad=ad_img, fd=fd_img, mc=pfit_img)


if __name__ == "__main__":
    main()

