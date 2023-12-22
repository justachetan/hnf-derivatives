import os
import torch 
import numpy as np 
import cv2 as cv

eps = 1e-3

spp = 2

abs_path_prefix = os.path.dirname(__file__)

def normalize(v):
    return v / (torch.norm(v, dim=1, keepdim=True) + 1e-6)

def reflect(v, n):
    return 2 * torch.mul(n, torch.sum(n * v, dim=1, keepdim=True)) - v

def sdf(x):
    return torch.norm(x, dim=1) - 0.54

def sdf_gradient(x):
    return normalize(x)

class Camera:
    def __init__(self, origin, focal_length, resolution):
        self.origin = origin
        self.focal_length = focal_length
        self.h, self.w = resolution
        self.aspect_ratio = resolution[1] / resolution[0]
    
    def get_ray(self, i, j):
        return torch.tensor([
            -(i/self.h - 0.5 - (0.5/self.h)), 
            self.aspect_ratio * (0.5 - j / self.w) + (0.5/self.w), self.focal_length
        ])
    
    def get_rays(self):
        rays = torch.zeros((self.h * self.w, 3))
        for i in range(self.h):
            for j in range(self.w):
                rays[i * self.w + j, :] = self.get_ray(i, j)
        return normalize(rays)

# everything is in torch tensors
def sphere_trace(sdf, origin, dir, max_steps=100):
    num = dir.shape[0]
    bs = 2**18
    
    p = torch.tensor(origin).repeat(num, 1).cuda()
    dir = dir.cuda()

    mask = torch.zeros(num, dtype=torch.bool).cuda()
    for i in range(max_steps):
            
            
            for idx in range(0, num, bs):
                # p_batch = batch[0].cuda()
                # p_batch = batch[0]
                # dir_batch = batch[1]

                sidx = idx
                eidx = min(idx + bs, p.shape[0])

                p_batch = p[sidx:eidx]
                dir_batch = dir[sidx:eidx]

                dist = sdf(p_batch) # bs x 1

                p_batch += dist * dir_batch

                is_itx = (torch.abs(dist) < eps).squeeze(1).bool()
                in_range = torch.all(torch.abs(p_batch)<1, dim=1, keepdim=False).bool()
          
                mask[sidx:eidx] = torch.logical_and(is_itx, in_range).bool()
                
                p[sidx:eidx] = p_batch.detach()
                p_batch = p_batch.cpu()
                dir_batch = dir_batch.cpu()
                dist = dist.cpu()
            
            if mask.all():
                print("all converged")
                break

    return p, mask.cpu()

def shade_diffuse_once(mask, itx, normal, wi):
    wi = normalize(wi)
    
    r = torch.randn(3)
    r /= torch.norm(r)
    
    wo = normal + r
    wo = normalize(wo)

    u = torch.atan2(wo[:, 0], wo[:, 2]) / (2 * np.pi)
    v = 0.5 - torch.asin(wo[:, 1]) / np.pi

    env = cv.imread(os.path.join(abs_path_prefix, "rendering_assets/env6.hdr"))
    env = torch.tensor(env)

    # Convert UV coordinates to pixel coordinates
    height, width = env.shape[0], env.shape[1]
    x = (u * (width - 1)) % width
    y = (v * (height - 1)) % height
    x = x.long()
    y = y.long()

    # Get the environment map colors at the corresponding pixel coordinates
    colors = env[y, x]
    colors = colors.float() / 255

    print(wo.shape)
    print(normal.shape)

    cos = wo * normal 
    print(cos.shape)
    print((cos*colors).shape)


    diffuse = torch.max(torch.zeros_like(wo[:, 0]), (cos * colors)[:,]).unsqueeze(1)
    
    return mask * diffuse 

def shade_diffuse(mask, itx, normal, wi):
    light = torch.tensor([0., 0., -20.])
    wi = normalize(wi)
    wo = normalize(light - itx)
    
    diffuse = torch.max(torch.zeros_like(wo[:, 0]), torch.sum(wo * normal, dim=1)).unsqueeze(1)
    
    return (1.5 * diffuse * mask * torch.tensor([.5, .5, .5])).detach() # red

def shade_specular(mask, itx, normal, wi):
    light = torch.Tensor([-10, 10, 2])
    wi = normalize(wi)
    wo = normalize(light - itx)
    r = 2 * torch.mul(normal, torch.sum(normal * wo, dim=1, keepdim=True)) - wo
    r = normalize(r)
    specular = torch.max(torch.zeros_like(wi[:, 0]), -torch.sum(wi * r, dim=1)).unsqueeze(1)
    
    return (mask * specular).detach()

def shade_mirror(mask, itx, normal, wi):
    env = cv.imread(os.path.join(abs_path_prefix, "rendering_assets/env6.hdr"))
    env = torch.tensor(env)
    
    wi = normalize(wi)
    wo = reflect(wi, normal)
    wo = normalize(wo)
    
    u = torch.atan2(wo[:, 0], wo[:, 2]) / (2 * np.pi)
    v = 0.5 - torch.asin(wo[:, 1]) / np.pi

    # Convert UV coordinates to pixel coordinates
    height, width = env.shape[0], env.shape[1]
    x = (u * (width - 1)) % width
    y = (v * (height - 1)) % height
    x = x.long()
    y = y.long()


    # Get the environment map colors at the corresponding pixel coordinates
    colors = env[y, x]
    colors = colors.float() / 255

    return mask * colors

def shade_normal(mask, itx, normal, wi):
    if normal is None:
        n = normalize(sdf_gradient(itx))
        n_copy = n.clone()
        n_copy[:, 0] = 1 * n[:, 1]
        n_copy[:, 1] = -1 * n[:, 0]
        n_copy[:, 2] = -1 * n[:, 2]
        normal = n_copy
    colors = (normal + 1) / 2
    return mask * colors

def shade(mask, itx, normal, wi):
    if normal is None:
        normal = normalize(sdf_gradient(itx))

    return shade_mirror(mask, itx, normal, wi) + \
           1.0 * shade_specular(mask, itx, normal, wi) ** 10

def background(mask, img):
    
    background = cv.imread(os.path.join(abs_path_prefix, 'rendering_assets/env6.hdr'))
    backgronud = cv.GaussianBlur(background, (21, 21), 0)
    
    cropped = backgronud[
        :img.shape[0],
        background.shape[1]//2-img.shape[1]//2:background.shape[1]//2+img.shape[1]//2, 
        :
    ]
    cropped = torch.tensor(cropped)
    cropped = (~mask) * cropped
    cropped = cropped.float() / 255
    
    return img + cropped

def render(camera):
    rays = camera.get_rays()

    itx, mask = sphere_trace(sdf, camera.origin, rays)
    itx = itx.reshape((camera.h * camera.w, 3))

    wi = normalize(-rays)
    normal = normalize(sdf_gradient(itx))

    img = shade(mask, itx, normal, wi)
    
    mask = mask.reshape((camera.h, camera.w, 1))
    img = img.reshape((camera.h, camera.w, 3))
    img = background(mask, img)
    
    return img
