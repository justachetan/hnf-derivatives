import os
import os.path as osp

import importlib
from functools import partial

from pprint import pprint

import trimesh

import torch

import igl
import numpy as np

from trainers.base_trainer import BaseTrainer, grad_norm

from utils.polyfit_utils import pfit_grad_est
from utils.fd_utils import fd_stencil_cen
from utils.utils import gradient, get_zero_pts, seed_everything
from utils.eval_utils import get_on_surf_pts, l2_err_sfn, ang_err_sfn, perc_ang_acc_below_k, rec_rel_error, get_ad_stats_fo, get_ad_stats_so
from utils.viz import imf2mesh, plot_slice
from utils.mesh_metrics import compute_all_mesh_metrics_with_opcl

def mape_loss(pred, target, reduction='mean'):
    # pred, target: [B, 1], torch tenspr
    difference = (pred - target).abs()
    scale = 1 / (target.abs() + 1e-2)
    loss = difference * scale

    if reduction == 'mean':
        loss = loss.mean()

    return loss


class Trainer(BaseTrainer):

    def __init__(self, cfg, args, **kwargs):
        super().__init__(cfg, args, **kwargs)
        self.cfg = cfg
        self.args = args
        seed_everything(self.cfg.trainer.get("seed", 666))

        # The networks
        sn_lib = importlib.import_module(cfg.models.net.type)
        self.net = sn_lib.Net(cfg, cfg.models.net)
        self.net.cuda()

        self.ext_mesh = None
        self.prev_net = None
        if self.cfg.trainer.get("reg", False):
            sn_lib = importlib.import_module(cfg.models.net.type)
            self.prev_net = sn_lib.Net(cfg, cfg.models.net)
            self.prev_net.cuda()

        self.multi_gpu = cfg.trainer.get("multi_gpu", False)
        if self.multi_gpu:
            self.multi_gpu_wrapper(torch.nn.DataParallel)

        
        print("[SDFTrainer] Net:")
        print(self.net)
        print("%.5fM Parameters" %
              (sum([p.numel() for p in self.net.parameters()]) * 1e-6))

        # The optimizer
        cfg_opt = self.cfg.trainer.opt
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            lr=cfg_opt.lr,
            betas=(cfg_opt.get("beta1", 0.9),
                   cfg_opt.get("beta2", 0.99)),
            eps=cfg_opt.get("eps", 1e-15))
        self.sch = torch.optim.lr_scheduler.StepLR(
            self.opt,
            step_size=cfg_opt.get("step_size", 10),
            gamma=cfg_opt.get("gamma", 0.1))

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "meshes"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "val"), exist_ok=True)

        # TODO: make sure the random-seed here is set correctly
        self.data_lib = importlib.import_module(self.cfg.data.type)
        self.loaders = None

    def get_dataloader(self, split, **kwargs):
        if self.loaders is None:
            self.loaders = self.data_lib.get_data_loaders(self.cfg.data)
        if split == "train":
            return self.loaders["train_loader"], None
        elif split == "test":
            return self.loaders["test_loader"]
        else:
            raise ValueError

    def update(self, data, *args, **kwargs):
        self.net.train()
        self.opt.zero_grad(set_to_none=True)
        xyz = data["xyz"].cuda()
        sdf = data["sdf"].cuda()
        xyz = xyz.view(-1, xyz.size(-1))


        to_reg = self.cfg.trainer.get("reg", False)
        if to_reg:
            xyz.requires_grad_(True)
        out = self.net(xyz)

        loss_type = self.cfg.trainer.get("loss_type", "mape")
        loss = 0

        if to_reg: 
            if self.cfg.trainer.polyfit_reg.get("is_ft", True):
                with torch.no_grad():
                    sdf = self.prev_net(xyz).detach().reshape(*sdf.shape)
        if loss_type == "mape":
            loss = mape_loss(out, sdf)
        elif loss_type == "mse":
            loss = ((out - sdf) ** 2).mean()
        else:
            raise ValueError
        
        grad_loss = 0
        
        if to_reg:
            grad_reg_cfg = self.cfg.trainer.polyfit_reg.grad

            g_sigma = grad_reg_cfg.get("sigma", 1e-3)
            g_n_walks = grad_reg_cfg.get("n_walks", 256)
            g_doall = grad_reg_cfg.get("doall", True)
            grad_loss_wt = grad_reg_cfg.get("wt", 1e-3)
            grad_from_ep = grad_reg_cfg.get("from_epoch", 100)
            g_del_x = grad_reg_cfg.get("g_del_x", 2./32.)

            if grad_loss_wt > 0 and kwargs["epoch"] >= grad_from_ep:
                
                smooth_grad = None
                if grad_reg_cfg.get("mode", "polyfit") == "mesh":
                    if self.ext_mesh is None:
                        if not self.cfg.trainer.polyfit_reg.get("is_ft", True):
                            self.ext_mesh = trimesh.load(self.cfg.data.mesh_path, process=True)
                        else:
                            v, f, _ = get_zero_pts(self.prev_net, mesh_res=512)
                            self.ext_mesh = trimesh.Trimesh(v, f, validate=True)

                    v = np.array(self.ext_mesh.vertices)
                    f = np.array(self.ext_mesh.faces)
                    sd, fidx, closest_vert, mesh_grad = igl.signed_distance(xyz.detach().cpu().numpy(), v, f, return_normals=True)
                    mesh_grad /= (np.linalg.norm(mesh_grad, axis=-1, ord=2)[..., None] + 1e-16)
                    smooth_grad = torch.from_numpy(mesh_grad).cuda().float()

                elif grad_reg_cfg.get("mode", "polyfit") == "fd":

                    with torch.no_grad():
                        smooth_grad = fd_stencil_cen(x=xyz, delx=g_del_x, model= self.prev_net, normalize=True)
                
                else:
                    with torch.no_grad():
                        smooth_grad = pfit_grad_est(
                            xyz, self.prev_net, sigma=g_sigma, n_walks=g_n_walks, doall=g_doall, norm=True, eps=1e-16,
                        )
                ad_grad = gradient(out, xyz)
                ad_grad = ad_grad / (ad_grad.norm(dim=-1, keepdim=True) + 1e-16)

                grad_loss = torch.nn.functional.mse_loss(smooth_grad, ad_grad)
                loss += grad_loss_wt * grad_loss

    
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            acc = ((out > 0) == (sdf > 0)).float().mean()
            gnorm = grad_norm(self.net.parameters())

        if self.cfg.trainer.get("clip_norm", False):
            clip_amount = self.cfg.trainer.get("clip_norm", 0.01)
            clip_norm = torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), clip_amount)
        else:
            clip_norm = gnorm

        return {
            'loss': loss.detach().cpu().item(),
            'grad_loss': grad_loss.detach().cpu().item() if torch.is_tensor(grad_loss) else grad_loss,
            'scalar/loss': loss.detach().cpu().item(),
            'scalar/grad_loss': grad_loss.detach().cpu().item() if torch.is_tensor(grad_loss) else grad_loss,
            'scalar/gnorm': clip_norm.detach().cpu().item(),
            'scalar/acc': acc.detach().cpu().item(),
        }

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        super().log_train(
            train_info, train_data, writer=writer,
            step=step, epoch=epoch, visualize=visualize, **kwargs)

        writer_step = step if step is not None else epoch
        if visualize:
            with torch.no_grad():
                print("MC at %s" % step)
                res = int(self.cfg.trainer.get("mc_res", 128))
                thr = float(self.cfg.trainer.get("mc_thr", 0.))
                bound = float(self.cfg.trainer.get("mc_bound", 1.))
                mc_bs = int(self.cfg.trainer.get("mc_bs", 100000))
                print("   config:res=%d thr=%s bound=%s" % (res, thr, bound))

                mesh, mesh_stat = imf2mesh(
                    lambda x:  self.net(x),
                    return_stats=True,
                    res=res, threshold=thr, bound=bound, batch_size=mc_bs,
                    normalize=True, norm_type='res'
                )
                level_out_dir = osp.join(self.cfg.save_dir, "meshes")
                os.makedirs(level_out_dir, exist_ok=True)
                if mesh is not None:
                    save_name = "mesh_%diters.obj" % writer_step
                    mesh.export(osp.join(level_out_dir, save_name))
                    for k, v in mesh_stat.items():
                        writer.add_scalar('vis/mesh/%s' % k, v, writer_step)


                _, fig0 = plot_slice(self.net, res,
                                     return_fig=True, slice_axis=0)
                writer.add_figure(
                    "fig/center_slice_0", fig0,
                    global_step=writer_step, close=True)
                _, fig1 = plot_slice(self.net, res,
                                     return_fig=True, slice_axis=1)
                writer.add_figure(
                    "fig/center_slice_1", fig1,
                    global_step=writer_step, close=True)
                _, fig2 = plot_slice(self.net, res,
                                     return_fig=True, slice_axis=1)
                writer.add_figure(
                    "fig/center_slice_2", fig2,
                    global_step=writer_step, close=True)


    def validate(self, test_loader, epoch, *args, **kwargs):
        writer_step = epoch
        print("Validation at ecpoh: %d" % epoch)

        val_results = {}
        with torch.no_grad():
            for data in test_loader:
                break
            metric_npnts = int(self.cfg.get("metric_npnts", 300000))
            if 'mesh' in data:
                gtr_mesh = data['mesh'][0]
                print("Gtr mesh:",
                      gtr_mesh.vertices.max(),
                      gtr_mesh.vertices.mean(),
                      gtr_mesh.vertices.min(),
                      )
                pcl, fidx = trimesh.sample.sample_surface(
                    gtr_mesh, metric_npnts)
                sfn = gtr_mesh.face_normals[fidx]
            elif 'sfn' in data and 'pcl' in data:
                sfn = data['sfn']
                pcl = data['pcl']
            else:
                gtr_mesh = test_loader.dataset.mesh
                pcl, fidx = trimesh.sample.sample_surface(
                    gtr_mesh, metric_npnts)
                sfn = gtr_mesh.face_normals[fidx]

            res = int(self.cfg.get("mc_res", 256))
            thr = float(self.cfg.get("mc_thr", 0.))
            bound = float(self.cfg.get("mc_bound", 0.5))
            batch_size = int(self.cfg.get("mc_bs", 100000))
            print("   config:res=%d thr=%s bound=%s bs=%d"
                  % (res, thr, bound, batch_size))

            mesh, mesh_stat = imf2mesh(
                lambda x: self.net(x),
                return_stats=True,
                res=res, threshold=thr, bound=bound, batch_size=batch_size,
                normalize=True, norm_type='res')
            level_out_dir = osp.join(self.cfg.save_dir, "val")
            os.makedirs(level_out_dir, exist_ok=True)
            if mesh is not None:
                print("Export mesh")
                print("           ",
                      mesh.vertices.max(),
                      mesh.vertices.mean(),
                      mesh.vertices.min())
                save_name = "mesh_%diters.obj" % (writer_step)
                mesh.export(osp.join(level_out_dir, save_name))
                for k, v in mesh_stat.items():
                    val_results[
                        'scalar/val/mesh_stats_%s' % k] = v

                print("Compute validation metrics at level")
                points1, fidx = trimesh.sample.sample_surface(
                    mesh, metric_npnts)
                normals1 = mesh.face_normals[fidx]

                level_val_res = compute_all_mesh_metrics_with_opcl(
                    points1, pcl, normals1, sfn)
                pprint(level_val_res)
                for k, v in level_val_res.items():
                    val_results['scalar/val/mesh_metrics_%s' % k] = v

        if self.cfg.trainer.get("reg", False):
            mesh = test_loader.dataset.mesh
            onsurf_pts, gt_sfn = get_on_surf_pts(mesh, mode="sfn")
            onsurf_pts = torch.from_numpy(onsurf_pts).cuda()
            ad_normals, ad_stats = get_ad_stats_fo(onsurf_pts, self.net, err_funcs=[
                l2_err_sfn, ang_err_sfn, partial(perc_ang_acc_below_k, k=1), partial(perc_ang_acc_below_k, k=2)
            ], gt_normals=gt_sfn)
            onsurf_pts = onsurf_pts.detach().cpu()

            onsurf_pts, gt_curv = get_on_surf_pts(mesh, mode="curv")
            onsurf_pts = torch.from_numpy(onsurf_pts).cuda()
            ad_curv, ad_stats_curv = get_ad_stats_so(onsurf_pts, self.net, err_funcs=[
                rec_rel_error
            ], gt_curv=gt_curv)

            val_results['scalar/val/sfn_l2'] = ad_stats[0]
            val_results['scalar/val/sfn_ang'] = ad_stats[1]
            val_results['scalar/val/sfn_aa@1'] = ad_stats[2]
            val_results['scalar/val/sfn_aa@2'] = ad_stats[3]
            val_results['scalar/val/curv_relerr'] = ad_stats_curv[0]

        pprint(val_results)
        return val_results

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt': self.opt.state_dict(),
            'net': self.net.state_dict(),
            'sch': self.sch.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if self.sch is not None:
            d['sch'] = self.sch.state_dict()
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

        save_name = "latest.pt"
        path = os.path.join(self.cfg.save_dir, save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, only_model=False, **kwargs):
        ckpt = torch.load(path)
        model_key = "net" if "net" in ckpt else "model"
        self.net.load_state_dict(ckpt[model_key], strict=strict)
        start_epoch = ckpt['epoch']

        if self.cfg.trainer.get("reg", False):
            self.prev_net.load_state_dict(ckpt[model_key], strict=strict)
            self.prev_net.cuda()

        if not only_model:
            if "opt" in ckpt: self.opt.load_state_dict(ckpt['opt'])
            
            if self.sch is not None:
                if "sch" in ckpt: self.sch.load_state_dict(ckpt['sch'])

        return start_epoch

    def multi_gpu_wrapper(self, wrapper):
        self.net = wrapper(self.net)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.sch is not None:
            self.sch.step(epoch)
            if writer is not None:
                writer.add_scalar(
                    'lr', self.sch.get_last_lr()[0], epoch)