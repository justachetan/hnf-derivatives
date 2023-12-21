import os
import yaml
import time
import argparse
import importlib
import os.path as osp
from utils import AverageMeter, dict2namespace, update_cfgdict_hparam_lst, flat_dict
from torch.backends import cudnn
from utils import SummaryWriter


def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Repository entry')
    parser.add_argument('config', type=str,
                        help='The configuration file.')
    parser.add_argument('--log_dir', type=str, default="logs/",
                        help='The logging directory.')

    # distributed training
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # Resume:
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")
    # Resume if there is a lastest.pt, otherwise don't fail
    parser.add_argument('--soft_resume', default=False, action='store_true')

    # Test run:
    parser.add_argument('--test_run', default=False, action='store_true')
    parser.add_argument('--no_run_time_postfix',
                        default=False, action='store_true')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    parser.add_argument('--no_append_hparams_to_name',
                        default=False, action='store_true')
    args = parser.parse_args()

    # parse config file
    with open(args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    config_dict, hparam_str = update_cfgdict_hparam_lst(
        config_dict, args.hparams, strict=True)
    # config, hparam_str = update_cfg_hparam_lst(
    #     config, args.hparams, strict=True)

    # Currently save dir and log_dir are the same
    # if not hasattr(config, "log_dir"):
    if "log_dir" not in config_dict:
        #  Create log_name
        if args.test_run:
            cfg_file_name = "test"
        else:
            cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]

        if not args.no_run_time_postfix:
            run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
        else:
            run_time = ""
        post_fix = run_time
        if not args.no_append_hparams_to_name:
            post_fix = hparam_str + run_time

        os.makedirs(args.log_dir, exist_ok=True)
        config_dict["log_dir"] = "%s/%s_%s" % (args.log_dir, cfg_file_name, post_fix)
        config_dict["log_name"] = "%s/%s_%s" % (args.log_dir, cfg_file_name, post_fix)
        config_dict["log_name_small"] = "logs_small/%s_%s" % (cfg_file_name, post_fix)
        config_dict["save_dir"] = "%s/%s_%s" % (args.log_dir, cfg_file_name, post_fix)

        os.makedirs(osp.join(config_dict["log_dir"], 'config'), exist_ok=True)
        out_yaml_file = osp.join(config_dict["log_dir"], "config", "config.yaml")
        with open(out_yaml_file, "w") as outf:
            yaml.dump(dict2namespace(config_dict), outf)
        
    return args, config_dict


def main_worker(cfgdict, args):
    cfg = dict2namespace(cfgdict)
    # basic setup
    cudnn.benchmark = True

    # Customized summary writer that write another copy of scalars
    # into a small log_dir (so that it's easier to load for tensorboard)
    writer = SummaryWriter(
        log_dir=cfg.log_name,
        small_log_dir=getattr(cfg, "log_name_small", None))
    writer.add_hparams(flat_dict(cfgdict), {"loss": 0.})
    # writer.add_text("config", json.dumps(flat_dict(cfgdict), indent=2))
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)

    start_epoch = 0
    if args.resume or args.soft_resume:
        if args.pretrained is not None:
            only_model = True if cfg.trainer.get("reg", False) or cfg.trainer.get("is_gp", False) else False
            # import pdb
            # pdb.set_trace()
            start_epoch = trainer.resume(args.pretrained, only_model=only_model)
        else:
            latest = osp.join(cfg.log_dir, "latest.pt")
            if osp.isfile(latest) or not args.soft_resume:
                # If the file doesn't exist, and soft resume is not specified
                # then it will throw errors.
                start_epoch = trainer.resume(latest)

    # If test run, go through the validation loop first
    if args.test_run:
        trainer.save(epoch=-1, step=-1)
        test_loader = trainer.get_dataloader("test")
        val_info = trainer.validate(test_loader, epoch=-1)
        trainer.log_val(val_info, writer=writer, epoch=-1)

    # main training loop
    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs))
    step = 0
    duration_meter = AverageMeter("Duration")
    updatetime_meter = AverageMeter("Update")
    loader_meter = AverageMeter("Loader time")
    logtime_meter = AverageMeter("Log time")
    mctime_meter = AverageMeter("MC Time")
    mcfwdtime_meter = AverageMeter("MC Fwd Time")
    ffctime_meter = AverageMeter("FFC Time")

    # updatetime = 0
    # loadertime = 0
    # mc_time = 0
    # mc_fwd_time = 0
    # ffc_time = 0

    for epoch in range(start_epoch, cfg.trainer.epochs):
        train_loader, time_capsule = trainer.get_dataloader("train", epoch=epoch)
        test_loader = trainer.get_dataloader("test", epoch=epoch)

        pre_update_info = trainer.before_update(dataset=train_loader.dataset, epoch=epoch)
        # print("pre_update_info", type(pre_update_info))

        # train for one epoch
        iter_start = time.time()
        loader_start = time.time()
        for bidx, data in enumerate(train_loader):
            loader_duration = time.time() - loader_start
            loader_meter.update(loader_duration)
            # loadertime += loader_duration

            start_time = time.time()
            step = bidx + len(train_loader) * epoch + 1
            logs_info = trainer.update(data, epoch=epoch)
            duration = time.time() - start_time
            updatetime_meter.update(duration)
            # updatetime += duration

            logtime_start = time.time()
            if step % int(cfg.log_every_n_steps) == 0 or step == 1:
                # print("step", step)
                
                print("Epoch %d Batch [%2d/%2d] Time/Iter: Train[%3.2fs] "
                      "Update[%3.2fs] Log[%3.2fs] Load[%3.2fs] Loss %2.5f"
                      % (epoch, bidx, len(train_loader),
                         duration_meter.avg,
                         updatetime_meter.avg, logtime_meter.avg,
                         loader_meter.avg, logs_info['loss']))
                # writer.add_hparams(flat_dict(cfgdict), {"loss": logs_info["loss"]})
                visualize = step % int(cfg.viz_every_n_steps) == 0 or step == 1
                # save_mesh = step % int(cfg.save_mesh_every_n_steps) == 0 or step == 1
                # assert int(cfg.save_mesh_every_n_steps) >= int(cfg.viz_every_n_steps)
                # assert (int(cfg.save_mesh_every_n_steps) % int(cfg.viz_every_n_steps) == 0)
                
                logs_info["pre_update_info"] = pre_update_info

                if bidx == 0 and time_capsule is not None: # epoch has just started so these times need to be logged
                    mctime_meter.update(time_capsule["mc_time"])
                    logs_info["mc_time"] = mctime_meter.avg
                    mcfwdtime_meter.update(time_capsule["mc_fwd_time"])
                    logs_info["mc_fwd_time"] = mcfwdtime_meter.avg
                    ffctime_meter.update(time_capsule["ffc_time"])
                    logs_info["ffc_time"] = ffctime_meter.avg

                
                if (bidx + 1) == len(train_loader): # epoch is about to end so these times need to be logged
                    logs_info["opt_time"] = updatetime_meter.avg
                    logs_info["newsdfcalc_time"] = loader_meter.avg

                # print("visualize", visualize)
                trainer.log_train(
                    logs_info, data,
                    writer=writer, epoch=epoch, step=step, 
                    visualize=visualize)
            logtime_duration = time.time() - logtime_start
            logtime_meter.update(logtime_duration)
            iter_duration = time.time() - iter_start
            duration_meter.update(iter_duration)

            # Reset loader time
            loader_start = time.time()

        # Save first so that even if the visualization bugged,
        # we still have something
        if (epoch + 1) % int(cfg.save_every_n_epochs) == 0 and \
                int(cfg.save_every_n_epochs) > 0:
            trainer.save(epoch=epoch, step=step)

        if (epoch + 1) % int(cfg.val_every_n_epochs) == 0 and \
                int(cfg.val_every_n_epochs) > 0:
            val_info = trainer.validate(test_loader, epoch=epoch)
            trainer.log_val(val_info, writer=writer, epoch=epoch)

        # Signal the trainer to cleanup now that an epoch has ended
        trainer.epoch_end(epoch, writer=writer)

    # Final round of validation
    val_info = trainer.validate(test_loader, epoch=epoch + 1)
    trainer.log_val(val_info, writer=writer, epoch=epoch + 1)
    trainer.save(epoch=epoch, step=step)
    writer.close()


if __name__ == '__main__':
    # command line args
    args, cfgdict = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfgdict)

    main_worker(cfgdict, args)