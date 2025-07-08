import os
import datetime
from collections import OrderedDict
import wandb
from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from lib.utils.box_ops import map_boxes_back_batch, clip_box_batch, batch_bbox_voting
from lib.utils.pseudo_label_save import write_to_txt

from lib.utils.misc import get_world_size


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False, cfg=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard and wandb
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval
                self.wandb_writer = WandbWriter(settings.project_path[6:], cfg, tensorboard_writer_dir,
                                                cur_train_samples, interval, )

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        # init pl_box dict
        pl_box_dict = {}

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # 0703
            loader_type = loader.name
            # forward pass
            # 0708
            if not self.use_amp:
                # 0712
                if loader_type != 'train_extreme':
                    loss, stats = self.actor(data, loader_type)
                else:
                    loss, stats, out_dict = self.actor(data, loader_type)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # 0719
            # collect and handle all the pseudo label one epoch
            if loader_type == 'train_extreme':
                data['search_box_extract'] = data['search_box_extract'].permute(1, 0)
                data['search_resize_factors'] = data['search_resize_factors'].permute(1, 0)
                data['search_original_shape'] = data['search_original_shape'].permute(1, 0)

                if i == 1:
                    pl_box_dict['pred_boxes'] = out_dict['pred_boxes']
                    pl_box_dict['topk_score'] = out_dict['topk_score']
                    pl_box_dict['img_paths'] = data['search_frame_paths'][0]

                    pl_box_dict['search_box_extract'] = data['search_box_extract']
                    pl_box_dict['search_resize_factors'] = data['search_resize_factors']
                    pl_box_dict['search_original_shape'] = data['search_original_shape']
                    pl_box_dict['save_dir'] = data['settings'].save_dir
                else:
                    pl_box_dict['pred_boxes'] = torch.cat((pl_box_dict['pred_boxes'], out_dict['pred_boxes']), dim=0)
                    pl_box_dict['topk_score'] = torch.cat((pl_box_dict['topk_score'], out_dict['topk_score']), dim=0)
                    pl_box_dict['img_paths'] = pl_box_dict['img_paths'] + data['search_frame_paths'][0]

                    pl_box_dict['search_box_extract'] = torch.cat(
                        (pl_box_dict['search_box_extract'], data['search_box_extract']), dim=0)
                    pl_box_dict['search_resize_factors'] = torch.cat(
                        (pl_box_dict['search_resize_factors'], data['search_resize_factors']), dim=0)
                    pl_box_dict['search_original_shape'] = torch.cat(
                        (pl_box_dict['search_original_shape'], data['search_original_shape']), dim=0)

            # backward pass and update weights
            if loader.training and "extreme" not in loader.name:
                # #检查参数是否冻结
                # for name, param in self.actor.net.named_parameters():
                #     # 检查参数是否需要梯度更新
                #     if param.requires_grad:
                #         # 如果参数名中不包含 'pgn_module'，则报错并终止训练
                #         if "pgn_module" not in name:
                #             raise ValueError(
                #                 f"Error: Parameter '{name}' requires gradient but is not part of 'pgn_module'. Training terminated.")

                # print("------------------------------------")
                # print("Check Successfully!!!")
                # print("------------------------------------")


                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)



            # update wandb status
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                self.wandb_writer.write_log(self.stats, self.epoch)

        # 0719
        if loader.name == 'train_extreme':
            t1 = time.time()
            device = pl_box_dict['pred_boxes'].device

            #伪标签投票
            bbox_optimize = batch_bbox_voting(pl_box_dict['pred_boxes'], pl_box_dict['topk_score']).to(device)


            pl_box_dict['mapback_pred_boxes'] = torch.tensor(
                bbox_optimize.squeeze() * 256 / pl_box_dict['search_resize_factors'])

            pl_box_dict['mapback_pred_boxes'] = clip_box_batch(
                map_boxes_back_batch(pl_box_dict['search_box_extract'], pl_box_dict['mapback_pred_boxes'],
                                     pl_box_dict['search_resize_factors']),
                pl_box_dict['search_original_shape'],
                margin=10)

            #更新伪标签
            for i in range(len(pl_box_dict['img_paths'])):

                path_list = pl_box_dict['img_paths'][i].split('/')

                # pl_save_dir = os.path.join(pl_box_dict['save_dir'], 'pseudo_label', path_list[-4], path_list[-2])
                pl_save_dir = os.path.join('/mnt/ssd3/wzq/UMDATrack', 'pseudo_label', path_list[-4], path_list[-2])    
                img_id = int(path_list[-1].split('.')[0])
                txt_path = os.path.join(pl_save_dir, 'pl.txt')
                if not os.path.exists(pl_save_dir):
                    raise FileNotFoundError(f"The {pl_save_dir} does not exist!")

                write_to_txt(txt_path, img_id, pl_box_dict['mapback_pred_boxes'][i])

            t2 = time.time()

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))
        if loader.name == 'train_extreme':
            print("produce pl label cost:{:.1f} mins".format((t2 - t1) / 60))

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            # if loader.name == 'train_extreme':
            #     continue

            # 0708
            if self.epoch >= loader.epoch_begin and self.epoch <= loader.epoch_end and \
                    self.epoch % loader.epoch_interval == 0:

                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (
            self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)


            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if name == 'Coord': continue
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            # 0708
            if loader.training and "extreme" not in loader.name:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    if self.stats[loader.name] is None:
                        continue
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
