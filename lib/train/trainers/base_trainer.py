import os
import glob
import torch
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            '''2021.1.4 New function: specify checkpoint dir'''
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print("checkpoints will be saved to %s" % self._checkpoint_dir)

            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    print("Training with multiple GPUs. checkpoints directory doesn't exist. "
                          "Create checkpoints directory")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    # 0710
    def update_net_extreme_params(self, keep_rate=0.9996):
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
        net_extreme = self.actor.net_extreme.module if multigpu.is_multi_gpu(
            self.actor.net_extreme) else self.actor.net_extreme

        net_extreme_dict = OrderedDict()
        net_dict = net.state_dict()

        for key, value in net_extreme.state_dict().items():
            if key in net_dict.keys():
                net_extreme_dict[key] = (
                        net_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                print("error!")

        net_extreme.load_state_dict(net_extreme_dict)


    def move_pgn_to_teacher(self):
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
        net_extreme = self.actor.net_extreme.module if multigpu.is_multi_gpu(
            self.actor.net_extreme) else self.actor.net_extreme

        net_extreme_dict = OrderedDict()
        net_dict = net.state_dict()

        for key, value in net_extreme.state_dict().items():
            if key in net_dict.keys():
                if 'pgn_module' in key:
                    net_extreme_dict[key] = net_dict[key]
                else:
                    net_extreme_dict[key] = value
            else:
                print("error!")

        net_extreme.load_state_dict(net_extreme_dict)


    def move_weight_to_teacher(self, keep_rate=0.9996):
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
        net_extreme = self.actor.net_extreme.module if multigpu.is_multi_gpu(
            self.actor.net_extreme) else self.actor.net_extreme

        net_extreme_dict = OrderedDict()
        net_dict = net.state_dict()

        for key, value in net_extreme.state_dict().items():
            if key in net_dict.keys():
                if 'pgn_module' in key or 'head' in key:
                    net_extreme_dict[key] = (
                        net_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
                else:
                    net_extreme_dict[key] = value
            else:
                print("error!")

        net_extreme.load_state_dict(net_extreme_dict)



    def train(self, max_epochs, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False, stage=False):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            # try:
            if load_latest:
                self.load_checkpoint()

            if load_previous_ckpt:
                directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_prv)
                self.load_state_dict(directory)
            if distill:
                directory_teacher = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_teacher)
                self.load_state_dict(directory_teacher, distill=True)

            for epoch in range(self.epoch + 1, max_epochs + 1):
                self.epoch = epoch

                if stage:

                    if epoch == 1:
                        for name, param in self.actor.net.named_parameters():
                        # 如果参数属于 pgn_module，则保留 requires_grad=True
                            if 'pgn_module' not in name and 'head' not in name:
                                param.requires_grad = False
                    
                            # else:
                            #     print(f'{name}')
                    
                        if self.settings.local_rank in [-1, 0]:
                            print("-------------------------------------------")
                            print("Parameters Frozen!!!")
                            print("-------------------------------------------")

                    if epoch == 6:
                        self.move_pgn_to_teacher()
                    
                    if epoch >= 7:
                        self.move_weight_to_teacher(keep_rate=0.99)
                else:

                    if self.epoch == 1:  # epoch == 1
                        self.update_net_extreme_params(keep_rate=0.)
                    else:
                        self.update_net_extreme_params(keep_rate=0.99)


                self.train_epoch()

                if self.lr_scheduler is not None:
                    if self.settings.scheduler_type != 'cosine':
                        self.lr_scheduler.step()
                    else:
                        self.lr_scheduler.step(epoch - 1)

                save_every_epoch = getattr(self.settings, "save_every_epoch", False)
                save_epochs = [1]
                # save_epochs = [1, 10, 20, 30, 40, 50]
                #or (epoch >= 50 and epoch % 3 == 0)

                if save_every_epoch or (epoch % 20 == 0 and epoch >= 0) or epoch in save_epochs \
                        or (epoch % 10 == 0 and epoch >= 200) or (epoch % 3 == 0 and epoch >= 230):

                # if epoch in save_epochs or (epoch >= 50 and epoch % 3 == 0):
                    if self._checkpoint_dir:
                        if self.settings.local_rank in [-1, 0]:
                            self.save_checkpoint(stage)
                            self.save_net_extreme_checkpoint(stage)

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self, eval=False, stage=False):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        if eval:
            state = {
                'net': net.state_dict()
            }
        else:
            state = {
                'epoch': self.epoch,
                'actor_type': actor_type,
                'net_type': net_type,
                'net': net.state_dict(),
                'net_info': getattr(net, 'info', None),
                'constructor': getattr(net, 'constructor', None),
                'optimizer': self.optimizer.state_dict(),
                'stats': self.stats,
                'settings': self.settings
            }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        print(directory)
        if not os.path.exists(directory):
            print("directory doesn't exist. creating...")
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)
        
        if stage:
            file_path = '{}/{}_prompt_dark_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
        else:
            file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)

    def save_net_extreme_checkpoint(self, eval=False, stage=False):
        """Saves a checkpoint of the network and other variables."""

        net_extreme = self.actor.net_extreme.module if multigpu.is_multi_gpu(
            self.actor.net_extreme) else self.actor.net_extreme

        actor_type = type(self.actor).__name__
        net_type = type(net_extreme).__name__
        if eval:
            state = {
                'net': net_extreme.state_dict()
            }
        else:
            state = {
                'epoch': self.epoch,
                'actor_type': actor_type,
                'net_type': net_type,
                'net': net_extreme.state_dict(),
                'net_info': getattr(net_extreme, 'info', None),
                'constructor': getattr(net_extreme, 'constructor', None),
                'optimizer': self.optimizer.state_dict(),
                'stats': self.stats,
                'settings': self.settings
            }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        print(directory)
        if not os.path.exists(directory):
            print("directory doesn't exist. creating...")
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        if stage:
            file_path = '{}/{}_extreme_prompt_dark_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
        else:
            file_path = '{}/{}_extreme_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        # 0715
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
        net_extreme = self.actor.net_extreme.module if multigpu.is_multi_gpu(
            self.actor.net_extreme) else self.actor.net_extreme

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        net_extreme_type = type(net_extreme).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            checkpoint_list_extreme = sorted(glob.glob('{}/{}/{}_extreme_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                                             self.settings.project_path,
                                                                                             net_extreme_type)))

            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
                if checkpoint_list_extreme:
                    checkpoint_path_extreme = checkpoint_list_extreme[-1]
            else:
                print('No matching checkpoint file found')
                return

            checkpoint_path = '/home/ysy/zr/ostrack_pgn/output/checkpoints/train/UMDATrack/vit_256_ep250_all/UMDATrack_ep0160.pth.tar'
            checkpoint_path_extreme = '/home/ysy/zr/ostrack_pgn/output/checkpoints/train/UMDATrack/vit_256_ep250_all/UMDATrack_extreme_ep0160.pth.tar'

            print("-------------------------------------------")
            print("Loading checkpoint from {}".format(checkpoint_path))
            print("Loading extreme_checkpoint from {}".format(checkpoint_path_extreme))
            print("-------------------------------------------")


        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        if checkpoint_list_extreme:
            checkpoint_dict_extreme = torch.load(checkpoint_path_extreme, map_location='cpu')
            net_extreme.load_state_dict(checkpoint_dict_extreme["net"], strict=False)

        # assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
            # 2021.1.10 Update the epoch in data_samplers
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
        return True

    def delet_old_checkpoint(self, epoch_before=0):
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
        net_type = type(net).__name__

        checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                         self.settings.project_path, net_type)))

        checkpoint_list_to_remove = [ckpt_name for ckpt_name in checkpoint_list if
                                     int(ckpt_name[-12:-8]) < epoch_before]
        for ckpt in checkpoint_list_to_remove:
            os.unlink(ckpt)
        # print(checkpoint_list_to_remove)

    def load_state_dict(self, checkpoint=None, distill=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        if distill:
            net = self.actor.net_teacher.module if multigpu.is_multi_gpu(self.actor.net_teacher) \
                else self.actor.net_teacher
        else:
            net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        if isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        print("Loading pretrained model from ", checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        missing_k, unexpected_k = net.load_state_dict(checkpoint_dict["net"], strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)

        return True
