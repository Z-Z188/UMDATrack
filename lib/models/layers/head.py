import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from lib.utils.heatmap_test import save_individual_heatmaps
# from mmengine.model import constant_init, normal_init
# from mmcv.cnn import ConvModule
# from mmcv.ops import DeformConv2d
# from .decoder_head import DecoderHead
# from mmcv.ops import
# from mmdet.core.utils import DeformConv

from lib.models.layers.frozen_bn import FrozenBatchNorm2d

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def c_sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.grid_size = self.feat_sz+1
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn, kernel_size=2)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn, kernel_size=2)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():

            self.indice = torch.arange(0, self.grid_size ).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.grid_size , 1)) \
                .view((self.grid_size  * self.grid_size ,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.grid_size )) \
                .view((self.grid_size  * self.grid_size ,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, torch.stack((prob_vec_tl,prob_vec_br),1)
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.grid_size  * self.grid_size ))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, feat_tz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        # self.feat_tz = feat_tz
        self.stride = stride
        # self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, feat_size, gt_score_map=None, loader_type='train_mix', positions=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        if loader_type == 'train_extreme':
            bbox, topk_score = self.cal_bbox_extreme(score_map_ctr, feat_size, size_map, offset_map)
            return score_map_ctr, bbox, size_map, offset_map, topk_score

        else:
            if positions is None:
                bbox, t_score, positions = self.cal_bbox(score_map_ctr, feat_size, size_map, offset_map)
                return score_map_ctr, bbox, size_map, offset_map, t_score, positions
            else:
                s_score = self.get_s_score(score_map_ctr, positions)
                return s_score

    def get_s_score(self, score_map_ctr, positions):
        batch_size = score_map_ctr.size(0)
        y_coords, x_coords = positions[:, 0], positions[:, 1]
        values = score_map_ctr[torch.arange(batch_size), 0,  y_coords, x_coords]
        return values

    def cal_bbox(self, score_map_ctr, feat_size, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // feat_size
        idx_x = idx % feat_size

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)


        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / feat_size,
                          (idx_y.to(torch.float) + offset[:, 1:]) / feat_size,
                          size.squeeze(-1)], dim=1)

        positions = torch.cat([idx_y, idx_x], dim=1)

        return bbox, max_score, positions

    #0718/0722
    def cal_bbox_extreme(self, score_map_ctr, feat_size, size_map, offset_map, topk=25, return_score=True):
        # 选取前 topk 个分数最高的点
        topk_scores, topk_idx = torch.topk(score_map_ctr.flatten(1), topk, dim=1)

        idx_y = topk_idx // feat_size
        idx_x = topk_idx % feat_size

        #0721
        #score of the point which is far from the top point is set to 0
        center_point_y = idx_y[:, 0]
        center_point_x = idx_x[:, 0]

        for i in range(idx_x.shape[0]):
            for j in range(idx_x.shape[1]):
                if abs(idx_x[i, j]-center_point_x[i]) > 2 or abs(idx_y[i, j]-center_point_y[i]) > 2:
                    topk_scores[i, j] = 0

        idx_y = idx_y.unsqueeze(-1)
        idx_x = idx_x.unsqueeze(-1)

        topk_idx = topk_idx.unsqueeze(1).expand(topk_idx.shape[0], 2, topk)
        size = size_map.flatten(2).gather(dim=2, index=topk_idx)
        offset = offset_map.flatten(2).gather(dim=2, index=topk_idx)

        topk_idx = topk_idx.permute(0,2,1)
        size = size.permute(0,2,1)
        offset = offset.permute(0,2,1)


        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :, :1]) / feat_size,
                        (idx_y.to(torch.float) + offset[:, :, 1:]) / feat_size,
                        size], dim=2)

        if return_score:
            return bbox, topk_scores
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)

        # save_individual_heatmaps(score_map_ctr, save_dir="/nvme0n1/whj_file/models/Light-UAV-Track-Dual_0702/heatmap_test", type='')

        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

def build_box_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE

    if "CORNER" in cfg.MODEL.HEAD.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=hidden_dim, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.HEAD.TYPE == "CENTER":
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                      feat_sz=feat_sz, stride=stride)
        return center_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)


