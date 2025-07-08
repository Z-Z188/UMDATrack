import os
from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, map_boxes_back, map_boxes_back_batch, clip_box, clip_box_batch, batch_bbox_voting
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from lib.utils.pseudo_label_save import write_to_txt
class UMDATrackActor(BaseActor):
    """ Actor for training UMDATrack models """
    #0708
    def __init__(self, net, net_extreme, objective, loss_weight, settings, cfg=None):
        super().__init__(net, net_extreme, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    #0703
    def __call__(self, data, loader_type):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, loader_type)


        # compute losses
        loss, status = self.compute_losses(out_dict, data, loader_type=loader_type)

        #0712
        if loader_type != 'train_extreme':
            return loss, status
        else:
            return loss, status, out_dict

    def forward_pass(self, data, loader_type):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_img = data['template_images'][0].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 256, 256)


        #0708
        if "extreme" in loader_type:
            out_dict = self.net_extreme(template=template_img,
                            search=search_img,
                            mode='train',
                            loader_type=loader_type,
                            )


        # train_mix和 val会进来
        # 而val中没有合成数据集，因此需要特殊判断
        else:
            if loader_type == 'val':
                out_dict = self.net(template=template_img,
                                search=search_img,
                                mode='train',
                                loader_type=loader_type,
                                )
                return out_dict



            # 获取 template_frame_paths 中的非 None元素及其下标
            non_none_indices = [i for i, element in enumerate(data['template_frame_paths'][0]) if element is not None]

            len_extreme = len(non_none_indices)

            # 根据非 None 下标提取其他字段的对应元素
            extracted_data = {
                'template_images': data['template_images'][:, non_none_indices, :, :],
                'template_anno': data['template_anno'][:, non_none_indices, :],
                'template_frame_paths': [data['template_frame_paths'][0][i] for i in non_none_indices],
                'template_masks': [data['template_masks'][:, non_none_indices, :, :]],

                'search_images': data['search_images'][:, non_none_indices, :, :],
                'search_anno': data['search_anno'][:, non_none_indices, :],
                'search_frame_paths': [data['search_frame_paths'][0][i] for i in non_none_indices],
                'search_masks': [data['search_masks'][:, non_none_indices, :, :]],
            }
            template_img_extreme = extracted_data['template_images'][0].view(-1, *data['template_images'].shape[
                                                                                  2:])  # (batch, 3, 128, 128)
            search_img_extreme = extracted_data['search_images'][0].view(-1, *data['search_images'].shape[
                                                                              2:])  # (batch, 3, 256, 256)

            #将合成数据通过教师网络，计算出对应的分数和位置
            with torch.no_grad():
                out_teacher = self.net_extreme(template=template_img_extreme,
                                               search=search_img_extreme,
                                               mode='train',
                                               loader_type=loader_type,
                                               is_ot=True,
                                               is_teacher_ot=True
                                               )

            t_score = out_teacher['t_score'].squeeze(1)
            positions = out_teacher['positions'].detach()

            # 输入网络的是[template, template_extreme]concat之后的结果
            template = torch.cat((template_img, template_img_extreme), dim=0)
            search = torch.cat((search_img, search_img_extreme), dim=0)
            out_dict = self.net(template=template,
                                search=search,
                                mode='train',
                                loader_type=loader_type,
                                is_ot=True,
                                positions=positions,
                                bs=self.bs
                                )

            out_dict['t_score'] = t_score
            out_dict['positions'] = positions
            out_dict['len_extreme'] = len_extreme

            #要判断有没有extreme图片的输入，没有的话，就不用计算OT_Loss了
            if len_extreme > 0:
                out_dict['has_ot'] = True
            else:
                out_dict['has_ot'] = False

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True, loader_type=''):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        #0719
        if loader_type == 'train_extreme':
            pred_boxes = pred_dict['pred_boxes'][:, 0:1, :]
        else:
            pred_boxes = pred_dict['pred_boxes']

        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()


        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)


        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)


        #计算OT_loss
        if loader_type == 'train_mix' and pred_dict['has_ot'] == True:
            t_score = pred_dict['t_score']
            s_score = pred_dict['s_score']
            positions = pred_dict['positions']
            len_extreme = pred_dict['len_extreme']
            ot_loss = self.objective['ot_loss'](t_score, s_score, positions)

        else:
            len_extreme = 0
            ot_loss = torch.tensor(0.0, device=l1_loss.device)


        # weighted sum
        loss = (self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
                + self.loss_weight['focal'] * location_loss) + len_extreme / self.bs * ot_loss


        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/ot": ot_loss.item(),
                      "IoU": mean_iou.item(),
                      }
            return loss, status
        else:
            return loss


