from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import paddle
import paddle.nn as nn

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['EfficientDet']


@register
class EfficientDet(BaseArch):
    """
    EfficientDet architecture, see https://arxiv.org/abs/1911.09070
    Args:
        backbone (object): backbone instance
        neck (object): 'BiFPN' instance
        head (object): `EfficientHead` instance
        post_process (object): 'EfficientPostProcess' instance
    """

    __category__ = 'architecture'
    __inject__ = ['anchor_generator','postprocess']

    def __init__(self,
                 backbone,
                 neck,
                 head="EfficientHead",
                 anchor_generator="AnchorGenerator",
                 postprocess="EfficientPostProcess",
                 box_loss_weight=50.
                 ):
        super(EfficientDet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.anchor_generator = anchor_generator
        self.postprocess = postprocess
        self.box_loss_weight = box_loss_weight

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        anchor_generator = create(cfg["anchor_generator"])
        num_anchors = anchor_generator.num_anchors

        kwargs = {'input_shape': neck.out_shape[1:], "num_anchors": num_anchors}
        head = create(cfg['head'], **kwargs)

        return {'backbone': backbone, 'neck': neck, "head": head,"anchor_generator": anchor_generator}

    def _forward(self):
        body_feats = self.backbone(self.inputs['image'])
        neck_feat = self.neck(body_feats)
        anchors = self.anchor_generator(neck_feat)
        pred_scores, pred_boxes = self.head(neck_feat)

        pred_scores_list = [
            transpose_to_bs_hwa_k(s, self.head.num_classes) for s in pred_scores
        ]
        pred_boxes_list = [
            transpose_to_bs_hwa_k(s, 4) for s in pred_boxes
        ]
        if not self.training:
            if isinstance(self.inputs["im_shape"], list):
                self.inputs["im_shape"] = paddle.concat(self.inputs["im_shape"])

            if isinstance(self.inputs["scale_factor"], list):
                self.inputs["scale_factor"] = paddle.concat(self.inputs["scale_factor"])

            if "scale_factor" in self.inputs:
                self.inputs["scale_factor_wh"] = paddle.concat([self.inputs["scale_factor"][:, 1:2],
                                                                self.inputs["scale_factor"][:, 0:1]], axis=-1)
            else:
                self.inputs["scale_factor_wh"] = paddle.ones([len(self.inputs["im_shape"]), 2]).astype("float32")

            self.inputs["img_whwh"] = paddle.concat([self.inputs["im_shape"][:, 1:2],
                                                     self.inputs["im_shape"][:, 0:1],
                                                     self.inputs["im_shape"][:, 1:2],
                                                     self.inputs["im_shape"][:, 0:1]], axis=-1)

            bbox_pred, bbox_num = self.postprocess(
                pred_scores_list,
                pred_boxes_list,
                anchors,
                self.inputs["scale_factor_wh"],
                self.inputs["img_whwh"])

            return bbox_pred, bbox_num
        else:
            return anchors, pred_scores_list, pred_boxes_list

    def get_loss(self):

        anchors, pred_scores_list, pred_boxes_list = self._forward()

        loss_dict = self.head.losses(anchors, [pred_scores_list, pred_boxes_list], self.inputs)
        loss_cls = loss_dict['cls_loss']
        loss_bbox = loss_dict['reg_loss']
        total_loss = sum(loss_cls) + self.box_loss_weight * sum(loss_bbox)

        loss_dict.update({"loss": total_loss})

        return loss_dict

    def get_pred(self):

        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}

        return output



def transpose_to_bs_hwa_k(tensor, k):
    assert tensor.dim() == 4
    bs, _, h, w = tensor.shape
    tensor = tensor.reshape([bs, -1, k, h, w])
    tensor = tensor.transpose([0, 3, 4, 1, 2])

    return tensor.reshape([bs, -1, k])