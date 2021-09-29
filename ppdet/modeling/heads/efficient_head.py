from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .. import initializer as init
from ppdet.core.workspace import register


__all__ = ['EfficientHead']


@register
class EfficientHead(nn.Layer):
    """
    The head used in EfficientDet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """
    __inject__ = ['loss_func']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 repeat=3,
                 in_channel=64, #d0
                 num_anchors=9,
                 loss_func="EfficientDetLoss",
                 prior_prob=0.01,
        ):
        '''
        Args:
            input_shape (List[ShapeSpec]): input shape.
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors.
            conv_dims (List[int]): dimensions for each convolution layer.
            loss_func (class): the class is used to compute loss.
            prior_prob (float): Prior weight for computing bias.
        '''
        super(EfficientHead, self).__init__()
        self.repeat = repeat
        self.prior_prob = prior_prob
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.get_loss = loss_func

        cls_net = []
        reg_net = []

        for i in range(self.repeat):
            cls_net.append(SeparableConvBNSwich(in_channels=in_channel,out_channels=in_channel,kernel_size=3))
            reg_net.append(SeparableConvBNSwich(in_channels=in_channel,out_channels=in_channel,kernel_size=3))

        self.cls_net = nn.Sequential(*cls_net)
        self.reg_net = nn.Sequential(*reg_net)

        self.cls_header = SeparableConvBNSwich(in_channels=in_channel,out_channels=num_anchors*num_classes,kernel_size=3,norm=False,act=False)
        self.bbox_header = SeparableConvBNSwich(in_channels=in_channel,out_channels=num_anchors*4,kernel_size=3,norm=False,act=False)

        init.reset_initialized_parameter(self)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                init.normal_(m.weight, mean=0., std=0.01)
                init.constant_(m.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape,num_anchors):
        return {
            'in_channel': [i.channels for i in input_shape][0],
            'num_anchors': num_anchors,
        }
    
    def forward(self, feats):
        pred_scores = []
        pred_boxes = []

        for feat in feats:
            pred_scores.append(self.cls_header(self.cls_net(feat)))
            pred_boxes.append(self.bbox_header(self.reg_net(feat)))

        return pred_scores, pred_boxes

    def losses(self, anchors, preds, inputs):
        anchors = paddle.concat(anchors)

        return self.get_loss(anchors, preds, inputs)


class SeparableConvBNSwich(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding='same',
                 norm = True,
                 act = True,
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBNSwich(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            norm=norm,
            act=False,
            **kwargs)
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self.piontwise_conv = ConvBNSwich(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            norm=norm,
            act=act,
            data_format=data_format)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    elif paddle.distributed.ParallelEnv().nranks == 1:
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)

class ConvBNSwich(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 norm = True,
                 act=True,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        if norm:
            self._batch_norm = SyncBatchNorm(out_channels, data_format=data_format)
        else:
            self._batch_norm = None
        self.act = act

    def forward(self, x):
        x = self._conv(x)
        if self._batch_norm!=None:
            x = self._batch_norm(x)
        if self.act:
            x = F.swish(x)
        return x