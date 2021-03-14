import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet

from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args

REGISTERED_RPN_CLASSES = {}


def register_rpn(cls, name=None):
    global REGISTERED_RPN_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_RPN_CLASSES, f"exist class: {REGISTERED_RPN_CLASSES}"
    REGISTERED_RPN_CLASSES[name] = cls
    return cls


def get_rpn_class(name):
    global REGISTERED_RPN_CLASSES
    assert name in REGISTERED_RPN_CLASSES, f"available class: {REGISTERED_RPN_CLASSES}"
    return REGISTERED_RPN_CLASSES[name]


@register_rpn
class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """deprecated. exists for checkpoint backward compilability (SECOND v1.0)
        """
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        # assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        upsample_strides = [
            np.round(u).astype(np.int64) for u in upsample_strides
        ]
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
                np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_features, num_filters[0], 3,
                stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
            num_class += 1
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters),
                num_anchor_per_loc * num_direction_bins, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                1)

    def forward(self, x):
        # t = time.time()
        # torch.cuda.synchronize()

        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(x)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds
        # torch.cuda.synchronize()
        # print("rpn forward time", time.time() - t)

        return ret_dict


class RPNNoHeadBase(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNNoHeadBase, self).__init__()
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups

        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(
                        ConvTranspose2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        Conv2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_layers, stride=1):
        raise NotImplementedError

    def forward(self, x):
        ups = []
        stage_outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))

        res = {}

        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
            res["out"] = x

        for i, up in enumerate(ups):
            res[f"up{i}"] = up
        for i, out in enumerate(stage_outputs):
            res[f"stage{i}"] = out
        return res

    def forward_stage_block(self, inp_outp_dict):
        next_stg = inp_outp_dict["stages_executed"]+1
        assert (next_stg <= len(self._layer_nums))

        inp_outp_dict[f"stage{next_stg}"] = self.blocks[next_stg-1](
            inp_outp_dict[f"stage{next_stg-1}"])
        inp_outp_dict[f"up{next_stg}"] = self.deblocks[next_stg-1](
            inp_outp_dict[f"stage{next_stg}"])

        return inp_outp_dict

class RPNBase(RPNNoHeadBase):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNBase, self).__init__(
            use_norm=use_norm,
            num_class=num_class,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
            num_input_features=num_input_features,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=box_code_size,
            num_direction_bins=num_direction_bins,
            name=name)
        self._layer_nums=layer_nums
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        self._training = True

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
            self._num_class += 1
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        # I need three different versions of these three convolutions
        # Each of them being for 1, 2 or 3 stages
        print('Building RPN with ability to do imprecise computation')
        # assert len(num_upsample_filters) == 3
        self.conv_cls1, self.conv_cls2, self.conv_cls3 = None, None, None
        self.conv_cls_alternatives = [self.conv_cls1, self.conv_cls2, self.conv_cls3]
        self.conv_box1, self.conv_box2, self.conv_box3 = None, None, None
        self.conv_box_alternatives = [self.conv_box1, self.conv_box2, self.conv_box3]
        for i in range(1, len(num_upsample_filters) + 1):
            self.conv_cls_alternatives[i - 1] = nn.Conv2d(sum(num_upsample_filters[:i]), num_cls, 1)
            self.conv_box_alternatives[i - 1] = nn.Conv2d(sum(num_upsample_filters[:i]),
                                                          num_anchor_per_loc * box_code_size, 1)
        self.conv_cls1, self.conv_cls2, self.conv_cls3 = self.conv_cls_alternatives[:]
        self.conv_box1, self.conv_box2, self.conv_box3 = self.conv_box_alternatives[:]


        if use_direction_classifier:
            # assert len(num_upsample_filters) == 3
            self.conv_dir_cls1, self.conv_dir_cls2, self.conv_dir_cls3 = None, None, None
            self.conv_dir_cls_alternatives = [self.conv_dir_cls1, self.conv_dir_cls2, self.conv_dir_cls3]
            for i in range(1, len(num_upsample_filters) + 1):
                self.conv_dir_cls_alternatives[i - 1] = nn.Conv2d(sum(num_upsample_filters[:i]),
                                                                  num_anchor_per_loc * num_direction_bins, 1)
            self.conv_dir_cls1, self.conv_dir_cls2, self.conv_dir_cls3 = self.conv_dir_cls_alternatives[:]

    def forward(self, x):
        # RPN blocks are executed here
        # Calling forward directly is actually not recommended
        # but the original repo did that instead of using ()
        # so I am going to do it as well ! :)
        res = super().forward(x)
        all_cls_preds = []
        all_box_preds = []
        all_dir_cls_preds = []
        # Now we execute RPN Head
        for i in range(len(self._layer_nums)):
            if self._training or i + 1 == len(self._layer_nums):
                to_concat = []
                for j in range(i + 1):
                    to_concat.append(res["up" + str(j)])
                # last iter will save x, ignore previous ones
                x = torch.cat(to_concat, dim=1)
                x_cls = self.conv_cls_alternatives[i](x)
                C, H, W = x_cls.shape[1:]
                x_cls = x_cls.view(-1, self._num_anchor_per_loc, self._num_class, H, W).permute(
                    0, 1, 3, 4, 2).contiguous()
                all_cls_preds.append(x_cls)

                x_box = self.conv_box_alternatives[i](x)
                x_box = x_box.view(-1, self._num_anchor_per_loc, self._box_code_size, H, W).permute(
                    0, 1, 3, 4, 2).contiguous()
                all_box_preds.append(x_box)

                if self._use_direction_classifier:
                    x_dir = self.conv_dir_cls_alternatives[i](x)
                    x_dir = x_dir.view(-1, self._num_anchor_per_loc, self._num_direction_bins, H, W).permute(
                        0, 1, 3, 4, 2).contiguous()
                    all_dir_cls_preds.append(x_dir)

        cls_preds = torch.stack(all_cls_preds) if self._training else all_cls_preds[-1]
        box_preds = torch.stack(all_box_preds) if self._training else all_box_preds[-1]
        if self._use_direction_classifier:
            dir_cls_preds = torch.stack(all_dir_cls_preds) if self._training else all_dir_cls_preds[-1]

        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }

        if self._use_direction_classifier:
            ret_dict["dir_cls_preds"] = dir_cls_preds

        return ret_dict

    def forward_stage(self, inp_outp_dict):
        next_stg = inp_outp_dict["stages_executed"] + 1
        res = self.forward_stage_block(inp_outp_dict)
        if next_stg == 1:
            # Input of backbone should be assigned before this call:
            # inp_outp_dict["stage0"] = x
            backbone_out = res["up1"]
        else:
            to_concat = [res["up" + str(j+1)] for j in range(next_stg)]
            backbone_out = torch.cat(to_concat, dim=1)

        # This is detection head
        # overriding predictions with new ones is what is intended
        x_cls = self.conv_cls_alternatives[next_stg-1](backbone_out)
        C, H, W = x_cls.shape[1:]
        inp_outp_dict["cls_preds"] = x_cls.view(-1, self._num_anchor_per_loc, self._num_class, H, W).permute(
            0, 1, 3, 4, 2).contiguous()

        x_box = self.conv_box_alternatives[next_stg-1](backbone_out)
        inp_outp_dict["box_preds"] = x_box.view(-1, self._num_anchor_per_loc, self._box_code_size, H, W).permute(
            0, 1, 3, 4, 2).contiguous()

        if self._use_direction_classifier:
            x_dir = self.conv_dir_cls_alternatives[next_stg-1](backbone_out)
            inp_outp_dict["dir_cls_preds"] = x_dir.view(-1, self._num_anchor_per_loc, self._num_direction_bins, H, W).permute(
                0, 1, 3, 4, 2).contiguous()

        inp_outp_dict["stages_executed"] += 1

        return inp_outp_dict

    def set_mode_training(self):
        self._training = True

    def set_mode_evaluation(self):
        self._training = False


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


@register_rpn
class ResNetRPN(RPNBase):
    def __init__(self, *args, **kw):
        self.inplanes = -1
        super(ResNetRPN, self).__init__(*args, **kw)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # if zero_init_residual:
        for m in self.modules():
            if isinstance(m, resnet.Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, resnet.BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, inplanes, planes, num_layers, stride=1):
        if self.inplanes == -1:
            self.inplanes = self._num_input_features
        block = resnet.BasicBlock
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_layers):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers), self.inplanes


@register_rpn
class RPNV2(RPNBase):
    def _make_layer(self, inplanes, planes, num_layers, stride=1):
        """
        Constructs a single RPN branch
        Args:
            inplanes:
            planes:
            num_layers:
            stride:

        Returns:

        """
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_layers):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes


@register_rpn
class RPNNoHead(RPNNoHeadBase):
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes
