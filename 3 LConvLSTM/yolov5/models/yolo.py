import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

import math

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, NMS, autoShape, AttBottleneckCSP, AttConv
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import check_anchor_order, make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr, dfs_freeze


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=(), bs=2, sz=640, sl=2):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.bn = nn.ModuleList(nn.BatchNorm2d(x) for x in ch)  # output conv
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.sz = sz
        self.sl = sl
        # self.final = nn.ModuleList(nn.Conv2d(self.no * self.na, self.no * self.na, 1) for _ in ch)  # output conv

        self.export = False  # onnx export
        self.bs = bs
        # self.upsample = nn.Upsample(sz, mode='bilinear', align_corners=True)
        # ch = [24, 24, 24]

        # self.qrnn1 = QRNN3DLayer(ch[0], ch[0])
        # self.qrnn2 = QRNN3DLayer(ch[1], ch[1])
        # self.qrnn3 = QRNN3DLayer(ch[2], ch[2])
        # self.qrnn = [self.qrnn1, self.qrnn2, self.qrnn3]
        self.convlstm1 = ConvLSTM(input_dim=ch[0],
                                  hidden_dim=[ch[0]],
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=True)

        self.convlstm2 = ConvLSTM(input_dim=ch[1],
                                  hidden_dim=[ch[1]],
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=True)

        self.convlstm3 = ConvLSTM(input_dim=ch[2],
                                  hidden_dim=[ch[2]],
                                  kernel_size=(3, 3),
                                  num_layers=1,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=True)

        self.convlstms = [self.convlstm1, self.convlstm2, self.convlstm3]


        self.maps_outputs = []

    def init_convlstms(self):
        for convlstm in self.convlstms:
            convlstm.init_convlstm()

    def forward(self, x, masksOP=None):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.maps_outputs.clear()
        self.training |= self.export
        # self.sl = 3
        for i in range(self.nl):

            # cx = x[i].view(-1, x[i].shape[1], self.sl, x[i].shape[2], x[i].shape[3])
            # masksOP = torch.nn.functional.interpolate(masksOP, (x[i].shape[2], x[i].shape[3]))
            # xo = self.qrnn[i](cx)
            # cx = xo.view(-1, x[i].shape[1], x[i].shape[2], x[i].shape[3])

            cx = x[i].view(-1, self.sl, x[i].shape[1], x[i].shape[2], x[i].shape[3])
            xo, xhc, att_maps = self.convlstms[i](cx)
            cx = xo[0].view(-1, x[i].shape[1], x[i].shape[2], x[i].shape[3])
            self.maps_outputs.append(att_maps)

            cx = self.act(self.bn[i](cx))

            # self.maps_outputs.append(att_maps)

            x[i] = self.m[i](cx)  # conv

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # if not self.training:  # inference
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))

        return torch.cat(z, 1), x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, seq_len=2, size=640,
                 batch_size=2):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.seq = seq_len

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], bs=batch_size, sz=size,
                                            sl=seq_len)  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = [8, 16, 32]
            _, x_list = self.forward(torch.zeros(2 * seq_len, ch, size, size), torch.zeros(2, seq_len, size, size))
            m.stride = torch.tensor([size / x.shape[-2] for x in x_list])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, masksOP = None, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, masksOP, profile)  # single-scale inference, train

    def forward_once(self, x, masksOP = None, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            if m is self.model[-1]:
                x = m(x, masksOP)
            else:
                x = m(x)
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


def parse_model(d, ch, bs, sz, sl):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            args.append(bs)
            args.append(sz)
            args.append(sl)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        dfs_freeze(m_)
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        # self.bn_x = nn.BatchNorm2d(4 * self.hidden_dim)
        # self.bn_h = nn.BatchNorm2d(4 * self.hidden_dim)
        # self.bn_c = nn.BatchNorm2d(self.hidden_dim)

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        # self.conv_x = nn.Conv2d(in_channels=self.input_dim,
        #                         out_channels=4 * self.hidden_dim,
        #                         kernel_size=self.kernel_size,
        #                         padding=self.padding,
        #                         bias=self.bias)

        # self.conv_h = nn.Conv2d(in_channels=self.hidden_dim,
        #                         out_channels=4 * self.hidden_dim,
        #                         kernel_size=self.kernel_size,
        #                         padding=self.padding,
        #                         bias=self.bias)

        # self.mul_c = None

    def init_convlstm_cell(self):
        pass
        # self.conv_x.bias.data[self.input_dim:self.input_dim * 2] = 2
        # self.conv_h.bias.data[self.input_dim:self.input_dim * 2] = 2
        # nn.init.orthogonal_(self.conv.weight)

    # def forward(self, input_tensor, cur_state):
    #     h_cur, c_cur = cur_state

    #     x_concat = self.bn_x(self.conv_x(input_tensor))
    #     h_concat = self.bn_h(self.conv_h(h_cur))

    #     i_x, f_x, c_x, o_x = torch.split(x_concat, self.hidden_dim, dim=1)
    #     i_h, f_h, c_h, o_h = torch.split(h_concat, self.hidden_dim, dim=1)

    #     i = torch.sigmoid(i_x + i_h)
    #     f = torch.sigmoid(f_x + f_h)
    #     o = torch.sigmoid(o_x + o_h)
    #     g = torch.tanh(c_x + c_h)

    #     c_next = f * c_cur + i * g
    #     h_next = o * torch.tanh(self.bn_c(c_next))

    #     return h_next, c_next

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    # def forward(self, input_tensor, cur_state):
    #     h_cur, c_cur = cur_state
    #
    #     x_concat = self.bn_x(self.conv_x(input_tensor))
    #     h_concat = self.bn_h(self.conv_h(h_cur))
    #
    #     i_x, f_x, c_x, o_x = torch.split(x_concat, self.hidden_dim, dim=1)
    #     i_h, f_h, c_h, o_h = torch.split(h_concat, self.hidden_dim, dim=1)
    #     i_c, f_c, o_c = torch.split(self.mul_c, self.hidden_dim, dim=1)
    #
    #     i = torch.sigmoid(i_x + i_h + i_c * c_cur)
    #     f = torch.sigmoid(f_x + f_h + f_c * c_cur)
    #     g = torch.tanh(c_x + c_h)
    #     c_next = f * c_cur + i * g
    #     o = torch.sigmoid(o_x + o_h + o_c * c_next)
    #     h_next = o * torch.tanh(self.bn_c(c_next))
    #
    #     return h_next, c_next

    # def init_hidden(self, batch_size, image_size):
    #     height, width = image_size
    #     if 'cpu' in str(self.conv_x.weight.device):
    #         # self.mul_c = torch.zeros(batch_size, self.hidden_dim * 3, height, width, device=self.conv_x.weight.device)
    #         return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_x.weight.device),
    #                 torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_x.weight.device))
    #     else:
    #         # self.mul_c = torch.zeros(batch_size, self.hidden_dim * 3, height, width,
    #         #                          device=self.conv_x.weight.device).half()
    #         return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_x.weight.device).half(),
    #                 torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_x.weight.device).half())

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        if 'cpu' in str(self.conv.weight.device):
            # self.mul_c = torch.zeros(batch_size, self.hidden_dim * 3, height, width, device=self.conv_x.weight.device)
            return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                    torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
        else:
            # self.mul_c = torch.zeros(batch_size, self.hidden_dim * 3, height, width,
            #                          device=self.conv_x.weight.device).half()
            return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).half(),
                    torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).half())


class ConvAttention(nn.Module):
    def __init__(self, inchannel):
        super(ConvAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inchannel, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feats):
        feats = self.attention(feats)
        return feats


class ConvLSTM(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        # self.attention = ConvAttention(input_dim * 2)
        # self.attention = AttBottleneckCSP(input_dim * 2, 1, 4, e=1)

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def init_convlstm(self):
        for convlstm_cell in self.cell_list:
            convlstm_cell.init_convlstm_cell()

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        att_maps_t = torch.zeros([b, seq_len, h, w], dtype=input_tensor.dtype)

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # att_t = torch.cat((cur_layer_input[:, t, :, :, :], h), 1)
                # att, sig_att = self.attention(att_t)
                #
                # att_maps_t[:, t:t + 1, :, :] = att[:, :, :, :]
                # att_in = cur_layer_input[:, t, :, :, :] * sig_att
                # # att_in = att_in * sig_ch_att.expand(att_in.shape)
                # h, c = self.cell_list[layer_idx](input_tensor=att_in, cur_state=[h, c])
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list, att_maps_t

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
