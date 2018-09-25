import os
from collections import OrderedDict
from typing import List
import heapq

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child.named_parameters():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x ** 2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq, pad, pad, pad, pad), 2),
                            torch.cat((pad, x_sq, pad, pad, pad), 2),
                            torch.cat((pad, pad, x_sq, pad, pad), 2),
                            torch.cat((pad, pad, pad, x_sq, pad), 2),
                            torch.cat((pad, pad, pad, pad, x_sq), 2)), 1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:, 2:-2, :, :]
        x = x / ((2. + 0.0001 * x_sumsq) ** 0.75)
        return x


class MDNet(nn.Module):
    class FilterMeta:
        def __init__(self,
                     block_idx, layer_idx, filter_idx,
                     target_rel_thresh,
                     unactivated_thresh,
                     unactivated_cnt_thresh):
            self.block_idx = block_idx
            self.layer_idx = layer_idx
            self.filter_idx = filter_idx

            self.target_rel_thresh = target_rel_thresh
            self.unactivated_thresh = unactivated_thresh
            self.unactivated_cnt_thresh = unactivated_cnt_thresh

            self.resp_sum = 0
            self.resp_cnt = 0

            self.max_resp = 0
            self.unactivated_cnt = 0

            self.bg_resp_sum = 0
            self.bg_resp_cnt = 0
            self.target_resp_sum = 0
            self.target_resp_cnt = 0

        def reset(self):
            self.resp_sum = 0
            self.resp_cnt = 0

            self.max_resp = 0
            self.unactivated_cnt = 0

            self.bg_resp_sum = 0
            self.bg_resp_cnt = 0
            self.target_resp_sum = 0
            self.target_resp_cnt = 0

        def report_resp(self, resp: float, is_target=False, is_bg=False):
            self.resp_sum += resp
            self.resp_cnt += 1

            if resp > self.max_resp:
                self.max_resp = resp

            if resp <= self.max_resp * self.unactivated_thresh:
                self.unactivated_cnt += 1
            else:
                self.unactivated_cnt = 0

            if is_bg:
                self.bg_resp_sum += resp
                self.bg_resp_cnt += 1
            elif is_target:
                self.target_resp_sum += resp
                self.target_resp_cnt += 1

        def average_resp(self):
            return self.resp_sum / self.resp_cnt

        def target_rel(self):
            return 100 \
                if self.target_resp_cnt * self.bg_resp_sum == 0 \
                else self.target_resp_sum * self.bg_resp_cnt / (self.target_resp_cnt * self.bg_resp_sum)

        def to_be_evolved(self, resp_thresh=0):
            return self.resp_cnt > 0 and \
                   (self.unactivated_cnt >= self.unactivated_cnt_thresh or
                    self.average_resp() < resp_thresh) and \
                   self.target_rel() <= self.target_rel_thresh

    class LayerMeta:
        def __init__(self,
                     block_name,
                     block_idx,
                     layer_idx,
                     num_filters,
                     transposed,
                     bg_rel_thresh,
                     unactivated_thresh,
                     unactivated_cnt_thresh):
            self.block_name = block_name
            self.block_idx = block_idx
            self.layer_idx = layer_idx
            self.transposed = transposed
            self.filter_meta = [MDNet.FilterMeta(block_idx, layer_idx, filter_idx,
                                                 bg_rel_thresh,
                                                 unactivated_thresh,
                                                 unactivated_cnt_thresh)
                                for filter_idx in range(num_filters)]
            self.next_layer_meta = None

            self.resp_sum = 0
            self.resp_cnt = 0

        def average_resp(self):
            return self.resp_sum / self.resp_cnt if self.resp_cnt > 0 else 0

        def report_resp(self, responses, is_target=False, is_bg=False):
            self.resp_sum += sum(responses)
            self.resp_cnt += len(responses)
            for idx, resp in enumerate(responses):
                self.filter_meta[idx].report_resp(resp, is_target=is_target, is_bg=is_bg)

    def __init__(self,
                 model_path=None,
                 K=1,
                 fe_layers=None,
                 target_rel_thresh=0.1,
                 unactivated_thresh=0.01,
                 unactivated_cnt_thresh: int = 1000,
                 low_resp_thresh=0.1,
                 record_resp=False,
                 lr_boost=1.5):
        super(MDNet, self).__init__()
        if fe_layers is None:
            fe_layers = set()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU())),
            ('fc4', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512 * 3 * 3, 512),
                                  nn.ReLU())),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, 512),
                                  nn.ReLU()))]))

        self.filter_resp_on_pos_samples = OrderedDict([
            (name, []) for name, _ in self.layers.named_children() if name in fe_layers
        ]) if record_resp else None
        self.filter_resp_on_neg_samples = OrderedDict([
            (name, []) for name, _ in self.layers.named_children() if name in fe_layers
        ]) if record_resp else None

        self.fe_layer_meta = self.create_fe_layer_meta(fe_layers,
                                                       target_rel_thresh,
                                                       unactivated_thresh,
                                                       unactivated_cnt_thresh)
        self.low_resp_thresh = low_resp_thresh
        self.lr_boost = lr_boost

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unknown model format: %s" % (model_path))
        self.build_param_dict()

    def create_fe_layer_meta(self,
                             fe_layers: List[str],
                             bg_rel_thresh,
                             unactivated_thresh,
                             unactivated_cnt_thresh) -> OrderedDict:
        prev_layer_meta = None
        layer_meta = []
        for block_idx, (block_name, layer_block) in enumerate(self.layers.named_children()):
            for idx, layer in enumerate(layer_block):
                if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
                    assert not (type(layer) is nn.Conv2d and layer.groups != 1), \
                        'Grouped convolution layer is not supported!'
                    is_fe_layer = block_name in fe_layers
                    if prev_layer_meta is not None or is_fe_layer:
                        meta = MDNet.LayerMeta(block_name,
                                               block_idx,
                                               idx,
                                               layer.bias.shape[0],
                                               layer.transposed if type(layer) is nn.Conv2d else False,
                                               bg_rel_thresh,
                                               unactivated_thresh,
                                               unactivated_cnt_thresh)
                        if prev_layer_meta is not None:
                            prev_layer_meta.next_layer_meta = meta
                            prev_layer_meta = None
                        if is_fe_layer:
                            layer_meta.append((block_name, meta))
                            prev_layer_meta = meta
        if prev_layer_meta is not None:
            prev_layer_meta.next_layer_meta = MDNet.LayerMeta('', -1, 1, 2, False, 0, 0, 0)
        return OrderedDict(layer_meta)

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d' % (k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6', is_target=False, is_bg=False):
        #
        # forward model from in_layer to out_layer

        output = None
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)

                if is_target or is_bg:
                    if self.filter_resp_on_pos_samples is not None:
                        if is_target:
                            self.filter_resp_on_pos_samples[name].append(
                                torch.mean(torch.nn.functional.avg_pool2d(x.data, x.shape[-2:]),
                                           dim=0).cpu().view(-1).numpy()
                                if x.dim() == 4
                                else torch.mean(x.data, dim=0).view(-1).cpu().numpy()
                            )
                        else:
                            self.filter_resp_on_neg_samples[name].append(
                                torch.mean(torch.nn.functional.avg_pool2d(x.data, x.shape[-2:]),
                                           dim=0).cpu().view(-1).numpy()
                                if x.dim() == 4
                                else torch.mean(x.data, dim=0).view(-1).cpu().numpy()
                            )

                    if name in self.fe_layer_meta:
                        responses = torch.mean(torch.nn.functional.avg_pool2d(x.data, x.shape[-2:]),
                                               dim=0).view(-1).cpu().numpy() \
                            if x.dim() == 4 \
                            else torch.mean(x.data, dim=0).view(-1).cpu().numpy()
                        self.fe_layer_meta[name].report_resp(responses, is_target=is_target, is_bg=is_bg)

                if name == 'conv3':
                    x = x.view(x.size(0), -1)
                if name == out_layer:
                    output = x

        x = self.branches[k](x)
        if out_layer == 'fc6':
            output = x
        elif out_layer == 'fc6_softmax':
            output = F.softmax(x)
        return output

    def evolve_filters(self):
        # print('Start filter evolution...')
        for block_name, layer_meta in self.fe_layer_meta.items():
            layer = self.layers[layer_meta.block_idx][layer_meta.layer_idx]
            next_layer = self.layers[layer_meta.next_layer_meta.block_idx][layer_meta.next_layer_meta.layer_idx]
            resp_thresh = layer_meta.average_resp() * self.low_resp_thresh

            filters_to_be_evolved = list(filter(lambda meta: meta.to_be_evolved(resp_thresh),
                                                layer_meta.filter_meta))
            max_num_evolving_filters = len(layer_meta.filter_meta) >> 4
            if len(filters_to_be_evolved) > max_num_evolving_filters:
                filters_to_be_evolved = heapq.nsmallest(max_num_evolving_filters,
                                                        filters_to_be_evolved,
                                                        lambda meta: meta.target_rel())

            # print('Evolving {} filters in {}...'.format(len(filters_to_be_evolved), block_name))

            for filter_meta in filters_to_be_evolved:
                # print('Evolving filter {} in {}'.format(filter_idx, block_name))
                if layer_meta.transposed:
                    layer.weight.data[:, filter_meta.filter_idx, ...] = 0
                else:
                    layer.weight.data[filter_meta.filter_idx, ...] = 0
                layer.bias.data[:] = max(resp_thresh, filter_meta.average_resp()) / self.lr_boost

                if layer_meta.next_layer_meta.transposed:
                    next_layer.weight.data[filter_meta.filter_idx, ...] = \
                        -next_layer.weight.data[filter_meta.filter_idx, ...] * self.lr_boost
                else:
                    next_layer.weight.data[:, filter_meta.filter_idx, ...] = \
                        -next_layer.weight.data[:, filter_meta.filter_idx, ...] * self.lr_boost

                filter_meta.reset()

    def dump_filter_resp(self, prefix='filter_resp', output_dir=os.path.join('analysis', 'data')):
        if self.filter_resp_on_pos_samples is not None:
            print('Dumping filter responses...')
            os.makedirs(output_dir, exist_ok=True)
            for name, resp in self.filter_resp_on_pos_samples.items():
                fn = os.path.abspath(os.path.join(output_dir, '{}_target_{}.csv'.format(prefix, name)))
                print('Dumping average response on target of {} into {}'.format(name, fn))
                with open(fn, 'w') as f:
                    for resp_per_frame in resp:
                        f.write('{}\n'.format(','.join(map(str, resp_per_frame))))
            for name, resp in self.filter_resp_on_neg_samples.items():
                fn = os.path.abspath(os.path.join(output_dir, '{}_bg_{}.csv'.format(prefix, name)))
                print('Dumping average response on background of {} into {}'.format(name, fn))
                with open(fn, 'w') as f:
                    for resp_per_frame in resp:
                        f.write('{}\n'.format(','.join(map(str, resp_per_frame))))
        else:
            raise RuntimeError("Filter responses are not recorded!")

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        return loss


class Accuracy:
    def __init__(self):
        pass

    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision:
    def __init__(self):
        pass

    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)

        return prec.data[0]
