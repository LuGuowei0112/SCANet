import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange

import torchvision.models as models

import models
from models.CAFF import TransformerFusionBlock
from models.HSF import MSPC, PFA1, PFA2, SGR
from models.help_funcs import TwoLayerConv2d


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'SCANet':
        net = SCANet(output_nc=2)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)

###############################################################################
# main Functions
###############################################################################

class SCANet(nn.Module):
    def __init__(self, output_nc, crossattention=True,
                 channel=32):
        super(SCANet, self).__init__()
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        #使用预训练的 VGG16_BN（也可以替换为其他 ResNet 版本）
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.F_MS_2 = MSPC(256, channel)
        self.F_MS_3 = MSPC(512, channel)
        self.F_MS_4 = MSPC(512, channel)
        self.agg1 = PFA1(channel)

        self.F_MS_0 = MSPC(64, channel)
        self.F_MS_1 = MSPC(128, channel)
        self.F_MS_2_new = MSPC(channel, channel)
        self.agg2 = PFA2(channel)

        self.crossattention = crossattention
        self.transformerfusionfblock0 = TransformerFusionBlock(d_model=64, vert_anchors=16, horz_anchors=16)
        self.transformerfusionfblock1 = TransformerFusionBlock(d_model=128, vert_anchors=16, horz_anchors=16)
        self.transformerfusionfblock2 = TransformerFusionBlock(d_model=256,vert_anchors=16,horz_anchors=16)
        self.transformerfusionfblock3 = TransformerFusionBlock(d_model=512,vert_anchors=16,horz_anchors=16)
        self.transformerfusionfblock4 = TransformerFusionBlock(d_model=512,vert_anchors=16,horz_anchors=16)

        self.SGR_map = SGR()
        self.classifier = TwoLayerConv2d(in_channels=32*3, out_channels=output_nc)
        #
        # self.agant1 = self._make_agant_layer(32 * 3, 32 * 2)
        # self.agant2 = self._make_agant_layer(32 * 2, 32)
        # self.output_nc = output_nc
        # self.out_conv = nn.Conv2d(32 * 1, self.output_nc, kernel_size=1, stride=1, bias=True)


        #
        # def _make_agant_layer(self, inplanes, planes):
        #     layers = nn.Sequential(
        #         nn.Conv2d(inplanes, planes, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(planes),
        #         nn.ReLU(inplace=True)
        #     )
        #     return layers

    def forward(self, A, B):
        # forward backbone resnet
        layer0_A= self.inc(A) #64*64*64
        layer1_A = self.down1(layer0_A) #256*64*64
        layer2_A = self.down2(layer1_A) #512*32*32
        layer3_A = self.down3(layer2_A) #1024*16*16
        layer4_A = self.down4(layer3_A) #2048*8*8

        layer0_B= self.inc(B) #64*64*64
        layer1_B = self.down1(layer0_B) #256*64*64
        layer2_B = self.down2(layer1_B) #512*32*32
        layer3_B = self.down3(layer2_B) #1024*16*16
        layer4_B = self.down4(layer3_B) #2048*8*8

        if self.crossattention:
            layer2 = self.transformerfusionfblock2([layer2_A, layer2_B])
            layer3 = self.transformerfusionfblock3([layer3_A, layer3_B])
            layer4 = self.transformerfusionfblock4([layer4_A, layer4_B])

            layer0 = self.transformerfusionfblock0([layer0_A, layer0_B])
            layer1 = self.transformerfusionfblock1([layer1_A, layer1_B])

        layer2 = self.F_MS_2(layer2)
        layer3 = self.F_MS_3(layer3)
        layer4 = self.F_MS_4(layer4)
        Semantic_map = self.agg1(layer4,layer3,layer2)

        layer0,layer1,layer2 = self.SGR_map(Semantic_map.sigmoid(),layer0,layer1,layer2)

        layer0 = self.F_MS_0(layer0)
        layer1 = self.F_MS_1(layer1)
        layer2 = self.F_MS_2_new(layer2)

        y = self.agg2(layer2,layer1,layer0)

        # forward small cnn
        x = self.classifier(y)

        return x

