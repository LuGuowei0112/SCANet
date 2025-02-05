import torch.nn as nn
import torch
from torch.nn import init, Sequential
import cv2
import numpy as np
import pandas as pd
import math
import torch.nn.functional as F

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.Q_T1 = nn.Linear(d_model, h * self.d_k)  # query projection
        self.K_T1 = nn.Linear(d_model, h * self.d_k)  # key projection
        self.V_T1 = nn.Linear(d_model, h * self.d_v)  # value projection

        self.Q_T2 = nn.Linear(d_model, h * self.d_k)  # query projection
        self.K_T2 = nn.Linear(d_model, h * self.d_k)  # key projection
        self.V_T2 = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_T1 = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_T2 = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        T1_flat = x[0]
        T2_flat = x[1]
        b_s, nq = T1_flat.shape[:2]
        nk = T1_flat.shape[1]

        # Self-Attention
        T1_flat = self.LN1(T1_flat)
        Q_pro_T1 = self.Q_T1(T1_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1,
                                                                                                     3)  # (b_s, h, nq, d_k)
        K_pro_T1 = self.K_T1(T1_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3,
                                                                                                     1)  # (b_s, h, d_k, nk) K^T
        V_pro_T1 = self.V_T1(T1_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1,
                                                                                                     3)  # (b_s, h, nk, d_v)

        T2_flat = self.LN2(T2_flat)
        Q_pro_T2 = self.Q_T2(T2_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1,
                                                                                                  3)  # (b_s, h, nq, d_k)
        K_pro_T2 = self.K_T2(T2_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3,
                                                                                                  1)  # (b_s, h, d_k, nk) K^T
        V_pro_T2 = self.V_T2(T2_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1,
                                                                                                  3)  # (b_s, h, nk, d_v)

        att_T1 = torch.matmul(Q_pro_T2, K_pro_T1) / np.sqrt(self.d_k)
        att_T2 = torch.matmul(Q_pro_T1, K_pro_T2) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_T1 = torch.softmax(att_T1, -1)
        att_T1 = self.attn_drop(att_T1)
        att_T2 = torch.softmax(att_T2, -1)
        att_T2 = self.attn_drop(att_T2)

        # output
        out_T1 = torch.matmul(att_T1, V_pro_T1).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                                     self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_T1 = self.resid_drop(self.out_proj_T1(out_T1))  # (b_s, nq, d_model)
        out_T2 = torch.matmul(att_T2, V_pro_T2).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                                  self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_T2 = self.resid_drop(self.out_proj_T2(out_T2))  # (b_s, nq, d_model)

        return [out_T1, out_T2]


class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_T1 = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_T2 = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 # nn.SiLU(),  # changed from GELU
                                 nn.GELU(),  # changed from GELU
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        T1_flat = x[0]
        T2_flat = x[1]
        assert T1_flat.shape[0] == T2_flat.shape[0]
        bs, nx, c = T1_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            # with Learnable Coefficient
            T1_out, T2_out = self.crossatt([T1_flat, T2_flat])
            T1_att_out = self.coefficient1(T1_flat) + self.coefficient2(T1_out)
            T2_att_out = self.coefficient3(T2_flat) + self.coefficient4(T2_out)
            T1_flat = self.coefficient5(T1_att_out) + self.coefficient6(self.mlp_T1(self.LN2(T1_att_out)))
            T2_flat = self.coefficient7(T2_att_out) + self.coefficient8(self.mlp_T2(self.LN2(T2_att_out)))

            # without Learnable Coefficient
            # rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            # rgb_att_out = rgb_fea_flat + rgb_fea_out
            # ir_att_out = ir_fea_flat + ir_fea_out
            # rgb_fea_flat = rgb_att_out + self.mlp_vis(self.LN2(rgb_att_out))
            # ir_fea_flat = ir_att_out + self.mlp_ir(self.LN2(ir_att_out))

        return [T1_flat, T2_flat]


class TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_S=16, horz_S=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1,
                 attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_S = vert_S
        self.horz_S = horz_S
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.positional_embedding_T1 = nn.Parameter(torch.zeros(1, vert_S * horz_S, self.n_embd))
        self.positional_embedding_T2 = nn.Parameter(torch.zeros(1, vert_S * horz_S, self.n_embd))

        # downsampling
        # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))

        self.avgpool = AdaptivePool2d(self.vert_S, self.horz_S, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_S, self.horz_S, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        self.crosstransformer = nn.Sequential(
            *[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in
              range(n_layer)])

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        T1_fea = x[0]
        T2_fea = x[1]
        assert T1_fea.shape[0] == T2_fea.shape[0]
        bs, c, h, w = T1_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        # new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_T1_fea = self.vis_coefficient(self.avgpool(T1_fea), self.maxpool(T1_fea))
        new_c, new_h, new_w = new_T1_fea.shape[1], new_T1_fea.shape[2], new_T1_fea.shape[3]
        T1_fea_flat = new_T1_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis

        # new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_T2_fea = self.ir_coefficient(self.avgpool(T2_fea), self.maxpool(T2_fea))
        T2_fea_flat = new_T2_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        T1_fea_flat, T2_fea_flat = self.crosstransformer([T1_fea_flat, T2_fea_flat])

        T1_fea_CFE = T1_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            T1_fea_CFE = F.interpolate(T1_fea_CFE, size=([h, w]), mode='nearest')
        else:
            T1_fea_CFE = F.interpolate(T1_fea_CFE, size=([h, w]), mode='bilinear')
        new_T1_fea = T1_fea_CFE + T1_fea
        T2_fea_CFE = T2_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            T2_fea_CFE = F.interpolate(T2_fea_CFE, size=([h, w]), mode='nearest')
        else:
            T2_fea_CFE = F.interpolate(T2_fea_CFE, size=([h, w]), mode='bilinear')
        new_T2_fea = T2_fea_CFE + T2_fea

        new_fea = self.concat([new_T1_fea, new_T2_fea])
        new_fea = self.conv1x1_out(new_fea)

        return new_fea


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (
            input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y

