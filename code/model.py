import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

from miscc.config import cfg
from attention import MultiSpatialAttention, MultiChannelAttention

class NetGOrigin(nn.Module):
    def __init__(self, ngf=64, nz=100,lstm = None):
        super(NetGOrigin, self).__init__()
        self.ngf = ngf
        #self.rnn = rnn
        self.lstm = lstm
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.fc = nn.Linear(nz, ngf*8*4*4)
        self.block0 = G_Block(ngf * 8, ngf * 8,lstm)#4x4
        self.block1 = G_Block(ngf * 8, ngf * 8,lstm)#4x4
        self.block2 = G_Block(ngf * 8, ngf * 8,lstm)#8x8
        self.block3 = G_Block(ngf * 8, ngf * 8,lstm)#16x16
        self.block4 = G_Block(ngf * 8, ngf * 4,lstm)#32x32
        self.block5 = G_Block(ngf * 4, ngf * 2,lstm)#64x64
        self.block6 = G_Block(ngf * 2, ngf * 1,lstm)#128x128

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, c):

        out = self.fc(x)
        out = out.view(x.size(0), 8*self.ngf, 4, 4)
        out = self.block0(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block1(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block2(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block3(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block4(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out,c)

        out = self.conv_img(out)

        return out

class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100,lstm = None):
        super(NetG, self).__init__()
        self.ngf = ngf
        self.ef_dim = cfg.TEXT.EMBEDDING_DIM
        #self.rnn = rnn
        self.lstm = lstm
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.fc = nn.Linear(nz, ngf*8*4*4)
        self.block0 = G_Block(ngf * 8, ngf * 8,lstm)#4x4
        self.block1 = G_Block(ngf * 8, ngf * 8,lstm)#4x4
        self.block2 = G_Block(ngf * 8, ngf * 8,lstm)#8x8
        self.block3 = G_Block(ngf * 8, ngf * 8,lstm)#16x16
        self.block4 = G_Block(ngf * 8, ngf * 4,lstm)#32x32
        self.block5 = G_Block(ngf * 4, ngf * 2,lstm)#64x64
        #self.block6 = G_Block(ngf * 2, ngf * 1,lstm)#128x128
        self.block6 = G_Block(ngf * 2 * 3, ngf * 1, lstm)  # attention added, 128x128

        #####20220601 idea-2 add word attention #####
        self.att = MultiSpatialAttention(ngf * 2, self.ef_dim, 1)
        self.channel_att = MultiChannelAttention(ngf * 2, self.ef_dim, 128, 1)
        #####20220601 idea-2 add word attention #####

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, c, word_embs, mask):

        out = self.fc(x)
        out = out.view(x.size(0), 8*self.ngf, 4, 4)
        out = self.block0(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block1(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block2(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block3(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block4(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out,c)

        #####20220601 idea-2 add word attention #####
        self.att.applyMask(mask)
        out_code128 = out
        c_code, att = self.att(out_code128, word_embs)
        c_code_channel, att_channel = self.channel_att(c_code, word_embs, out_code128.size(2), out_code128.size(3))
        c_code = c_code.view(word_embs.size(0), -1, out_code128.size(2), out_code128.size(3))
        h_c_code = torch.cat((out_code128, c_code), 1)
        h_c_c_code = torch.cat((h_c_code, c_code_channel), 1)
        out = h_c_c_code
        #####20220601 idea-2 add word attention #####

        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out,c)

        out = self.conv_img(out)

        return out, att

class G_Block(nn.Module):

    def __init__(self, in_ch, out_ch,lstm):
        super(G_Block, self).__init__()
        self.lstm = lstm

        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch)
        self.affine1 = affine(in_ch)
        self.affine4 = affine(in_ch)
        self.affine2 = affine(out_ch)
        self.affine3 = affine(out_ch)
        self.affine5 = affine(out_ch)
        self.fea_l = nn.Linear(in_ch,256)
        self.fea_ll = nn.Linear(out_ch,256)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, yy=None):

        lstm_input = yy
        y,_  =  self.lstm(lstm_input)
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        

        lstm_input = yy
        y,_  =  self.lstm(lstm_input)        
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        
        
        
        h = self.c1(h)
        
 
        lstm_input = yy
        y,_  =  self.lstm(lstm_input)
        
        
        h = self.affine2(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)

        lstm_input = yy
        y,_  =  self.lstm(lstm_input)
        
        
        h = self.affine3(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
          
        return self.c2(h)



class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
   
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias



class D_GET_LOGITS_att(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS_att, self).__init__()
        self.df_dim = ndf

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16+256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )
        self.block = resD(ndf * 16+256, ndf * 16)#4

        self.joint_conv_att = nn.Sequential(
            nn.Conv2d(ndf * 16+256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )        
        self.softmax= nn.Softmax(2)
    def forward(self, out, y_):

        y = y_.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 8, 8)
        h_c_code = torch.cat((out, y), 1)
        p = self.joint_conv_att(h_c_code)        
        p = self.softmax(p.view(-1,1,64))
        p = p.reshape(-1,1,8,8)
        self.p = p
        p = p.repeat(1, 256, 1, 1)
        y = torch.mul(y,p)  
        h_c_code = torch.cat((out, y), 1)
        h_c_code = self.block(h_c_code)

        y = y_.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)        
        h_c_code = torch.cat((h_c_code, y), 1)
        out = self.joint_conv(h_c_code)
        return out





# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)#128
        self.block0 = resD(ndf * 1, ndf * 2)#64
        self.block1 = resD(ndf * 2, ndf * 4)#32
        self.block2 = resD(ndf * 4, ndf * 8)#16
        self.block3 = resD(ndf * 8, ndf * 16)#8
        self.block4 = resD(ndf * 16, ndf * 16)#4
        self.block5 = resD(ndf * 16, ndf * 16)#4

        self.COND_DNET = D_GET_LOGITS_att(ndf)

    def forward(self,x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        return out




class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)












