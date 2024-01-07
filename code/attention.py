import torch
import torch.nn as nn
from miscc.config import cfg

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1):

    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    attn = torch.bmm(contextT, query) 
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)  
    attn = attn.view(batch_size, sourceL, queryL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class SpatialAttention(nn.Module):
    def __init__(self, idf, cdf):
        super(SpatialAttention, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax()
        self.mask = None
        self.idf = idf

    def applyMask(self, mask):
        self.mask = mask  

    def forward(self, input, context):

        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = context.unsqueeze(3)

        sourceT = self.conv_context(sourceT).squeeze(3)

        attn = torch.bmm(targetT, sourceT)
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        attn = self.sm(attn)  
        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()

        weightedContext = torch.bmm(sourceT, attn)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn

class ChannelAttention(nn.Module):
    def __init__(self, idf, cdf, size):
        super(ChannelAttention, self).__init__()
        self.conv_context = conv1x1(cdf, size*size)
        self.sm = nn.Softmax()
        self.idf = idf

    def forward(self, weightedContext, context, ih, iw):

        batch_size, sourceL = context.size(0), context.size(2)
        sourceC = context.unsqueeze(3)
        sourceC = self.conv_context(sourceC).squeeze(3) 
        attn_c = torch.bmm(weightedContext, sourceC)
        attn_c = attn_c.view(batch_size * self.idf, sourceL)
        attn_c = self.sm(attn_c)
        attn_c = attn_c.view(batch_size, self.idf, sourceL)
        
        attn_c = torch.transpose(attn_c, 1, 2).contiguous()
        weightedContext_T = torch.bmm(sourceC, attn_c)
        weightedContext_T = torch.transpose(weightedContext_T, 1, 2).contiguous()
        weightedContext_T = weightedContext_T.view(batch_size, -1, ih, iw)

        return weightedContext_T, attn_c

#MAGAN: Multi-attention Generative Adversarial Networks for Text-to-Image Generation Implementation
class MultiSpatialAttention(nn.Module):
    def __init__(self, idf, cdf, num_head):
        super(MultiSpatialAttention, self).__init__()
        self.num_head = num_head

        self.conv_context = []
        for i in range(num_head):
            cov = conv1x1(cdf, idf)
            self.conv_context.append(cov)

        self.sm = nn.Softmax()
        self.mask = None
        self.idf = idf

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):

        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        target = input.view(batch_size, -1, queryL)
        targetTIn = torch.transpose(target, 1, 2).contiguous()
        sourceTIn = context.unsqueeze(3)

        weightedContextOut = []
        attnOut = []

        for i in range(self.num_head):
            conv_context = self.conv_context[i]
            if cfg.CUDA:
                conv_context = conv_context.cuda()

            sourceT = conv_context(sourceTIn).squeeze(3)

            attn = torch.bmm(targetTIn, sourceT)
            attn = attn.view(batch_size*queryL, sourceL)

            if self.mask is not None:
                mask = self.mask.repeat(queryL, 1)
                attn.data.masked_fill_(mask.data, -float('inf'))

            attn = self.sm(attn)
            attn = attn.view(batch_size, queryL, sourceL)
            attn = torch.transpose(attn, 1, 2).contiguous()

            weightedContext = torch.bmm(sourceT, attn)
            attn = attn.view(batch_size, -1, ih, iw)

            if i == 0:
                weightedContextOut = weightedContext
                attnOut = attn
            else:
                weightedContextOut = weightedContext + weightedContextOut
                attnOut = attn + attnOut

        return weightedContextOut, attnOut

class MultiChannelAttention(nn.Module):
    def __init__(self, idf, cdf, size, num_head):
        super(MultiChannelAttention, self).__init__()
        self.num_head = num_head

        self.conv_context = []
        for i in range(num_head):
            cov = conv1x1(cdf, size * size)
            self.conv_context.append(cov)

        self.sm = nn.Softmax()
        self.idf = idf

    def forward(self, weightedContext, context, ih, iw):
        batch_size, sourceL = context.size(0), context.size(2)
        sourceCIn = context.unsqueeze(3)

        weightedContextOut = []
        attnOut = []

        for i in range(self.num_head):
            conv_context = self.conv_context[i]
            if cfg.CUDA:
                conv_context = conv_context.cuda()

            sourceC = conv_context(sourceCIn).squeeze(3)
            attn_c = torch.bmm(weightedContext, sourceC)
            attn_c = attn_c.view(batch_size * self.idf, sourceL)
            attn_c = self.sm(attn_c)
            attn_c = attn_c.view(batch_size, self.idf, sourceL)

            attn_c = torch.transpose(attn_c, 1, 2).contiguous()
            weightedContext_T = torch.bmm(sourceC, attn_c)
            weightedContext_T = torch.transpose(weightedContext_T, 1, 2).contiguous()
            weightedContext_T = weightedContext_T.view(batch_size, -1, ih, iw)

            if i == 0:
                weightedContextOut = weightedContext_T
                attnOut = attn_c
            else:
                weightedContextOut = weightedContext_T + weightedContextOut
                attnOut = attn_c + attnOut

        return weightedContextOut, attnOut
