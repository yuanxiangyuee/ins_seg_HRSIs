import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import datetime

__all__ = ['Carafe']
class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN_ReLU tuple'''
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                input_channel, output_channel, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class KernelPredeictionModule(nn.Module):

    def __init__(self, input_channel, channel_mid=64, kernel_up=5, kernel_encoder=3, enlarge_rate=2):
        super(KernelPredeictionModule,self).__init__()
        self.input_channel = input_channel
        self.channel_mid = channel_mid
        self.kernel_up = kernel_up
        self.kernel_encoder = kernel_encoder
        self.enlarge_rate = enlarge_rate
        self.channel_compressor = ConvBNReLU(input_channel, channel_mid, kernel_size=1, stride=1, padding=0, dilation=1)
        self.context_encoder = ConvBNReLU(channel_mid, (self.enlarge_rate*self.kernel_up)**2,
                                          kernel_size=self.kernel_encoder, stride=1, padding=self.kernel_encoder//2,
                                          dilation=1, use_relu=False)
        # self.pix_shf = F.pixel_shuffle(self.enlarge_rate)
        self.kernel_normalizer = nn.Softmax(dim=1)
    def forward(self, x):
        # start_time = datetime.datetime.now()
        x = self.channel_compressor(x)
        x = self.context_encoder(x)
        x = F.pixel_shuffle(x,self.enlarge_rate)
        x = self.kernel_normalizer(x)
        # print("KP cost:{}".format(datetime.datetime.now() - start_time))
        return x

class Carafe(nn.Module):
    def __init__(self, input_channel, channel_cm=64, kernel_up=5, kernel_encoder=3, enlarge_rate=2):
        """
        The Carafe upsample model(unoffical)
        :param input_channel: The channel of input
        :param channel_cm:    The channel of Cm, paper give this parameter 64
        :param kernel_up:     The kernel up, paper give this parameter 5
        :param kernel_encoder:The kernel encoder, paper suggest it kernel_up-2, so 3 here
        :param enlarge_rate:  The enlarge rate , your rate for upsample (2x usually)
        """
        super(Carafe, self).__init__()
        self.kernel_up = kernel_up
        self.enlarge_rate = enlarge_rate
        self.KPModule = KernelPredeictionModule(input_channel,channel_cm,kernel_up,kernel_encoder,enlarge_rate)
        self.upsmp = nn.Upsample(scale_factor=self.enlarge_rate, mode='nearest')
        # self.unfold = nn.Unfold(kernel_size=self.kernel_up, dilation=self.enlarge_rate,
        #                         padding=kernel_up//2*self.enlarge_rate)
        self.unfold = nn.Unfold(kernel_size=self.kernel_up, dilation=1,
                                padding=kernel_up // 2, stride=1)

    def forward(self, x):

        # KernelPredeictionModule : cost 0.7175s
        # start_time = datetime.datetime.now()
        kpresult = self.KPModule(x) # (b,kup*kup,e_w,e_h)
        # print("time cost:{}".format(datetime.datetime.now() - start_time))

        ############Context-aware Reassembly Module########################
        ######## Step1 formal_pic deal : cost 0.1164s
        # start_time = datetime.datetime.now()
        x_mat = self.generate_kup_mat(x) # (b,c,kup*kup,w,h)
        # print("time cost:{}".format(datetime.datetime.now() - start_time))

        ######## Step2 kernel mul : cost 0.0009s
        # start_time = datetime.datetime.now()
        output = x_mat * (kpresult.unsqueeze(1))
        # print("time cost:{}".format(datetime.datetime.now() - start_time))

        ######## Step3 sum the kup dim : cost 0.0002s
        # start_time = datetime.datetime.now()
        output = torch.sum(output, dim=2)
        # print("time cost:{}".format(datetime.datetime.now() - start_time))

        # output = torch.einsum('bkhw,bckhw->bchw', [kpresult, x_mat])
        return output

    def generate_kup_mat(self,x):
        """
        generate the mat matrix, make a new dim kup for mul
        :param x:(batch,channel,w,h)
        :return: (batch,channel,kup*kup,enlarged_w,enlarged_h)
        """
        batch, channel, w ,h = x.shape
        w_out, h_out = w*self.enlarge_rate, h*self.enlarge_rate
        # stride to sample
        r = int(self.kernel_up / 2)
        # # pad the x to stride
        # start_time = datetime.datetime.now()
        # pad = F.pad(x, (r, r, r, r))
        # x_mat = torch.zeros((batch, channel, self.kernel_up**2 , w, h),device=x.device)
        # print("formal cost:{}".format(datetime.datetime.now() - start_time))
        # start_time = datetime.datetime.now()
        # for i in range(w):
        #     for j in range(h):
        #         pad_x = i + r
        #         pad_y = j + r
        #         x_mat[:, :, :, i, j] = pad[:, :, pad_x - r:pad_x + r + 1, pad_y - r:pad_y + r + 1]\
        #             .reshape(batch, channel, -1)
        # print("for cost:{}".format(datetime.datetime.now() - start_time))
        # x_mat = x_mat.repeat(1, 1, 1, self.enlarge_rate, self.enlarge_rate)
        # # each part of the stride part the same!

        # get the dim kup**2 with unfold with the stride windows
        x_mat = self.upsmp(x)
        x_mat = self.unfold(x_mat)
        # make the result to (b,c,kup**2,w,h)
        x_mat = x_mat.view((batch, channel, -1, w*self.enlarge_rate, h*self.enlarge_rate))
        # # nearest inter the number for i map the region [i:i/enlarge,j:j/enlarge]
        # x_mat_size = [x_mat.size(i) for i in range(len(x_mat.size()))]
        # x_mat_upsmp = self.upsmp(x_mat)
        return x_mat

    # def repeat_kernel(self,weight,channel):
    #     """
    #     Generate the channel dim for the weight
    #     repeat the Kernel Prediction Module output for channel times,
    #     and it can be mul just like the depth-width conv (The repeat on the batch dim)
    #     :param weight:  (batch,kup*kup,enlarged_w,enlarged_h)
    #     :param channel: the channel num to repeat
    #     :return: (batch,channel,kup*kup,enlarged_w,enlarged_h)
    #     """
    #     batch, kup_2, w, h = weight.shape
    #     # copy the channel in batch
    #     w_mat = torch.stack([i.expand(channel, kup_2, w, h) for i in weight])
    #     # each channel in batch is the same!
    #     # print(torch.equal(w_mat[0, 0, ...], w_mat[0, 1, ...]))
    #     return w_mat


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # model = Carafe(input_channel=16,channel_cm=64).cuda()
    # x = torch.rand((1,16,24,24)).cuda()
    # # from model_summary import summary
    # # summary(model,x)
    #
    # model = model.cuda()
    # x = x.cuda()
    # # model(x)
    # start_time = datetime.datetime.now()
    # out = model(x)
    # print("time cost:{}".format(datetime.datetime.now()-start_time))
    # print(out.shape)
    x = torch.rand(1, 16, 24, 24)
    carafe = Carafe(x.size(1))
    # start_time = datetime.datetime.now()
    oup = carafe(x)
    # print("time cost:{}".format(datetime.datetime.now()-start_time))
    print(oup.size())