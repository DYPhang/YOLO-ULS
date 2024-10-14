import torch
from torch import nn
import torch.nn.functional as F
import math
from ..modules.conv import Conv


__all__ = ['EMA', 'CoordAtt','GCCA', 'MLCA']

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out



class GCCA(nn.Module):
    def __init__(self, ch) -> None:
        super().__init__()

        self.GLOBAL = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(ch, ch)
        )

        self.ph = nn.AdaptiveAvgPool2d((None, 1))
        self.pw = nn.AdaptiveAvgPool2d((1, None))
        self.hw = Conv(ch, ch, (3, 1))

        self.phw = Conv(ch, ch, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        x_ph, x_pw, x_ch = self.ph(x), self.pw(x).permute(0, 1, 3, 2), self.GLOBAL(x)

        x_hw = torch.cat([x_ph, x_pw], dim=2)
        x_hw = self.hw(x_hw)
        x_ph, x_pw = torch.split(x_hw, [h, w], dim=2)
        hw_weight = self.phw(x_hw).sigmoid()
        h_weight, w_weight = torch.split(hw_weight, [h, w], dim=2)
        x_ph, x_pw = x_ph * h_weight, x_pw * w_weight
        x_ch = x_ch * torch.mean(hw_weight, dim=2, keepdim=True)

        return x * x_ph.sigmoid() * x_pw.permute(0, 1, 3, 2).sigmoid() * x_ch.sigmoid()


# class GCCA(nn.Module):
#     def __init__(self, ch) -> None:
#         super().__init__()
#
#         self.GLOBAL = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             Conv(ch, ch)
#         )
#
#         self.ph = nn.AdaptiveAvgPool2d((None, 1))
#         self.pw = nn.AdaptiveAvgPool2d((1, None))
#         self.hw = Conv(ch, ch, (3, 1))
#
#         self.phw = Conv(ch, ch, 1)
#
#     def forward(self, x):
#         _, _, h, w = x.size()
#         x_ph, x_pw, x_ch = self.ph(x), self.pw(x).permute(0, 1, 3, 2), self.GLOBAL(x)
#
#         x_hw = torch.cat([x_ph, x_pw], dim=2)
#         x_hw = self.hw(x_hw)
#         x_ph, x_pw = torch.split(x_hw, [h, w], dim=2)
#         hw_weight = self.phw(x_hw).sigmoid()
#         h_weight, w_weight = torch.split(hw_weight, [h, w], dim=2)
#         x_ph, x_pw = x_ph * h_weight, x_pw * w_weight
#         x_ch = torch.mean(hw_weight, dim=2, keepdim=True)
#
#         return x * x_ph.sigmoid() * x_pw.permute(0, 1, 3, 2).sigmoid() * x_ch.sigmoid()

class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)

        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])

        x=x * att_all
        return x


