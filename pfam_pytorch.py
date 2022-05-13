


from torch import nn


class PFAMLayer(nn.Module):
    def __init__(self):
        super(AutoAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        z = torch.mean(x,1)
        return self.sigmoid(y*z)
    
