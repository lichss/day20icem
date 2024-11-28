import torch
import torch.nn as nn

@torch.jit.script
def exp_attractor(dx, alpha: float = 300, gamma: int = 2):
    """Exponential attractor: dc = exp(-alpha*|dx|^gamma) * dx , where dx = a - c, a = attractor point, c = bin center, dc = shift in bin centermmary for exp_attractor

    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength. Lower alpha = greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly number of bin centers affected. Lower gamma = farther reach. Defaults to 2.

    Returns:
        torch.Tensor : Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return torch.exp(-alpha*(torch.abs(dx)**gamma)) * (dx)


@torch.jit.script
def inv_attractor(dx, alpha: float = 300, gamma: int = 2):
    """Inverse attractor: dc = dx / (1 + alpha*dx^gamma), where dx = a - c, a = attractor point, c = bin center, dc = shift in bin center
    This is the default one according to the accompanying paper. 

    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength. Lower alpha = greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly number of bin centers affected. Lower gamma = farther reach. Defaults to 2.

    Returns:
        torch.Tensor: Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return dx.div(1+alpha*dx.pow(gamma))

class XcsSeedBinRegressorUnnormed(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):
        """Bin center regressor network. Bin centers are unbounded

        Args:
            in_features (int): input channels
            n_bins (int, optional): Number of bin centers. Defaults to 16.
            mlp_dim (int, optional): Hidden dimension. Defaults to 256.
            min_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
            max_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
        """
        super().__init__()
        self.version = "1_1"
        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.BatchNorm2d(mlp_dim),  # 批归一化
            nn.ReLU(inplace=True),
            SELayer(mlp_dim),  # 添加 SE 模块
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.BatchNorm2d(n_bins),  # 批归一化
            nn.Softplus()
        )

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B_centers = self._net(x)
        return B_centers, B_centers


class PositionWiseFeedForward1(nn.Module):
    def __init__(self, in_features, out_features, mlp_dim=128, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        # in_features: 输入通道数
        # out_features: 输出通道数
        # mlp_dim: 中间层的维度
        self.conv1 = nn.Conv2d(in_features, mlp_dim, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout2d(dropout)  # 注意这里使用了Dropout2d以适应二维输入
        self.conv2 = nn.Conv2d(mlp_dim, out_features, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, in_features, height, width]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        # d_model: 输入和输出的维度（相当于特征图的通道数）
        # d_ff: 中间层的维度
        self.linear1 = nn.Linear(d_model, d_ff)
        # self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x

class CSAttention(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CSAttention, self).__init__()
        self.se = SELayer(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.se(x)
        x = self.sa(x)
        return x

class XcsProjector0(nn.Module):
    def __init__(self, in_features, out_features, mlp_dim=128, dropout_rate=0.2):
        """改进后的Projector MLP

        Args:
            in_features (int): 输入特征图的通道数
            out_features (int): 输出特征图的通道数
            mlp_dim (int, optional): 中间隐藏层的维度,默认为128
            dropout_rate (float, optional): Dropout概率,默认为0.2
        """
        super().__init__()

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),  # 第一个1x1卷积层
            nn.BatchNorm2d(mlp_dim),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Dropout2d(p=dropout_rate),  # Dropout层
            CSAttention(mlp_dim),  # 添加Channel-wise and Spatial-wise Attention
            nn.Conv2d(mlp_dim, out_features, 1, 1, 0)  # 第二个1x1卷积层
        )

    def forward(self, x):
        return self._net(x)

class XcsProjector1(nn.Module):
    def __init__(self, in_features, out_features, mlp_dim=128, dropout_rate=0.2):
        """改进后的Projector MLP

        Args:
            in_features (int): 输入特征图的通道数
            out_features (int): 输出特征图的通道数
            mlp_dim (int, optional): 中间隐藏层的维度,默认为128
            dropout_rate (float, optional): Dropout概率,默认为0.2
        """
        super().__init__()

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),  # 第一个1x1卷积层
            # 暂时不要 # nn.BatchNorm2d(mlp_dim),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
            # 暂时不要 # nn.Dropout2d(p=dropout_rate),  # Dropout层
            CSAttention(mlp_dim),  # 添加Channel-wise and Spatial-wise Attention
            nn.Conv2d(mlp_dim, out_features, 1, 1, 0)  # 第二个1x1卷积层
        )

    def forward(self, x):
        return self._net(x)

## 我要狠狠的改你

class AttractorLayerUnnormed(nn.Module):
    def __init__(self, in_features, n_bins, n_attractors=16, mlp_dim=128, min_depth=1e-3, max_depth=10,
                 alpha=300, gamma=2, kind='sum', attractor_type='exp', memory_efficient=False):
        """
        Attractor layer for bin centers. Bin centers are unbounded
        """
        super().__init__()

        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.attractor_type = attractor_type
        self.memory_efficient = memory_efficient

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_attractors, 1, 1, 0),
            nn.Softplus()
        )

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        Args:
            x (torch.Tensor) : feature block; shape - n, c, h, w
            b_prev (torch.Tensor) : previous bin centers normed; shape - n, prev_nbins, h, w
        
        Returns:
            tuple(torch.Tensor,torch.Tensor) : new bin centers unbounded; shape - n, nbins, h, w. Two outputs just to keep the API consistent with the normed version
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(
                    prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding

        A = self._net(x)
        n, c, h, w = A.shape

        b_prev = nn.functional.interpolate(
            b_prev, (h, w), mode='bilinear', align_corners=True)
        b_centers = b_prev

        if self.attractor_type == 'exp':
            dist = exp_attractor
        else:
            dist = inv_attractor

        if not self.memory_efficient:   # False
            func = {'mean': torch.mean, 'sum': torch.sum}[self.kind]
            # .shape N, nbins, h, w
            delta_c = func(
                dist(A.unsqueeze(2) - b_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(b_centers, device=b_centers.device)
            for i in range(self.n_attractors):
                delta_c += dist(A[:, i, ...].unsqueeze(1) -
                                b_centers)  # .shape N, nbins, h, w

            if self.kind == 'mean':   #True
                delta_c = delta_c / self.n_attractors

        b_new_centers = b_centers + delta_c
        B_centers = b_new_centers

        return b_new_centers, B_centers

