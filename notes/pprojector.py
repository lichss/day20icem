import torch
import torch.nn as nn

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

# 示例使用

