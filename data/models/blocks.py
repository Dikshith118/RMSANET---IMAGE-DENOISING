import torch
import torch.nn as nn
import torch.nn.functional as F

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.conv1x1 = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        
        fused_features = torch.cat(features, 1)
        out = self.conv1x1(fused_features)
        return out + x # Residual connection

class MSFE(nn.Module):
    def __init__(self, in_channels):
        super(MSFE, self).__init__()
        self.branch_a = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.branch_b = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.branch_c = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        
        self.fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
    
    def forward(self, x):
        branch_a_out = self.branch_a(x)
        branch_b_out = self.branch_b(x)
        branch_c_out = self.branch_c(x)
        
        combined = torch.cat([branch_a_out, branch_b_out, branch_c_out], dim=1)
        return self.fusion(combined)

class EAM(nn.Module):
    def __init__(self, in_channels):
        super(EAM, self).__init__()
        # Channel Attention (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        channel_out = self.channel_attention(x)
        x = x * channel_out
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.spatial_attention(spatial_in)
        
        return x * spatial_out
