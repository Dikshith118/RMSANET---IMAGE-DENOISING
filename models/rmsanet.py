import torch
import torch.nn as nn
from .blocks import RDB, MSFE, EAM # Imports the blocks from the same directory

class RMSANet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RMSANet, self).__init__()
        
        # 1. Shallow Feature Extraction: in_channels -> 64 filters
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Backbone (RIR Groups): 4 RDBs as requested
        self.rir_groups = nn.Sequential(
            RDB(64, growth_rate=32, num_layers=4),
            RDB(64, growth_rate=32, num_layers=4),
            RDB(64, growth_rate=32, num_layers=4),
            RDB(64, growth_rate=32, num_layers=4)
        )
        
        # 3. Multi-Scale Feature Extraction
        self.msfe = MSFE(64)
        
        # 4. Enhanced Attention Module
        self.eam = EAM(64)
        
        # 5. Feature Fusion: Combining and scaling features to 128 filters
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 3, 128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        
        # 6. Reconstruction and Residual Learning: Increased filters
        self.reconstruction = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x_noisy):
        shallow_features = self.shallow_conv(x_noisy)
        rir_out = self.rir_groups(shallow_features)
        
        eam_out = self.eam(rir_out)
        
        fused_features = torch.cat([shallow_features, rir_out, eam_out], dim=1)
        fusion_out = self.fusion(fused_features)
        
        predicted_noise = self.reconstruction(fusion_out)
        
        denoised_image = x_noisy - predicted_noise
        
        return denoised_image, predicted_noise
