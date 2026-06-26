import torch
import torch.nn as nn
import torch.nn.functional as F


class PIMSRModel(nn.Module):
    """Physics-Informed Multimodal Super-Resolution (PIMSR) model
    
    For fusing low-resolution DEM with high-resolution SAR data
    """
    
    def __init__(self, in_channels_dem=1, in_channels_sar=1, out_channels=1, upscale_factor=4):
        """Initialize PIMSR model
        
        Args:
            in_channels_dem: Number of DEM channels
            in_channels_sar: Number of SAR channels
            out_channels: Number of output channels
            upscale_factor: Upscaling factor (2x, 4x, 8x)
        """
        super().__init__()
        self.upscale_factor = upscale_factor
        self.in_channels_dem = in_channels_dem
        self.in_channels_sar = in_channels_sar
        
        # DEM feature extraction
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(in_channels_dem, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        # SAR feature extraction
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(in_channels_sar, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        # Multi-modal fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1, 1, 0),
        )
        
        # Super-resolution decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling layers
        self.upsample_layers = self._build_upsample_layers()
        
        # Final reconstruction
        self.reconstruction = nn.Conv2d(256, out_channels, 3, 1, 1)
    
    def _build_upsample_layers(self):
        """Build upsampling layers based on upscale factor"""
        layers = []
        if self.upscale_factor == 4:
            layers.extend([
                nn.Conv2d(256, 256 * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 256 * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ])
        elif self.upscale_factor == 2:
            layers.extend([
                nn.Conv2d(256, 256 * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ])
        return nn.Sequential(*layers)
    
    def forward(self, dem_lr, sar_hr):
        """Forward pass for DEM fusion
        
        Args:
            dem_lr: Low-resolution DEM tensor (B, 1, H, W)
            sar_hr: High-resolution SAR tensor (B, 1, H*upscale, W*upscale)
            
        Returns:
            Super-resolved DEM of shape (B, 1, H*upscale, W*upscale)
        """
        # Upsample LR DEM to match SAR resolution
        dem_lr_up = F.interpolate(dem_lr, size=sar_hr.shape[-2:], mode='bilinear', align_corners=False)
        
        # Extract features
        dem_features = self.dem_encoder(dem_lr_up)
        sar_features = self.sar_encoder(sar_hr)
        
        # Fuse multi-modal features
        fused = torch.cat([dem_features, sar_features], dim=1)
        fused = self.fusion(fused)
        
        # Decode
        decoded = self.decoder(fused)
        
        # Upsample
        upsampled = self.upsample_layers(decoded)
        
        # Reconstruct
        dem_sr = self.reconstruction(upsampled)
        
        return dem_sr
