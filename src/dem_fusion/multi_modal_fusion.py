import torch
import torch.nn as nn


class MultiModalFusion(nn.Module):
    """Multi-modal fusion layer for DEM and SAR"""
    
    def __init__(self, dim=128, reduction=4):
        """Initialize fusion module
        
        Args:
            dim: Feature dimension
            reduction: Channel reduction ratio
        """
        super().__init__()
        
        # Channel attention for DEM
        self.dem_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        
        # Channel attention for SAR
        self.sar_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial fusion
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
    
    def forward(self, dem_feat, sar_feat):
        """Fuse DEM and SAR features
        
        Args:
            dem_feat: DEM features
            sar_feat: SAR features
            
        Returns:
            Fused features
        """
        # Apply channel attention
        dem_att = self.dem_attention(dem_feat)
        sar_att = self.sar_attention(sar_feat)
        
        dem_feat_att = dem_feat * dem_att
        sar_feat_att = sar_feat * sar_att
        
        # Spatial fusion
        fused = torch.cat([dem_feat_att, sar_feat_att], dim=1)
        fused = self.spatial_fusion(fused)
        
        return fused
