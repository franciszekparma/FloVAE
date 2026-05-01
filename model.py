import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleDownConv(nn.Module):
  def __init__(self, c_in, c_out):
    super().__init__()
    
    self.double_conv = nn.Sequential(
      nn.Conv2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True
      ),
      nn.GroupNorm(8, c_out),
      nn.GELU(),
      
      nn.Conv2d(
        in_channels=c_out,
        out_channels=c_out,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True
      ),
      nn.GroupNorm(8, c_out),
      nn.GELU(),
      
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
  def forward(self, X):
    return self.double_conv(X)


class DoubleUpConv(nn.Module):
  def __init__(self, c_in, c_out, last=False):
    super().__init__()
    
    if not last:
      self.double_conv = nn.Sequential(
        nn.Conv2d(
          in_channels=c_in,
          out_channels=c_out,
          kernel_size=3,
          padding=1,
          stride=1,
          bias=True
        ),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
        
        nn.Conv2d(
          in_channels=c_out,
          out_channels=c_out,
          kernel_size=3,
          padding=1,
          stride=1,
          bias=True
        ),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
        
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.GroupNorm(8, c_out)
      )
    else:
      self.double_conv = nn.Sequential(
          nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True
          ),
          nn.GroupNorm(3, 3),
          nn.GELU(),
          
          nn.Conv2d(
            in_channels=c_out,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True
          ),
          nn.GroupNorm(3, 3),
          nn.GELU(),
          
          nn.ConvTranspose2d(
            in_channels=c_out,
            out_channels=c_out,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True
          ),
          nn.GroupNorm(1, 3)
        )
    
  def forward(self, X):
    return self.double_conv(X)


class VAE(nn.Module):
  def __init__(self, num_layers=4, hid_c_dim=256, z_dim=80):
    super().__init__()
    
    self.encoder = nn.Sequential(
      DoubleDownConv(3, 32),
      DoubleDownConv(32, 64),
      DoubleDownConv(64, 128),
      DoubleDownConv(128, 256)
    )
    
    self.last_encoder_layer = nn.Sequential(
      nn.GroupNorm(8, 256),
      
      nn.Conv2d(
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=False
      ),
      nn.GroupNorm(8, 256),
      nn.GELU(),
      
      nn.Conv2d(
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True
      )
    )
    
    self.hid_2_mean = nn.Sequential(
      nn.Conv2d(
        in_channels=256,
        out_channels=128,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True
      ),
      nn.GroupNorm(8, 128),
      nn.GELU(),
      
      nn.AdaptiveAvgPool2d((4, 4)),
      nn.Flatten(),
      
      nn.Linear(2048, z_dim * 4),
      nn.LayerNorm(z_dim * 4),
      nn.GELU(),
      
      nn.Linear(z_dim * 4, z_dim)
    )
    
    self.hid_2_logvar = nn.Sequential(
      nn.Conv2d(
        in_channels=256,
        out_channels=128,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True
      ),
      nn.GroupNorm(8, 128),
      nn.GELU(),
      
      nn.AdaptiveAvgPool2d((4, 4)),
      nn.Flatten(),
      
      nn.Linear(2048, z_dim * 4),
      nn.LayerNorm(z_dim * 4),
      nn.GELU(),
      
      nn.Linear(z_dim * 4, z_dim)
    )
    
    self.z_2_hid = nn.Sequential(
      nn.Linear(z_dim, (128//2**4)**2 * 128),
      nn.Unflatten(1, (128, 8, 8)),
      
      nn.Conv2d(
        in_channels=128,
        out_channels=256,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True
      ),
      nn.GroupNorm(8, 256),
      nn.GELU()
    )
    
    self.decoder = nn.Sequential(
      DoubleUpConv(256, 128),
      DoubleUpConv(128, 64),
      DoubleUpConv(64, 32),
      DoubleUpConv(32, 3, last=True)
    )
    
    self.last_decoder_layer = nn.Sequential(
      nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=False
      ),
      nn.GroupNorm(1, 3),
      nn.GELU(),
      
      nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=1,
        stride=1,
        bias=True
      )
    )
    
  def encode(self, img):
    B = img.shape[0]
    
    h = self.last_encoder_layer(self.encoder(img))
    mean, logvar = self.hid_2_mean(h), self.hid_2_logvar(h)
    return mean, logvar    
  
  def decode(self, z):
    h = self.z_2_hid(z)
    out = self.last_decoder_layer(self.decoder(h))
    
    return out
  
  def forward(self, X):
    mean, logvar = self.encode(X)
    std = torch.exp(logvar * 0.5)
    
    z_reparam = mean + std * torch.randn_like(mean)
    out = self.decode(z_reparam)
    
    return out, mean, logvar