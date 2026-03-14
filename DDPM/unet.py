# unet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    """
    A simple UNet for 28x28 MNIST, designed for CPU/MPS-friendly training.
    ~110k params by default.
    """
    def __init__(self, base_ch=32, time_dim=64):
        super().__init__()
        self.in_ch = 1      #Just 1 in channel, as MNIST just contains greyscale values.
        self.out_ch = 1     #Just 1 out channel, as MNIST just contains greyscale values.
        self.base_ch = base_ch  #Hidden dimension. Typically more channels compared to in and out channels.
        self.time_dim = time_dim    #Dimension of time embeddings. The time embeddings encode the current deniosing step.

        # Small MLP for the time embedding.
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Encoder
        self.enc1 = ConvBlock(self.in_ch, base_ch, time_dim)     #1channel -> 32 channels
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)    #28x28 -> 14x14 
        self.enc2 = ConvBlock(base_ch, base_ch * 2, time_dim)   #32 channels -> 64 channels
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)    #14x14 -> 7x7 

        # Bottleneck
        self.bot = ConvBlock(base_ch * 2, base_ch * 2, time_dim)    #64 channels -> 64 channels

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)     #7x7 -> 14x14
        self.dec1 = ConvBlock(base_ch * 2 + base_ch * 2, base_ch, time_dim)     #64 channels -> 32 channels
        self.up2 = nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1)         #14x14 -> 28x28
        self.dec2 = ConvBlock(base_ch + base_ch, base_ch, time_dim)     #32 channels -> 1 channels

        self.out = nn.Conv2d(base_ch, self.out_ch, 1)    #28x28 -> 28x28

    def forward(self, x, t):
        """
        Perfomres the forward pass

        Batched input samples x: (batch_dimension, 1, image_size, simage_size)

        Returns predicted noise εθ at denoising step t with same shape as x
        """
        # Create time embedding for t. 
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # encoder
        e1 = self.enc1(x, t_emb)                # (batch_dimension, base_ch, 28, 28)
        d1 = self.down1(e1)                     # (batch_dimension, base_ch, 14, 14)
        e2 = self.enc2(d1, t_emb)               # (batch_dimension, 2*base_ch, 14, 14)
        d2 = self.down2(e2)                     # (batch_dimension, 2*base_ch, 7, 7)

        # bottleneck
        b = self.bot(d2, t_emb)

        # decoder
        u1 = self.up1(b)                        # (batch_dimension, 2*base_ch, 14, 14)
        u1 = torch.cat([u1, e2], dim=1)         # skip
        u1 = self.dec1(u1, t_emb)               # (batch_dimension, base_ch, 14, 14)
        u2 = self.up2(u1)                       # (batch_dimension, base_ch, 28, 28)
        u2 = torch.cat([u2, e1], dim=1)         # skip
        u2 = self.dec2(u2, t_emb)               # (batch_dimension, base_ch, 28, 28)

        return self.out(u2)                     # (batch_dimension,1,28,28)


class ConvBlock(nn.Module):
    """
    Helper class which handles the convolution operation for the UNet.
    """
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(),
        )
        # FiLM-style conditioning from t-embedding -> scale & shift
        self.to_scale = nn.Linear(time_dim, out_ch)
        self.to_shift = nn.Linear(time_dim, out_ch)

    def forward(self, x, t_emb):
        h = self.conv(x)
        # reshape FiLM params to (B, C, 1, 1)
        scale = self.to_scale(t_emb).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(t_emb).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + scale) + shift


# --- Sinusoidal timestep embedding (classic DDPM-style) ---
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Helper function which creates transformer like positional encodings.

    timesteps: (B,) long
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0,1))
    return emb  # (B, dim)