"""
VIVIT model from https://arxiv.org/abs/2103.15691.
Code from https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
class LeFF(nn.Module):
    
    def __init__(self, dim = 192, scale = 4, depth_kernel = 3):
        super().__init__()
        
        scale_dim = dim*scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(scale_dim),
                                    nn.GELU(),
                                    Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                    )
        
        self.depth_conv =  nn.Sequential(nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
                          nn.BatchNorm2d(scale_dim),
                          nn.GELU(),
                          Rearrange('b c h w -> b (h w) c', h=14, w=14)
                          )
        
        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c')
                                    )
        
    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x
    
    
class LCAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = q[:, :, -1, :].unsqueeze(2) # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
        

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(nn.Module):
    def __init__(self, image_size=252, int_image_size=256, patch_size=16, num_predictions=32, num_frames=4, dim = 192, depth = 8, heads = 8, pool = 'cls', in_channels = 11, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, **kwargs):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert int_image_size % patch_size == 0, f'Path size ({patch_size}) must divide {int_image_size} entirely.'
        
        assert dim % heads == 0, f"Cannot divide {dim}-dimensional vector into {heads} heads"
        
        # Attributes
        self.image_size = image_size
        self.int_image_size = int_image_size
        self.patch_size = patch_size
        self.num_predictions = num_predictions
        self.num_frames = num_frames
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.pool = pool
        self.in_channels = in_channels
        self.dim_head = dim // heads
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.scale_dim = scale_dim
        
        num_classes = self.num_predictions * image_size **2
        
        num_patches = (int_image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, self.dim_head, dim*scale_dim, dropout)

        #self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        #self.temporal_transformer = Transformer(dim, depth, heads, self.dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

        self.magnifier = nn.Linear(self.dim, (self.int_image_size // self.patch_size)**2)
        
        self.conv_head = nn.Sequential(
            nn.LayerNorm((self.int_image_size, self.int_image_size)),
            nn.Conv2d(4, self.num_predictions, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Interpolating frames to be of prefferred shape (256 x 256)
        # BCTHW -> BTCHW
        x = x.transpose(1, 2)
        
        in_shape = (self.in_channels, self.int_image_size, self.int_image_size)
        x = torch.nn.functional.interpolate(x, size=in_shape, mode="nearest")
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        # x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)  # Originally in VIVIT (taking class token only)
        x = rearrange(x, '(b t) ... -> b t ...', b=b)  # Alternative: Keeping all tokens

        # TODO: Handle temporal transformer keeping all tokens
        # cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_temporal_tokens, x), dim=1)
        # x = self.temporal_transformer(x)
        
        x = x[:, :, 1:, :]  # Removing class token
        x = self.magnifier(x)
        x = rearrange(x, 'b t (h w) (p1 p2) -> b t (h p1) (w p2)', h=int(self.int_image_size**0.5), p1=self.patch_size)

        x = self.conv_head(x).unsqueeze(2)
        out_shape = (1, self.image_size, self.image_size)
        out = torch.nn.functional.interpolate(x, size=out_shape, mode="trilinear")
        
        out = out.transpose(1, 2)
        return out

if __name__ == "__main__":
    from pytorch_model_summary import summary
    
    device = torch.device("cuda:7")
    
    with torch.no_grad():
        model = ViViT().to(device)
        
        
        B, C, T, H, W = 2, 11, 4, 252, 252
        x = torch.randn(B, C, T, H, W).to(device)
        
        summary(model, x, print_summary=True)
        
        out_shape = model(x).shape
        print(out_shape)
        assert out_shape == (B, 1, 32, H, W)
