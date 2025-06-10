
import torch
import torch.nn as nn

from typing import Tuple
from .patches import Patchify, PositionalEncoding

class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=(16, 16), in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False, num_frames=16, embed_per_frame=False, spacetime_separable_pos_embed=False, block_func=Block, block_kwargs={}):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = (tubelet_size,) + patch_size
        self.pt, self.ph, self.pw = self.patch_size

        self._embed_per_frame = embed_per_frame
        if not self._embed_per_frame:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size,num_frames=num_frames)
            num_patches = self.patch_embed.num_patches
        elif self._embed_per_frame:
            assert (num_frames % tubelet_size) == 0
            num_embeddings = (num_frames // tubelet_size)
            self.patch_embed = nn.ModuleList([
                PatchEmbed(
                    img_size=img_size, patch_size=patch_size,
                    in_chans=in_chans, embed_dim=embed_dim,
                    tubelet_size=tubelet_size, num_frames=tubelet_size)
                for _ in range(num_embeddings)])
            num_patches = self.patch_embed[0].num_patches * num_embeddings

        self.image_size = img_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        print("NUM PATCHES IN ENCODER", self.num_patches)

        # TODO: Add the cls token
        if num_patches is None:
            self.pos_embed = None
        elif use_learnable_pos_emb:
            self._learnable_pos_embed = True
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        else:
            # sine-cosine positional embeddings
            self._learnable_pos_embed = False
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_func(
                dim=embed_dim, in_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, **block_kwargs)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _set_pos_embed(self, dim=None):
        if dim is None:
            dim = self.embed_dim
        if self.pos_embed is None:
            self.pos_embed = get_sinusoid_encoding_table(
                self.num_patches, dim)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self):
        return self.pos_embed

    def forward_block(self, x, idx):
        return self.blocks[idx](x)

    def tokenize(self, x, mask=None):

        if not self._embed_per_frame:
            x = self.patch_embed(x)
        elif self._embed_per_frame:
            x = torch.cat([
                self.patch_embed[i](
                    x[:,:,(i*self.pt):((i+1)*self.pt)])
                for i in range(len(self.patch_embed))], 1)
            
        pos_embed = self._get_pos_embed().type_as(x).to(x.device).clone()
        if not self._learnable_pos_embed:
            pos_embed = pos_embed.detach()
        x = x + pos_embed
        return (x, mask)

    def tokenize_and_mask(self, x, mask):

        x, mask = self.tokenize(x, mask)
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)
        return x_vis

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        if not self._embed_per_frame:
            x = self.patch_embed(x)
        elif self._embed_per_frame:
            x = torch.cat([
                self.patch_embed[i](
                    x[:,:,(i*self.pt):((i+1)*self.pt)])
                for i in range(len(self.patch_embed))], 1)

        pos_embed = self._get_pos_embed().type_as(x).to(x.device).clone()
        if not self._learnable_pos_embed:
            pos_embed = pos_embed.detach()
        x = x + pos_embed
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def _set_inputs(self, *args, **kwargs):
        pass

    def forward(self, x, mask, *args, **kwargs):
        self._set_inputs(x, mask, *args, **kwargs)
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=(16, 16), num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, block_func=Block, block_kwargs={}
                 ):
        super().__init__()


        self.num_classes = num_classes

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_func(
                dim=embed_dim, in_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, **block_kwargs)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_block(self, x, idx):
        return self.blocks[idx](x)

    def get_last_tokens(self, x, return_token_num):
        if return_token_num > 0:
            return self.head(self.norm(x[:,-return_token_num:]))
        elif return_token_num == 0:
            return self.head(self.norm(x))[:,x.size(1):]
        else:
            return self.head(self.norm(x))

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class CounterFactualModel(nn.Module):
    def __init__(self,
            resolution : Tuple[int, int],
            patch_size : Tuple[int, int]
        ):
        super().__init__()
        self.patchify = Patchify(patch_size = patch_size, temporal_dim = 0)
        #self.patch_encoder = PositionalEncoding()
        self.encoder = None
        self.decoder = None

        self._patch_size = patch_size
        self._resolution = resolution

    @property
    def patch_size(self): return self._patch_size

    @property
    def resolution(self): return self._resolution

    def reconstruct(self, x0, xt, mask = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x0 : the first frame as image [B,W,H,C]
            xt : the second frame as image [B,W,H,C]
            mask : the mask for the second frame [B,W,H]
        Returns:
            recon : the reconstruction of the next frame
            loss  : the mse loss compare to the gt xt
        """
        B, W, H, C = x0.shape # check shape of the input image
        return

    def train(self, dataset, epochs : int, lr : float = 2e-4):
        return