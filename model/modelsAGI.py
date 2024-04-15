from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp

from utils.pos_embed import get_2d_sincos_pos_embed

import numpy as np
import scipy.stats as stats
import math


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale

        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class myLoss(nn.Module):

    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, probs, postive, negative):
        log_probs_pos = -torch.log(probs + 1e-6)
        postive_prob = torch.mul(log_probs_pos, postive)

        log_probs_meg = -torch.log(1 - probs - 1e-6)
        negative_prob = torch.mul(log_probs_meg, negative)

        # print(torch.sum(postive_prob), torch.sum(negative_prob))

        loss = (torch.sum(postive_prob) + torch.sum(negative_prob))  # / (torch.sum(postive) + torch.sum(negative))

        return loss


class myLossRegression(nn.Module):

    def __init__(self):
        super(myLossRegression, self).__init__()
        self.criteria = nn.MSELoss()

    def forward(self, probs, anchor, postive, negative):
        loss_initial = self.criteria(probs, anchor)
        loss_pos = torch.mul(loss_initial, postive)

        loss_neg = torch.mul(loss_initial, negative)

        # print(torch.sum(postive_prob), torch.sum(negative_prob))

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg))  # / (torch.sum(postive) + torch.sum(negative))

        return loss


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        torch.nn.init.normal_(self.word_embeddings.weight, std=.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std=.02)

    def forward(
            self, input_ids
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings  # todo: delete this line

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ContinuousValueEncoder(nn.Module):
    def __init__(self, output_dim, max_position_embeddings=96):
        super(ContinuousValueEncoder, self).__init__()
        self.linear = nn.Linear(1, output_dim)
        # 为-1的情况预定义一个编码向量
        self.special_encoding = nn.Parameter(torch.randn(output_dim))
        self.position_embeddings = nn.Embedding(max_position_embeddings, output_dim)

    def forward(self, x):
        # 判断输入是否为-1
        is_special = (x == -1).float().unsqueeze(1)  # 增加维度以适应广播
        # 通过线性层对输入进行编码
        x_encoded = self.linear(x)
        # 使用特殊值的编码向量替换-1对应的编码
        x_encoded = torch.where(is_special == 1, self.special_encoding.expand_as(x_encoded), x_encoded)

        seq_length = x.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        x_encoded = x_encoded + position_embeddings  # todo: delete this line

        x_encoded = self.dropout(x_encoded)
        return x_encoded


# 3 stage:
# 2. calculate loss
# 3. calculate acc

class MaskedGenerativeEncoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    # cyo
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vqgan_ckpt_path='vqgan_jax_strongaug.ckpt', grid_num=69, regression=1):
        super().__init__()

        # --------------------------------------------------------------------------
        print("grid_num", grid_num)
        self.codebook_size = grid_num + 1
        self.max_len = grid_num

        vocab_size = self.codebook_size
        self.fake_class_label = self.codebook_size - 1
        self.mask_token_label = vocab_size - 1

        if regression == 1:
            self.token_emb = ContinuousValueEncoder(embed_dim,
                                                    max_position_embeddings=self.max_len)
        else:
            self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                            hidden_size=embed_dim,
                                            max_position_embeddings=self.max_len,
                                            dropout=0.1)
        # --------------------------------------------------------------------------

        # ur encoder specifics
        dropout_rate = 0.1

        self.cls_token = torch.zeros(1, 1, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.out = nn.Linear(embed_dim, 1)

        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss

        if regression:
            self.criterion = myLossRegression()
        else:
            self.criterion = myLoss()

        self.initialize_weights()

        self.regression = regression

    def get_mask_token_label(self):
        return self.mask_token_label

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, train_tuple):
        # tokenization
        x = train_tuple[0]
        # 69 69 69 69 4 69 69 7 69 69 69
        b, c = x.shape

        # bert embedding
        if self.regression == 1:
            masks = torch.ones((b, self.codebook_size - 1), device=x.device, dtype=torch.float) * -1
            masks = torch.where(train_tuple[1] == 1, masks, train_tuple[0])

            input_embeddings = self.token_emb(masks)
        else:
            masks = torch.ones((b, self.codebook_size - 1), device=x.device, dtype=torch.long) * self.mask_token_label
            indices = torch.arange(0, self.codebook_size - 1, device=x.device, dtype=torch.long) \
                .unsqueeze(0).expand(b, -1)
            # print(x.shape, masks.shape)
            masks[x.bool()] = indices[x.bool()]
            input_embeddings = self.token_emb(masks)

        for blk in self.blocks:
            input_embeddings = blk(input_embeddings)
        input_embeddings = self.norm(input_embeddings)

        probs = self.out(input_embeddings)
        # use sigmoid to get the probability
        if self.regression == 0:
            probs = torch.sigmoid(probs).squeeze(-1)

        return probs

    def forward_loss(self, probs, train_tuple):
        postive = train_tuple[1]
        negative = train_tuple[2]
        if self.regression == 1:
            loss = self.criterion(probs, train_tuple[0], postive, negative)
        else:
            loss = self.criterion(probs, postive, negative)
        return loss

    def calculate_acc(self, probs, train_tuple):
        # print(probs)
        postive = train_tuple[1]
        negative = train_tuple[2]
        acc_pos = (int(torch.sum((probs > 0.5) * postive)), int(torch.sum(postive)))
        acc_neg = (int(torch.sum(((1 - probs) > 0.5) * negative)), int(torch.sum(negative)))
        return acc_pos, acc_neg

    def forward(self, train_tuple):
        probs = self.forward_encoder(train_tuple)
        loss = self.forward_loss(probs, train_tuple)
        acc = ((probs, train_tuple) if self.regression == 1 else self.calculate_acc(probs, train_tuple))
        return loss, acc


def ur_vit_base_patch16(grid_num=69, regression=1, **kwargs):
    print('ur', grid_num)
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=512, depth=6, num_heads=8,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), grid_num=grid_num, regression=regression, **kwargs)
    return model


def ur_vit_large_patch16(grid_num=69, **kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), grid_num=grid_num, **kwargs)
    return model
