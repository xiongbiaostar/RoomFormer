# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", poly_refine=True, return_intermediate_dec=False, aux_loss=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4, query_pos_type="none", use_mqs=False,
                 use_look_forward_twice=False, num_queries=800):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.use_mqs = use_mqs
        self.look_foward_twice = use_look_forward_twice
        self.num_queries = num_queries

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, poly_refine,
                                                    return_intermediate_dec, aux_loss, query_pos_type,
                                                    use_look_forward_twice=use_look_forward_twice)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if query_pos_type == 'sine':
            self.decoder.pos_trans = nn.Linear(d_model, d_model)
            self.decoder.pos_trans_norm = nn.LayerNorm(d_model)
        if use_mqs:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.tgt_embed = nn.Embedding(800, d_model)

            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, d_model):
        num_pos_feats = 128  #
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        print("proposal_pos_embed", proposals.shape)
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        # [2, 1360, 256]
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)  # 每个批次的有效高度和宽度的尺度
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            # lvl=0：scale:torch.Size([2, 1, 1, 2]);   grid: torch.Size([2, 32, 32, 2]);  wh:torch.Size([2, 32, 32, 2])
            # lvl=1:scale:torch.Size([2, 1, 1, 2]);   grid: torch.Size([2, 16, 16, 2]);  wh:torch.Size([2, 16, 16, 2])
            # lvl=2:scale:torch.Size([2, 1, 1, 2]);   grid: torch.Size([2, 8, 8, 2]);  wh:torch.Size([2, 8, 8, 2])
            # lvl=3:scale:torch.Size([2, 1, 1, 2]);   grid: torch.Size([2, 4, 4, 2]);  wh:torch.Size([2, 4, 4, 2])
            # print("gen_encoder_output_proposals",scale.shape,grid.shape,wh.shape,lvl)
            proposal = grid.view(N_, -1, 2)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)  # 2, 1360, 2]

        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # 把padding的值给mask掉（原来mask为true的地方继续mask）
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # 把不满足output_proposals_valid的地方mask
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def forward(self, srcs, masks, pos_embeds, query_embed=None, tgt=None, tgt_masks=None):
        assert self.use_mqs or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        # [2,1360,256]
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape

        # use mix query selection
        if self.use_mqs:
            #[bs,1360,2]
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            # hack implementation for two-stage Deformable DETR

            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.coords_embed[self.decoder.num_layers](
                output_memory) + output_proposals

            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]  # [bs,topk]
            # 根据topk_proposals对应的索引从enc_outputs_coord_unact收集坐标值。[bs,topk,2]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1,
                                             topk_proposals.unsqueeze(-1).repeat(1, 1, 2))  # [bs,topk,2]

            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()

            reference_points_mqs = reference_points

            # sometimes the target is empty, add a zero part of query_embed to avoid unused parameters
            reference_points_mqs += self.tgt_embed.weight[0][0] * torch.tensor(0).cuda()
            tgt_mqs = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            # print("//",reference_points_mqs.shape, tgt_mqs.shape,mask_flatten.shape)

            # query_embed is non None when training.
            if query_embed is not None:
                reference_points_dab = query_embed.sigmoid()
                tgt_dab = tgt

                reference_points = torch.cat([reference_points_dab, reference_points_mqs], dim=1)
                tgt = torch.cat([tgt_dab, tgt_mqs], dim=1)
            else:
                reference_points = reference_points_mqs
                tgt = tgt_mqs

            # reference_points_radom = query_embed.sigmoid()
            #
            # reference_points = torch.cat([reference_points_radom, reference_points_mqs], dim=1)
            # tgt = torch.cat([tgt, tgt_mqs], dim=1)
            # print("use_mqs",reference_points.shape,tgt.shape)



        else:
            # 不需要这两行代码，因为在DN里由这部分处理
            # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = query_embed.sigmoid()
        init_reference_out = reference_points
        # print(reference_points.shape)

        # decoder
        # print("decoder前",mask_flatten.shape,tgt_masks.shape)
        hs, inter_references, inter_classes = self.decoder(tgt, reference_points, memory, src_flatten,
                                                           spatial_shapes, level_start_index, valid_ratios, query_embed,
                                                           mask_flatten, tgt_masks)

        if not self.use_mqs:
            enc_outputs_class, enc_outputs_coord_unact = None, None

        return hs, init_reference_out, inter_references, inter_classes, enc_outputs_class, enc_outputs_coord_unact


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, tgt_masks=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=tgt_masks)[
            0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, poly_refine=True, return_intermediate=False, aux_loss=False,
                 query_pos_type='none', use_look_forward_twice=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.poly_refine = poly_refine
        self.return_intermediate = return_intermediate
        self.aux_loss = aux_loss
        self.query_pos_type = query_pos_type

        self.coords_embed = None
        self.class_embed = None
        self.pos_trans = None
        self.pos_trans_norm = None

        # use_look_forward_twice
        self.use_look_forward_twice = use_look_forward_twice

    def get_query_pos_embed(self, ref_points):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=ref_points.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # [128]
        # N, L, 2
        ref_points = ref_points * scale
        # N, L, 2, 128
        pos = ref_points[:, :, :, None] / dim_t
        # N, L, 256
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(self, tgt, reference_points, src, src_flatten, src_spatial_shapes, src_level_start_index,
                src_valid_ratios,
                query_pos=None, src_padding_mask=None, tgt_masks=None):
        output = tgt  # [10, 800, 256]

        intermediate = []
        intermediate_reference_points = []
        intermediate_classes = []
        point_classes = torch.zeros(output.shape[:2]).unsqueeze(-1).to(output.device)
        for lid, layer in enumerate(self.layers):
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            if self.query_pos_type == 'sine':
                query_pos = self.pos_trans_norm(self.pos_trans(self.get_query_pos_embed(reference_points)))

            elif self.query_pos_type == 'none':
                query_pos = None
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                           src_padding_mask, tgt_masks)

            # iterative polygon refinement
            if self.poly_refine:
                offset = self.coords_embed[lid](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = offset
                new_reference_points = offset + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points

            # if not using iterative polygon refinement, just output the reference points decoded from the last layer
            elif lid == len(self.layers) - 1:
                offset = self.coords_embed[-1](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = offset
                new_reference_points = offset + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            # If aux loss supervision, we predict classes label from each layer and supervise loss
            if self.aux_loss:
                point_classes = self.class_embed[lid](output)
            # Otherwise, we only predict class label from the last layer
            elif lid == len(self.layers) - 1:
                point_classes = self.class_embed[-1](output)

            if self.return_intermediate:
                intermediate.append(output)
                if self.use_look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)

                # intermediate_reference_points.append(reference_points)
                intermediate_classes.append(point_classes)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(
                intermediate_classes)

        return output, reference_points, point_classes


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        poly_refine=args.with_poly_refine,
        return_intermediate_dec=True,
        aux_loss=args.aux_loss,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        query_pos_type=args.query_pos_type,
        use_mqs=args.use_mqs,
        use_look_forward_twice=True,
        num_queries=args.num_queries

    )
