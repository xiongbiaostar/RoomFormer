# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
import numpy as np

import matplotlib.pyplot as plt
from datasets import build_dataset

from models.losses import custom_L1_loss, dn_L1_loss
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

import torch.nn.functional as F
from torch import nn
from util.poly_ops import pad_gt_polys, get_gt_polys




def prepare_for_dn(dn_args, tgt_weight, embedweight, batch_size, training, num_queries, num_classes, hidden_dim, label_enc,num_query_per_poly=40):
    """
    The major difference from DN-DAB-DETR is that the author process pattern embedding pattern embedding in its detector
    forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, scalar(number of dn groups), label_noise_scale, box_noise_scale, num_patterns
    :param tgt_weight: use learnbal tgt in dab deformable detr
    :param embedweight: positional anchor queries
    :param batch_size: bs
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param num_classes: number of classes
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn  #在目标检测中指的是物体类别，是否可以考虑调整为有效性和无效性？
    :return:
    """

    if training:
        targets, scalar, label_noise_scale, poly_noise_scale = dn_args


    if tgt_weight is not None and embedweight is not None:
        # decoder的content query和position query
        # content query
        indicator0 = torch.zeros([num_queries, 1]).cuda()
        # sometimes the target is empty, add a zero part of label_enc to avoid unused parameters
        tgt = torch.cat([tgt_weight, indicator0], dim=1) + label_enc.weight[0][0] * torch.tensor(0).cuda()
        # position query
        refpoint_emb = embedweight  # [800,2]
    else:
        tgt = None
        refpoint_emb = None

    # 去噪任务
    if training:
        #一些索引
        #在DN-DETR中list 中的每個都是值為 1 shape 為 (num_gt_img,) 的張量

        known = [(torch.ones_like(t['labels'])).cuda() for t in targets] #[4,40]4是多边形的数量
        # torch.nonzero() 返回的是張量中值不為0的元素的索引，list 中的每個張量 shape 是 (num_gt_img,1)
        #TODO：我感觉不太需要这个。因为coco中0是背景，所以要排除在外。如果我按照roomformer的设置，那就只有0和1
        known_num = [sum(k) for k in known]
        points_num_each_poly = [t['lengths'] for t in targets] #[[poly_num]]
        # print("known_num",known_num)
        # you can uncomment this to use fix number of dn queries
        # if int(max(known_num))>0:
        #     scalar=scalar//int(max(known_num))

        # can be modified to selectively denosie some label or boxes; also known label prediction




        labels = torch.cat([t['labels'] for t in targets]) #有效点/无效点[poly_num_allbatch]
        coords = torch.cat([t['coords'].reshape(-1,2) for t in targets])#[poly_num_allbatch,2]
        # print("点点带年",coords.shape)

        #torch.full_like是构造一个和t['labels']的shape一样的张量，且张量里面的值为i，标识当前label属于batch中的哪个场景
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_lengths = torch.cat([t['lengths'] for t in targets])

        # add noise
        # 顶点个数减少，涉及张量：labels，coords，known_num，points_num_each_poly
        # prob_point = 0.5
        modify_flag = True  # torch.rand(1).item() < prob_point #有50%的概率会进行顶点增减
        #
        # print("s1", coords.shape, labels.shape, batch_idx, known[0].shape,known[1].shape,known_num)
        if modify_flag:
            # 随机选择一个多边形
            point_index = torch.randint(1, labels.shape[0], (1,)).item()
            point_batch_idx = batch_idx[point_index].item()
            coords = torch.cat((coords[:point_index-1,:],coords[point_index:,:]))
            labels = torch.cat((labels[:point_index-1],labels[point_index:]))
            batch_idx = torch.cat((batch_idx[:point_index-1],batch_idx[point_index:]))
            # print("s",point_index,point_batch_idx,coords.shape,labels.shape,batch_idx,known)
            known_len = known[point_batch_idx].shape[0]
            known[point_batch_idx] = known[point_batch_idx][:known_len-1]
            # print("s2",point_index,point_batch_idx,coords.shape,labels.shape,batch_idx,known)
            known_num[point_batch_idx] = known_num[point_batch_idx]-1
            # known_lengths
            cumulative_sum = torch.cumsum(known_lengths, dim=0)

            # 为了便于索引，我们在累积和的列表前添加一个0
           # cumulative_sum = torch.cat(([0], cumulative_sum))

            # 使用循环或向量化操作来定位点所属的多边形
            # 由于我们的累积和列表是递增的，我们可以使用torch.searchsorted来高效地完成这个任务
            # 注意：searchsorted返回的是应该插入以保持顺序的索引，因此我们需要减去1来得到正确的多边形索引
            polygon_index = torch.searchsorted(cumulative_sum, point_index, right=True).item()
            known_lengths[polygon_index]=known_lengths[polygon_index]-2

            # print("s3",labels.shape,known_num,known_lengths,known_lengths.shape,polygon_index,cumulative_sum,polygon_index)



            # point_index = torch.randint(0, num_points_poly, (1,)).item()
        know_idx = [torch.nonzero(t) for t in known]  # [[poly_num*40,2]]

        unmask_poly = unmask_label = torch.cat(known)  # [poly_num_allbatch,40]
        known_indice = torch.nonzero(unmask_label + unmask_poly)
        known_indice = known_indice.view(-1)
        # point_index = torch.randint(0, len(known_num), (1,)).item()

        #print("加噪部分", unmask_poly.shape,coords.shape,labels.shape,points_num_each_poly, known_num,known.shape,batch_idx.shape,labels.shape)
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_coords = coords.repeat(scalar, 1)
        # print("哪里",known_labels.shape,labels.shape)
        known_labels_expaned = known_labels.clone()
        known_coords_expand = known_coords.clone()
        #

        known_lengths = known_lengths.repeat(scalar,1).view(-1)
        # print("dn-components",coords.shape,known_coords.shape,known_coords_expand.shape,known_num,points_num_each_poly,known_indice.shape,unmask_label.shape,unmask_poly.shape,known_lengths,known_lengths.shape)

        # noise on the label
        if label_noise_scale > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of poly noise
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        # noise on the box
        if poly_noise_scale > 0:
            diff = torch.zeros_like(known_coords_expand)
            #应该是保证中心点移动后还在原来的bbox范围内。
            # 对于多边形的顶点加噪，顶点的增减和移动。暂时先只考虑顶点的偏移试试
            #diff[:, :2] = known_bbox_expand[:, 2:] / 2
            #diff[:, 2:] = known_bbox_expand[:, 2:]
            diff = known_coords_expand
            # vis_noisePoly(known_coords_expand, points_num_each_poly[0])
            # vis_noisePoly(known_coords_expand[known_num[0]:],points_num_each_poly[1])
            # torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),diff)是加的噪声，box_noise_scale负责调节噪声幅度。
            known_coords_expand += torch.mul((torch.rand_like(known_coords_expand) * 2 - 1.0),
                                           diff).cuda() * poly_noise_scale

            known_coords_expand = known_coords_expand.clamp(min=0.0, max=1.0)

            # vis_noisePoly(known_coords_expand, points_num_each_poly[0])
            # vis_noisePoly(known_coords_expand[known_num[0]:], points_num_each_poly[1])

        m = known_labels_expaned.long().to('cuda')

        input_label_embed = label_enc(m)
        # add dn part indicator
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()
        # print("mm", m.shape,input_label_embed.shape,indicator1.shape)
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1) #295,256

        input_coords_embed = inverse_sigmoid(known_coords_expand)#295,2
        single_pad = int(max(known_num))
        # print("single",single_pad)
        pad_size = int(single_pad * scalar)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_coords = torch.zeros(pad_size, 2).cuda() #每个坐标两个点
        if tgt is not None and refpoint_emb is not None:
            #print(padding_label.shape,tgt.shape)
            input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
            input_query_coords = torch.cat([padding_coords, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
        else:
            input_query_label = padding_label.repeat(batch_size, 1, 1)
            input_query_coords = padding_coords.repeat(batch_size, 1, 1)

        # input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
        # input_query_coords = torch.cat([padding_coords, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
        # print("input",input_query_coords.shape)

        # map in order
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])

            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            # print(known_num,input_query_label.shape)
        if len(known_bid):
            # (known_bid.shape,map_known_indice.shape,input_coords_embed.shape,input_query_coords.shape,input_label_embed.shape,input_query_label.shape)
            # print(known_bid.shape,map_known_indice.shape,input_label_embed.shape)
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_coords[(known_bid.long(), map_known_indice)] = input_coords_embed.float()

        tgt_size = pad_size + num_queries
        # (i,j) = True 代表 i 不可見 j
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct 令匹配任務的 queries 看不到做去噪任務的 queries，因為後者含有真實標籤的信息
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        # 对于去噪任务的 queries，只有同组内的相互可见，避免跨组泄露真實標籤的信息，
        # 因为每组中，gt 和 query 是 one-to-one 的。
        # 于是，在同一组内，对于每个 query 来说，其它 queries 都不会有自己 gt 的信息
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_polys': (known_labels, known_coords),
            'know_idx': know_idx,
            'pad_size': pad_size,
            "known_lengths":known_lengths
        }
    else:  # no dn for inference
        if tgt is not None and refpoint_emb is not None:
            input_query_label = tgt.repeat(batch_size, 1, 1)
            input_query_coords = refpoint_emb.repeat(batch_size, 1, 1)
        else:
            input_query_label = None
            input_query_coords = None
        attn_mask = None
        mask_dict = None

    # input_query_label = input_query_label.transpose(0, 1)
    # input_query_bbox = input_query_bbox.transpose(0, 1)
    # print("input_query",input_query_coords.shape,input_query_label.shape)
    return input_query_label, input_query_coords, attn_mask, mask_dict


def dn_post_process(outputs_class, outputs_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        # print("dn_post_process",output_known_class.shape,output_known_coord.shape,outputs_coord.shape,outputs_class.shape)
        #output_known_class.shape=[6, 2, 290, 1], output_known_coord.shape=[6, 2, 290, 2], outputs_coord.shape=[6, 2, 800, 2], outputs_class.shape=[6, 2, 800, 1]
        mask_dict['output_known_lbs_polys']=(output_known_class,output_known_coord)
        # print("dn_post_process",outputs_class.shape, outputs_coord.shape,output_known_class.shape,output_known_coord.shape)
    return outputs_class, outputs_coord


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    Returns:

    """
    output_known_class, output_known_coord = mask_dict['output_known_lbs_polys']
    known_labels, known_polys = mask_dict['known_lbs_polys']
    map_known_indice = mask_dict['map_known_indice']

    known_indice = mask_dict['known_indice']
    known_lengths = mask_dict['known_lengths']

    batch_idx = mask_dict['batch_idx']
    bid = batch_idx[known_indice]
    # print("prepare_for_loss",output_known_coord.shape,output_known_class.shape,len(output_known_class))#torch.Size([6, 2, 290, 2]) torch.Size([6, 2, 290, 1]) 6
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
    num_tgt = known_indice.numel() #计算known_indice中元素总数
    # print("prepare for loss",num_tgt, known_lengths)#loss 380 [tensor([ 8, 20,  8], device='cuda:0'), tensor([ 8, 20, 12,  8,  8,  8, 12, 12, 12, 16], device='c
    return known_labels, known_polys, output_known_class, output_known_coord, known_lengths


def tgt_loss_polys(src_polys, tgt_polys,target_len):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    if len(tgt_polys) == 0:
        return {
            'tgt_loss_coords': torch.as_tensor(0.).to('cuda')
        }
    # print("dn-loss-polys",tgt_polys.shape,src_polys.shape,src_polys.flatten(1,2).shape)
    loss_coords = dn_L1_loss(src_polys, tgt_polys, target_len)

    losses = {}
    losses['tgt_loss_coords'] = loss_coords

    return losses


def tgt_loss_labels(src_logits_, tgt_labels_, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_ce': torch.as_tensor(0.).to('cuda'),
        }

    src_logits, tgt_labels= src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0).float()

    # print("tgt_loss_labels",src_logits.shape,tgt_labels.shape,src_logits.dtype,tgt_labels.dtype)


    loss_ce = F.binary_cross_entropy_with_logits(src_logits, tgt_labels)

    losses = {'tgt_loss_ce': loss_ce}

    return losses


def compute_dn_loss(mask_dict, training, aux_num):
    """
       compute dn loss in criterion
       Args:
           mask_dict: a dict for dn information
           training: training or inference flag
           aux_num: aux loss number
           focal_alpha:  for focal loss
       """
    losses = {}
    #print("mask_dict",mask_dict)
    if training and 'output_known_lbs_polys' in mask_dict:
        known_labels, known_polys, output_known_class, output_known_coord, known_lengths = prepare_for_loss(mask_dict)
        # print("compute_dn_loss",output_known_class[-1].shape, known_labels.shape,known_polys.shape,output_known_coord[-1].shape)
        losses.update(tgt_loss_labels(output_known_class[-1].view(-1), known_labels))
        losses.update(tgt_loss_polys(output_known_coord[-1], known_polys, known_lengths))
    else:
        losses['tgt_loss_coords'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
    # print("compute_dn_loss",aux_num)
    if aux_num:
        for i in range(aux_num):
            # dn aux loss
            if training and 'output_known_lbs_polys' in mask_dict:
                l_dict = tgt_loss_labels(output_known_class[i], known_labels)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = tgt_loss_polys(output_known_coord[i], known_polys, known_lengths )
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_coords'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    return losses

def vis_noisePoly(polygon_data,lengths):
    # 分割数据

    polygons = []

    index = 0

    for length in lengths:
        length=int(length/2)
        polygon = polygon_data[index:index + length].cpu()
        if not np.array_equal(polygon[0], polygon[-1]):
            polygon = np.vstack([polygon, polygon[0]])
        polygons.append(polygon)

        index += length

    for polygon in polygons:
        x, y = polygon.T

        plt.plot(x, y)

    plt.legend()

    plt.xlabel('X')

    plt.ylabel('Y')

    plt.title('Polygons')

    plt.grid(True)

    plt.show()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('RoomFormer training script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     dataset_train = build_dataset(image_set='train', args=args)
#     gt_0 = dataset_train.__getitem__(0)
#     gt_1 = dataset_train.__getitem__(1)
#     gts =[gt_0,gt_1]
#     device = torch.device(args.device)
#     targets = get_gt_polys([gt["instances"].to(device) for gt in gts],40,device)#
#     dn_args=(targets, 5, 0.2, 0.4)
#     query_embed = nn.Embedding(800, 2)
#     tgt_embed = nn.Embedding(800, 255)
#     label_enc = nn.Embedding(1 + 1, 256 - 1)
#     label_enc = label_enc.to(device)
#     tgt_weight = tgt_embed.weight.to(device)
#     query_embed_weight =query_embed.weight.to(device)
#     prepare_for_dn(dn_args=dn_args,tgt_weight=tgt_weight,embedweight=query_embed_weight,training=True,num_queries=800,num_classes=1,hidden_dim=256,label_enc=label_enc,batch_size=2)