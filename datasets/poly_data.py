from pathlib import Path

import torch
import torch.utils.data

from pycocotools.coco import COCO
from PIL import Image
import cv2

from util.poly_ops import resort_corners
from detectron2.data import transforms as T
from torch.utils.data import Dataset
import numpy as np
import os
from copy import deepcopy

from detectron2.data.detection_utils import annotations_to_instances, transform_instance_annotations
from detectron2.structures import BoxMode

#构建Dataet
class MultiPoly(Dataset):
    def __init__(self, img_folder, ann_file, transforms, semantic_classes,slice_number):
        super(MultiPoly, self).__init__()
        #设置根目录，变换方法，语义分类，coco平面图文件，深度图id，和prepare，即如何构建返回数据的类
        self.root = img_folder
        self._transforms = transforms
        self.semantic_classes = semantic_classes
        self.coco = COCO(ann_file)
        #coco文件读取的id，所以在写coco文件时最好是同属于一个数据的切片深度图共享一个id
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.slice_num = slice_number
        self.prepare = ConvertToCocoDict(self.root, self._transforms,self.slice_num)

    #get_image在哪里用到过，值得注意
    def get_image(self, path):
        return Image.open(os.path.join(self.root, path))

    #dataloader中可能会用到的，获取数据总长度
    def __len__(self):
        return len(self.ids)

    #dataloader中会用到的，获得单个数据项和对应标签
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: COCO format dict
            coco形式的字典
        """
        coco = self.coco
        img_id = self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        ### Note: here is a hack which assumes door/window have category_id 16, 17 in structured3D
        if self.semantic_classes == -1:
            target = [t for t in target if t['category_id'] not in [16, 17]]

        #justATest = coco.loadImgs(img_id)
        #这个地方loadImgs加载的只有coco中的图片信息，并没有加载图片本身
        path = coco.loadImgs(img_id)[0]['file_name']
        #调用了ConverToCocoDict来构造标签数据
        record = self.prepare(img_id, path, target)

        return record


class ConvertToCocoDict(object):
    def __init__(self, root, augmentations,slice_number):
        self.root = root
        #augmentations存储了transform，即图像随机变换的数据
        self.augmentations = augmentations
        self.slice_num = slice_number

    #构造数据和对应的标签的东西
    def __call__(self, img_id, path, target):

        #这部分是新写的call
        #path里存储的是图片的名字，现在要只提取“.png”前面的部分
        folder_name = path.split(".")[0]
        folder_path = os.path.join(self.root, folder_name)
        merged_image = None
        for sliceIndex in range(self.slice_num):
            #把切片的五张图堆叠在一起，最终merged的shape为(256,256,5)
            file_name = os.path.join(folder_path,str(sliceIndex)+".png")
            slice_img = np.array(Image.open(file_name))
            w, h = slice_img.shape
            slice_img = np.expand_dims(slice_img,2)
            if merged_image is not None:
               merged_image = np.append(merged_image,slice_img,axis=2)
            else:
               merged_image = slice_img
            

        record = {}
        record["file_name"] = folder_path+".png"
        record["height"] = h
        record["width"] = w
        record['image_id'] = img_id
        for obj in target: obj["bbox_mode"] = BoxMode.XYWH_ABS
        record['annotations'] = target

        # 如有变换，则把图像变换
        if self.augmentations is not None:
            merge_aug_input = T.AugInput(merged_image)
            merge_transforms = self.augmentations(merge_aug_input)
            merged_image = merge_aug_input.image
        #需要将h*w*c的图片转化为c*h*w的图片
        merged_image = np.transpose(merged_image,[2,0,1])
        #转化为tensor后输入到image
        record['image'] = (1 / 255) * torch.as_tensor(np.ascontiguousarray(merged_image))
        if self.augmentations is None:
            record['instances'] = annotations_to_instances(target, (h, w), mask_format="polygon")
        else:
            annos = [
                transform_instance_annotations(
                    obj, merge_transforms, (h,w)
                )
                for obj in record.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # resort corners after augmentation: so that all corners start from upper-left counterclockwise
            # 在变换后重新梳理corners的位置，使得corner顺序从左上角的开始
            for anno in annos:
                anno['segmentation'][0] = resort_corners(anno['segmentation'][0])

            record['instances'] = annotations_to_instances(annos, (h, w), mask_format="polygon")

        return record

        '''
        #这部分是旧的call
        file_name = os.path.join(self.root, path)
        #构造返回的img,和img相关的都需要改
        img = np.array(Image.open(file_name))
        #获得图像的长宽（need change）
        w, h = img.shape

        #向record中填入基本的数据
        record = {}
        record["file_name"] = file_name
        record["height"] = h
        record["width"] = w
        record['image_id'] = img_id
        
        for obj in target: obj["bbox_mode"] = BoxMode.XYWH_ABS

        record['annotations'] = target

        #image 转化为 tensor 并输入到 record, need to change
        if self.augmentations is None:
            record['image'] = (1/255) * torch.as_tensor(np.ascontiguousarray(np.expand_dims(img, 0)))
            record['instances'] = annotations_to_instances(target, (h, w), mask_format="polygon")
        #如果transform存在的话，就将图片transform后再输入到record中，值得注意的是transform是否能针对多通道的矩阵
        else:
            aug_input = T.AugInput(img)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            record['image'] = (1/255) * torch.as_tensor(np.array(np.expand_dims(image, 0)))
            #同样的，需要对于annotations也做transform同样的变换，这里面涉及到了img，也需要更改
            annos = [
                transform_instance_annotations(
                    obj, transforms, image.shape[:2]
                    )
                    for obj in record.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                    ]
            # resort corners after augmentation: so that all corners start from upper-left counterclockwise
            #在变换后重新梳理corners的位置，使得corner顺序从左上角的开始
            for anno in annos:
                anno['segmentation'][0] = resort_corners(anno['segmentation'][0])

            record['instances'] = annotations_to_instances(annos, (h, w), mask_format="polygon")
            
        return record
        '''

#做随机变换的部分，需要查证这些方法能否针对多通道的矩阵
#对训练集做随机变换，对验证和测试集则不做
def make_poly_transforms(image_set):

    if image_set == 'train':
        return T.AugmentationList([
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomRotation([0.0, 90.0, 180.0, 270.0], expand=False, center=None, sample_style="choice")
            ]) 
        
    if image_set == 'val' or image_set == 'test':
        return None

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.dataset_root)
    assert root.exists(), f'provided data path {root} does not exist'

    PATHS = {
        "train": (root / "train", root / "annotations" / 'train.json'),
        "val": (root / "val", root / "annotations" / 'val.json'),
        "test": (root / "test", root / "annotations" / 'test.json')
    }

    img_folder, ann_file = PATHS[image_set]
    
    dataset = MultiPoly(img_folder, ann_file, transforms=make_poly_transforms(image_set), semantic_classes=args.semantic_classes,slice_number=args.slice_num)
    
    return dataset
