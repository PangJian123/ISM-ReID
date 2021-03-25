# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import numpy as np
import torch
from torch import nn

from fastreid.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import *
from fastreid.modeling.Net import *
from .build import META_ARCH_REGISTRY
from fastreid.utils.weight_init import weights_init_kaiming

@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        self.teacher_net = build_backbone(cfg)
        self.student_net = build_backbone(cfg)
        self.D_Net = cam_Classifier(2048, 2).apply(weights_init_kaiming)
        if 'Dis_loss_cam' in self._cfg.MODEL.LOSSES.NAME:
            if "Hazy_DukeMTMC" in self._cfg.TDATASETS.NAMES:
                camid = int(8)
            elif "Hazy_Market1501" in self._cfg.TDATASETS.NAMES:
                camid = int(6)
            self.D_Net = CamClassifier(2048, camid)
        elif 'Dis_loss' in self._cfg.MODEL.LOSSES.NAME:
            if self._cfg.MODEL.PARAM.Dis_net == "cam_Classifier":
                self.D_Net = cam_Classifier(2048, 2).apply(weights_init_kaiming)
            elif self._cfg.MODEL.PARAM.Dis_net == "cam_Classifier_1024":
                self.D_Net = cam_Classifier_1024(2048, 2).apply(weights_init_kaiming)
            elif self._cfg.MODEL.PARAM.Dis_net == "cam_Classifier_1024_nobias":
                self.D_Net = cam_Classifier_1024_nobias(2048, 2).apply(weights_init_kaiming)
            elif self._cfg.MODEL.PARAM.Dis_net == "cam_Classifier_fc":
                self.D_Net = cam_Classifier_fc(2048, 2).apply(weights_init_kaiming)
            elif self._cfg.MODEL.PARAM.Dis_net == "cam_Classifier_fc_nobias_in_last_layer":
                self.D_Net = cam_Classifier_fc_nobias_in_last_layer(2048, 2).apply(weights_init_kaiming)

        self.D_Net = self.D_Net.to(torch.device(cfg.MODEL.DEVICE))
        self.CrossEntropy_loss = nn.CrossEntropyLoss().to(torch.device(cfg.MODEL.DEVICE))
        self.bn = nn.BatchNorm2d(2048)
        self.bn.bias.requires_grad_(False)
        self.bn.apply(weights_init_kaiming)

        # head
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'avgpool':      pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'maxpool':    pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":   pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        in_feat = cfg.MODEL.HEADS.IN_FEAT
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.teacher_heads = build_reid_heads(cfg, in_feat, num_classes, pool_layer)
        self.student_heads = build_reid_heads(cfg, in_feat, num_classes, pool_layer)

    @property
    def device(self):
        return self.pixel_mean.device

    def extract_data(self, batched_inputs, Paired=False):
        if Paired == True:
            real_imgs, hazy_imgs = self.preprocess_image(batched_inputs, Paired=Paired)
            targets = batched_inputs["targets"].long().to(self.device)
            hazy_targets = batched_inputs["hazy_targets"].long().to(self.device)
            return real_imgs, targets, hazy_imgs, hazy_targets
        else:
            real_imgs = self.preprocess_image(batched_inputs, Paired=Paired)
            targets = batched_inputs["targets"].long().to(self.device)
            return real_imgs, targets

    def forward(self, batched_inputs, t_batched_inputs=None, iters=None):

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            if t_batched_inputs == None:   # train baseline
                images = self.preprocess_image(batched_inputs)  # real_images
                t_features = self.teacher_net(images)  # real_features
                s_features = self.student_net(images)
                targets = batched_inputs["targets"].long().to(self.device)

                t_cls_outputs, t_features = self.teacher_heads(t_features, targets)
                s_cls_outputs, s_features = self.student_heads(s_features, targets)
                losses = self.losses(t_cls_outputs, t_features, targets,
                                     s_cls_outputs, s_features, targets,

                                     )
                return losses
            else:
                s_images = self.preprocess_image(batched_inputs)  # source real_images
                targets = batched_inputs["targets"].long().to(self.device)
                t_images = self.preprocess_image(t_batched_inputs)
                t_hazy_images = self.preprocess_image(t_batched_inputs, hazy=True)
                t_cam_id = t_batched_inputs["hazy_camid"].long().to(self.device)
                # teacher flow
                # t_features = self.teacher_net(s_images)
                # t_cls_outputs, t_features = self.teacher_heads(t_features, targets)
                t_tea_features = self.teacher_net(t_images)
                t_tea_features = self.teacher_heads(t_tea_features, t_data=True)
                # student flow
                s_features = self.student_net(s_images)
                s_cls_outputs, s_stu_features = self.student_heads(s_features, targets)
                t_hazy_stu_features = self.student_net(t_hazy_images)
                t_hazy_stu_features = self.student_heads(t_hazy_stu_features, t_data=True)
                losses = self.load_losses(s_cls_outputs, s_stu_features, targets, t_tea_features,
                                          t_hazy_stu_features, iters, t_cam_id)
                return losses


        else:
            images = self.preprocess_image(batched_inputs)  # real_images
            features = self.student_net(images)
            pred_features = self.student_heads(features)
            return pred_features  # real_feature



    def preprocess_image(self, batched_inputs, hazy=False):
        """
        Normalize and batch the input images.
        """
        if hazy:
            hazy_images = batched_inputs["hazy_images"].to(self.device)
            hazy_images.sub_(self.pixel_mean).div_(self.pixel_std)
            return hazy_images
        images = batched_inputs["images"].to(self.device)
        # images = batched_inputs
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def select_func(self, f, hazy_f, gt=None, hazy_gt=None):
        if hazy_gt==None:
            pair_pred_logits = self.D_net(torch.abs(f-hazy_f))
            pair_gt_labels = torch.tensor([0] * pair_pred_logits.size(0)).long().to(self.device)

            m = hazy_f.size(0)
            xx = torch.pow(hazy_f, 2).sum(1, keepdim=True).expand(m, m)
            yy = torch.pow(hazy_f, 2).sum(1, keepdim=True).expand(m, m).t()
            dist = xx + yy
            dist.addmm_(1, -2, hazy_f.float(), hazy_f.t().float())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
            dismat = dist.cpu().detach().numpy()
            indices = np.argsort(dismat, axis=1)
            fake_imgs_idx = indices[:, 0]  # select the soft hard negative sample in a minibatch
            fake_pair_pred_logits = self.D_net(torch.abs(f-hazy_f[fake_imgs_idx]))
            fake_pair_gt_labels = torch.tensor([1] * fake_pair_pred_logits.size(0)).long().to(self.device)
            return pair_pred_logits, pair_gt_labels, fake_pair_pred_logits, fake_pair_gt_labels
        else:
            pair_f = torch.cat([f, hazy_f], dim=1)
            pair_pred_logits = self.D_net(pair_f)
            pair_gt_labels = torch.tensor([0] * pair_pred_logits.size(0)).long().to(self.device)
            m = hazy_f.size(0)

            xx = torch.pow(hazy_f, 2).sum(1, keepdim=True).expand(m, m)
            yy = torch.pow(hazy_f, 2).sum(1, keepdim=True).expand(m, m).t()
            dist = xx + yy
            dist.addmm_(1, -2, hazy_f, hazy_f.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
            dismat = dist.cpu().detach().numpy()
            indices = np.argsort(dismat, axis=1)
            fake_imgs = indices[:, 1]  # select the hard negative sample in a minibatch
            fake_pair_f = torch.cat([hazy_f, hazy_f[fake_imgs]], dim=1)
            fake_pair_pred_logits = self.D_net(fake_pair_f)
            fake_pair_gt_labels = torch.tensor([1] * fake_pair_pred_logits.size(0)).long().to(self.device)
            return pair_pred_logits, pair_gt_labels, fake_pair_pred_logits, fake_pair_gt_labels

    def losses(self, t_cls_outputs, t_pred_features, gt_labels,
               s_cls_outputs, s_pred_features, hazy_gt_labels,
               images=None, hazy_images=None, t_features=None, t_hazy_features=None):
        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "t_TripletLoss" in loss_names:
            loss_dict['loss_triplet_t'] = TripletLoss(self._cfg)(t_pred_features.float(), gt_labels)
        if "s_TripletLoss" in loss_names:
            loss_dict['loss_triplet_s'] = TripletLoss(self._cfg)(s_pred_features.float(), gt_labels)

        if "CircleLoss" in loss_names:
            loss_dict['loss_circle'] = CircleLoss(self._cfg)(t_pred_features, gt_labels)

        if "t_CrossEntropyLoss" in loss_names:
            loss_dict['t_loss_cls'] = CrossEntropyLoss(self._cfg)(t_cls_outputs, gt_labels)

        if "Hazy_TripletLoss" in loss_names:
            loss_dict['hazy_loss_triplet'] = TripletLoss(self._cfg)(s_pred_features, hazy_gt_labels)

        if 's_CrossEntropyLoss' in loss_names:
            loss_dict['s_loss_cls'] = CrossEntropyLoss(self._cfg)(s_cls_outputs, hazy_gt_labels)


        if "Tar_KD_loss" in loss_names:
            loss_dict['Tar_KD_loss'] = KDLoss(self._cfg)(t_features, t_hazy_features)

        if "Dis_loss" in loss_names:
            pair_pred_logits, pair_gt_labels, \
            fake_pair_pred_logits, fake_pair_gt_labels = self.select_func(f=t_features, hazy_f=t_hazy_features)
            a = torch.cat([pair_pred_logits, fake_pair_pred_logits], dim=0)
            b = torch.cat([pair_gt_labels.reshape([-1, 1]), fake_pair_gt_labels.reshape([-1, 1])], dim=0).float()
            loss_dict['Dis_loss'] = Dis_loss()(a, b)

        return loss_dict

    def load_losses(self, s_cls_outputs, s_stu_features, targets, t_tea_features,
                    t_hazy_stu_features, iters=None, t_cam_id=None):
        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "s_TripletLoss" in loss_names:
            loss_dict['loss_triplet_s'] = TripletLoss(self._cfg)(s_stu_features.float(), targets)

        if 's_CrossEntropyLoss' in loss_names:
            loss_dict['s_loss_cls'] = CrossEntropyLoss(self._cfg)(s_cls_outputs, targets)

        if "Tar_KD_loss" in loss_names:
            loss_dict['Tar_KD_loss'] = KDLoss(self._cfg)(t_tea_features, t_hazy_stu_features)

        if "L1_loss" in loss_names:
            loss_dict['L1_loss'] = L1Loss(self._cfg)(t_tea_features, t_hazy_stu_features)

        if "BachDistance_loss_t" in loss_names:
            loss_dict['BachDistance_loss_t'] = BachDistance_loss(self._cfg)(t_tea_features, t_hazy_stu_features)


        if "Dis_loss" in loss_names:
            real_logits, real_fea = self.D_Net(t_tea_features.detach())
            hazy_logits, hazy_fea = self.D_Net(t_hazy_stu_features.detach())
            real_label = torch.tensor([0] * t_tea_features.size(0)).long().to(self.device)
            hazy_label = torch.tensor([1] * t_hazy_stu_features.size(0)).long().to(self.device)
            loss_real = self.CrossEntropy_loss(real_logits, real_label)
            loss_hazy = self.CrossEntropy_loss(hazy_logits, hazy_label)
            loss_dict['Dis_loss_1'] = loss_real + loss_hazy
            real_logits, real_fea = self.D_Net(t_tea_features)
            hazy_logits, hazy_fea = self.D_Net(t_hazy_stu_features)
            loss_dict['Dis_loss_2'] = KDLoss(self._cfg)(real_fea, hazy_fea)

        return loss_dict


