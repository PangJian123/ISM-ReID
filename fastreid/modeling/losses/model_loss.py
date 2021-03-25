import torch
from torch import nn as nn
from torch.nn import functional as F
class Dis_loss(object):
    """
    A class that stores information and compute losses about outputs of a Baseline head.
    """

    def __init__(self):
        # self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        # self._eps = cfg.MODEL.LOSSES.CE.EPSILON
        # self._alpha = cfg.MODEL.LOSSES.CE.ALPHA
        # self._scale = cfg.MODEL.LOSSES.CE.SCALE
        self.loss = torch.nn.BCEWithLogitsLoss()

    def __call__(self, pred_class_logits, gt_classes):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        return self.loss(pred_class_logits, gt_classes)


class L1Loss(object):
    """
    A class that stores information and compute losses about outputs of a Baseline head.
    """

    def __init__(self, cfg):
        # self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        # self._eps = cfg.MODEL.LOSSES.CE.EPSILON
        # self._alpha = cfg.MODEL.LOSSES.CE.ALPHA
        # self._scale = cfg.MODEL.LOSSES.CE.SCALE
        self.loss = torch.nn.L1Loss()
        self.param = cfg.MODEL.PARAM.L1_param

    def __call__(self, inputs, targets):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        return self.loss(inputs, targets) * self.param


# class KLLoss(object):
#     """
#     A class that stores information and compute losses about outputs of a Baseline head.
#     """
#
#     def __init__(self):
#         # self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
#         # self._eps = cfg.MODEL.LOSSES.CE.EPSILON
#         # self._alpha = cfg.MODEL.LOSSES.CE.ALPHA
#         # self._scale = cfg.MODEL.LOSSES.CE.SCALE
#         self.loss = torch.nn.BCELoss()
#
#     def __call__(self, pred_class_logits, gt_classes):
#         """
#         Compute the softmax cross entropy loss for box classification.
#         Returns:
#             scalar Tensor
#         """
#         return self.loss(pred_class_logits, gt_classes)

#---------------source_code---------------
class KDLoss_source_code(nn.Module):

    def __init__(self, temp: float, reduction: str):
        super(KDLoss_source_code, self).__init__()

        self.temp = temp
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)
        # self.temp = cfg.MODEL.PARAM.KD_TEMP
        # self.reduction = cfg.MODEL.PARAM.KD_red
        # self.kl_loss = nn.KLDivLoss(reduction=self.reduction)

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor):

        student_softmax = F.log_softmax(student_logits / self.temp, dim=-1)
        teacher_softmax = F.softmax(teacher_logits / self.temp, dim=-1)

        kl = nn.KLDivLoss(reduction='none')(student_softmax, teacher_softmax)
        kl = kl.sum() if self.reduction == 'sum' else kl.sum(1).mean()
        kl = kl * (self.temp ** 2)

        return kl

    def __call__(self, *args, **kwargs):
        return super(KDLoss_source_code, self).__call__(*args, **kwargs)


class KDLoss(nn.Module):
    def __init__(self, cfg):
        super(KDLoss, self).__init__()

        self.temp = cfg.MODEL.PARAM.KD_TEMP
        self.param = cfg.MODEL.PARAM.KD_PARAM
        self.reduction = cfg.MODEL.PARAM.KD_red
        self.kl_loss = nn.KLDivLoss(reduction=self.reduction)

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor):

        student_softmax = F.log_softmax(student_logits / self.temp, dim=-1)
        teacher_softmax = F.softmax(teacher_logits / self.temp, dim=-1)

        kl = nn.KLDivLoss(reduction='none')(student_softmax, teacher_softmax)
        kl = kl.sum() if self.reduction == 'sum' else kl.sum(1).mean()
        kl = kl * (self.temp ** 2)

        return kl * self.param

    def __call__(self, *args, **kwargs):
        return super(KDLoss, self).__call__(*args, **kwargs)
#     def __init__(self, cfg):
#         super(KDLoss, self).__init__()
# # temp=10., reduction='mean'
#         self.temp = cfg.MODEL.PARAM.KD_TEMP
#         self.reduction = cfg.MODEL.PARAM.KD_red
#         self.kl_loss = nn.KLDivLoss(reduction=self.reduction)
#
#     def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor):
#
#         student_softmax = F.softmax(student_logits / self.temp, dim=-1)
#         teacher_softmax = F.softmax(teacher_logits / self.temp, dim=-1)
#         kl = self.kl_loss(student_softmax, teacher_softmax)
#
#         # kl = nn.KLDivLoss(reduction='none')(student_softmax, teacher_softmax)
#         # kl = kl.sum() if self.reduction == 'sum' else kl.sum(1).mean()
#
#         return kl * self.temp
#
#     def __call__(self, *args, **kwargs):
#         return super(KDLoss, self).__call__(*args, **kwargs)


class BachDistance_loss(object):
    def __init__(self, cfg):
        self.loss = torch.nn.L1Loss(reduction='mean')
        # self.loss = torch.nn.MSELoss()
        self.param = cfg.MODEL.PARAM.BD_param
        self.metric = cfg.MODEL.PARAM.METRIC

    def __call__(self, f, hazy_f):
        if self.metric == "euclidean":
            m = f.size(0)
            xx = torch.pow(f, 2).sum(1, keepdim=True).expand(m, m)
            yy = torch.pow(f, 2).sum(1, keepdim=True).expand(m, m).t()
            dist_f = xx + yy
            dist_f.addmm_(1, -2, f.float(), f.t().float())
            dist_f = dist_f.clamp(min=1e-12).sqrt()  # for numerical stability


            m = hazy_f.size(0)
            xx = torch.pow(hazy_f, 2).sum(1, keepdim=True).expand(m, m)
            yy = torch.pow(hazy_f, 2).sum(1, keepdim=True).expand(m, m).t()
            dist_hf = xx + yy
            dist_hf.addmm_(1, -2, hazy_f.float(), hazy_f.t().float())
            dist_hf = dist_hf.clamp(min=1e-12).sqrt()  # for numerical stability
            loss = self.loss(dist_f, dist_hf)
            return loss * self.param
        elif self.metric == "cosine":

            f_feat = F.normalize(f, dim=1)
            dist_f = 1 - torch.mm(f_feat, f_feat.t())
            # cos_dist_f = f_feat.mm(f_feat.t())
            hf_feat = F.normalize(hazy_f, dim=1)
            dist_hf = 1 - torch.mm(hf_feat, hf_feat.t())
            # cos_dist_hf = hf_feat.mm(hf_feat.t())
            loss = self.loss(dist_f, dist_hf)
            # loss = self.loss(cos_dist_f, cos_dist_hf)
            return loss * self.param
        elif self.metric == "fusion":
            m = f.size(0)
            xx = torch.pow(f, 2).sum(1, keepdim=True).expand(m, m)
            yy = torch.pow(f, 2).sum(1, keepdim=True).expand(m, m).t()
            dist_f = xx + yy
            dist_f.addmm_(1, -2, f.float(), f.t().float())
            euc_dist_f = dist_f.clamp(min=1e-12).sqrt()  # for numerical stability


            m = hazy_f.size(0)
            xx = torch.pow(hazy_f, 2).sum(1, keepdim=True).expand(m, m)
            yy = torch.pow(hazy_f, 2).sum(1, keepdim=True).expand(m, m).t()
            dist_hf = xx + yy
            dist_hf.addmm_(1, -2, hazy_f.float(), hazy_f.t().float())
            euc_dist_hf = dist_hf.clamp(min=1e-12).sqrt()  # for numerical stability


            f_feat = F.normalize(f, dim=1)
            cos_dist_f = f_feat.mm(f_feat.t())
            hf_feat = F.normalize(hazy_f, dim=1)
            cos_dist_hf = hf_feat.mm(hf_feat.t())
            # loss = torch.cosine_similarity(f_feat, hf_feat, dim=1)
            dist_f = (euc_dist_f + cos_dist_f)
            dist_hf = (euc_dist_hf + cos_dist_hf)
            loss = self.loss(dist_f, dist_hf)

            return loss * self.param





        # dismat = dist.cpu().detach().numpy()
        # indices = np.argsort(dismat, axis=1)
        # fake_imgs_idx = indices[:, 0]  # select the soft hard negative sample in a minibatch
        # fake_pair_pred_logits = self.D_net(torch.abs(f-hazy_f[fake_imgs_idx]))
        # fake_pair_gt_labels = torch.tensor([1] * fake_pair_pred_logits.size(0)).long().to(self.device)
        # return pair_pred_logits, pair_gt_labels, fake_pair_pred_logits, fake_pair_gt_labels