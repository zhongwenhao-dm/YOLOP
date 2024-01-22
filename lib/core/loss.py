import torch.nn as nn
import torch
from .general import bbox_iou
from .postprocess import build_targets
from lib.core.evaluate import SegmentationMetric
import scipy.spatial as spt
import numpy as np

class MultiHeadLoss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """
        # head_losses = [ll
        #                 for l, f, t in zip(self.losses, head_fields, head_targets)
        #                 for ll in l(f, t)]
        #
        # assert len(self.lambdas) == len(head_losses)
        # loss_values = [lam * l
        #                for lam, l in zip(self.lambdas, head_losses)
        #                if l is not None]
        # total_loss = sum(loss_values) if loss_values else None
        # print(model.nc)
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        BCEcls, BCEobj, BCEseg = self.losses

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        drive_area_seg_predicts = predictions[1].view(-1)
        drive_area_seg_targets = targets[1].view(-1)
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)

        lane_line_seg_predicts = predictions[2].view(-1)
        lane_line_seg_targets = targets[2].view(-1)
        lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)

        metric = SegmentationMetric(2)
        nb, _, height, width = targets[1].shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        _,lane_line_pred=torch.max(predictions[2], 1)
        _,lane_line_gt=torch.max(targets[2], 1)
        lane_line_pred = lane_line_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
        lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        metric.reset()
        metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
        IoU = metric.IntersectionOverUnion()
        liou_ll = 1 - IoU

        s = 3 / no  # output count scaling
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]

        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
        liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[5]

        
        if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:
            lseg_da = 0 * lseg_da
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            
        if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox

        if cfg.TRAIN.LANE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_da = 0 * lseg_da

        if cfg.TRAIN.DRIVABLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll

        loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return loss, (lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), liou_ll.item(), loss.item())

def get_loss(cfg, device):
    """
    get MultiHeadLoss

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)

    """
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
    # segmentation loss criteria
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    # Focal loss
    gamma = cfg.LOSS.FL_GAMMA  # focal loss gamma
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    loss_list = [BCEcls, BCEobj, BCEseg]
    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss

class MultiHeadLoss_pole(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, po_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 4)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """
        # head_losses = [ll
        #                 for l, f, t in zip(self.losses, head_fields, head_targets)
        #                 for ll in l(f, t)]
        #
        # assert len(self.lambdas) == len(head_losses)
        # loss_values = [lam * l
        #                for lam, l in zip(self.lambdas, head_losses)
        #                if l is not None]
        # total_loss = sum(loss_values) if loss_values else None
        # print(model.nc)
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3, det_head4], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets, pole_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        BCEcls, BCEobj, BCEseg = self.losses

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        drive_area_seg_predicts = predictions[1].view(-1)  #view(-1)表示把2维展开成1维【
        drive_area_seg_targets = targets[1].view(-1)
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)

        lane_line_seg_predicts = predictions[2].view(-1)
        lane_line_seg_targets = targets[2].view(-1)
        lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)

        # add pole
        pole_line_seg_predicts = predictions[3].view(-1)
        pole_line_seg_targets = targets[3].view(-1)
        lseg_po = BCEseg(pole_line_seg_predicts, pole_line_seg_targets)

        metric = SegmentationMetric(2)
        nb, _, height, width = targets[1].shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        _,lane_line_pred=torch.max(predictions[2], 1)
        _,lane_line_gt=torch.max(targets[2], 1)
        lane_line_pred = lane_line_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
        lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        metric.reset()
        metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
        IoU = metric.IntersectionOverUnion()
        liou_ll = 1 - IoU

        s = 3 / no  # output count scaling
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]

        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
        # add pole
        lseg_po *= cfg.LOSS.PO_SEG_GAIN * self.lambdas[5]
        liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[6]

        # import pdb; pdb.set_trace()

        
        if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:
            lseg_da = 0 * lseg_da
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            lseg_po = 0 * lseg_po
            
        if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox

        if cfg.TRAIN.LANE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_da = 0 * lseg_da
            lseg_po = 0 * lseg_po

        if cfg.TRAIN.DRIVABLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            lseg_po = 0 * lseg_po

        if cfg.TRAIN.POLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            lseg_da = 0 * lseg_da

        loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll + lseg_po
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return loss, (lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), lseg_po.item(), liou_ll.item(), loss.item())

def get_loss_pole(cfg, device):
    """
    get MultiHeadLoss

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)

    """
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
    # segmentation loss criteria
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    # Focal loss
    gamma = cfg.LOSS.FL_GAMMA  # focal loss gamma
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    loss_list = [BCEcls, BCEobj, BCEseg]
    loss = MultiHeadLoss_pole(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss

# 加上了车道线损失laa
def get_loss_pole_da2ll(cfg, device):
    """
    get MultiHeadLoss
    ++

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)

    """
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
    # segmentation loss criteria
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    # Focal loss
    gamma = cfg.LOSS.FL_GAMMA  # focal loss gamma
    if gamma > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

    loss_list = [BCEcls, BCEobj, BCEseg]
    loss = MultiHeadLoss_pole_da2ll(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)
    return loss

class MultiHeadLoss_pole_da2ll(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, po_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 4)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """
        # head_losses = [ll
        #                 for l, f, t in zip(self.losses, head_fields, head_targets)
        #                 for ll in l(f, t)]
        #
        # assert len(self.lambdas) == len(head_losses)
        # loss_values = [lam * l
        #                for lam, l in zip(self.lambdas, head_losses)
        #                if l is not None]
        # total_loss = sum(loss_values) if loss_values else None
        # print(model.nc)
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3, det_head4], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets, pole_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        BCEcls, BCEobj, BCEseg = self.losses

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        drive_area_seg_predicts = predictions[1].view(-1)  #view(-1)表示把2维展开成1维【
        drive_area_seg_targets = targets[1].view(-1)
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)

        lane_line_seg_predicts = predictions[2].view(-1)
        lane_line_seg_targets = targets[2].view(-1)
        lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)

        # add area aware loss for ll
        laa_ll = area_aware_loss_for_ll(predictions[2], targets[1])
        laa_ll = 0

        # add boundary awary loss for da
        # lba_da = boundary_aware_loss_for_da(predictions[1], targets[2])

        # add pole
        pole_line_seg_predicts = predictions[3].view(-1)
        pole_line_seg_targets = targets[3].view(-1)
        lseg_po = BCEseg(pole_line_seg_predicts, pole_line_seg_targets)

        metric = SegmentationMetric(2)
        nb, _, height, width = targets[1].shape
        pad_w, pad_h = shapes[0][1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        _,lane_line_pred=torch.max(predictions[2], 1)
        _,lane_line_gt=torch.max(targets[2], 1)
        lane_line_pred = lane_line_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
        lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        metric.reset()
        metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
        IoU = metric.IntersectionOverUnion()
        liou_ll = 1 - IoU

        s = 3 / no  # output count scaling
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]

        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
        # add pole
        lseg_po *= cfg.LOSS.PO_SEG_GAIN * self.lambdas[5]
        liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[6]

        laa_ll *= 1
        # print(laa_ll)

        
        if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:
            lseg_da = 0 * lseg_da
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            lseg_po = 0 * lseg_po
            
        if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox

        if cfg.TRAIN.LANE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_da = 0 * lseg_da
            lseg_po = 0 * lseg_po

        if cfg.TRAIN.DRIVABLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            lseg_po = 0 * lseg_po

        if cfg.TRAIN.POLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_ll = 0 * lseg_ll
            liou_ll = 0 * liou_ll
            lseg_da = 0 * lseg_da

        loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll + lseg_po + laa_ll
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return loss, (lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), lseg_po.item(), liou_ll.item(), loss.item())


# example
# class L1_Loss(nn.Module)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

def area_aware_loss_for_ll_ori(ll_pre, da_gt):
    score = 0
    # ll_pre,da_gt : torch.Size([48, 2, 384, 640])
    # 从da_gt中值为1的点中，找到ll_pre中最近的点的概率p，然后累加1-p
    # ll_pre = ll_pre.cpu()
    # da_gt = da_gt.cpu()
    print('cal laa!-------')
    # print(ll_pre.shape)

    for i in range(ll_pre.shape[0]):
        s = 0
        da_gt_temp = da_gt[i,0]
        ll_pre_temp = ll_pre[i,0]
        # da_idx = np.array(np.where(da_gt_temp==0)).T
        # ll_idx = np.array(np.where(ll_pre_temp < 0.9)).T
        da_idx = torch.where(da_gt_temp==0)
        ll_idx = torch.where(ll_pre_temp < 0.9)
        da_idx = torch.stack(da_idx,dim=1)
        ll_idx = torch.stack(ll_idx,dim=1)

        if len(ll_idx)==0 or len(da_idx)==0:
            continue

        # import pdb; pdb.set_trace()

        tree = spt.KDTree(data=ll_idx)

        for i in range(len(da_idx)):
            point = da_idx[i].cpu()
            distance, index = tree.query(point, k=1)

            # print('距离{}最近的是{}，距离为{:.2f}，概率为{:.3f}'.format(point, ll_idx[index], distance, ll_pre[ll_idx[index][0],ll_idx[index][1]]))
            # print(index)
            # print(ll_idx[index])
            s += 1 - ll_pre_temp[ll_idx[index][0],ll_idx[index][1]]
        s /= (384*640)
        score += s

    return score

def area_aware_loss_for_ll(ll_pre, da_gt):
    # da_gt里面0是真值
    score = 0

    ll_pre_temp = ll_pre[:, 0]
    da_gt_temp = da_gt[:, 0]

    # 构建 KDtree
    # pre里面如何界定一个合适的值？比如对一些已经训练比较好的模型，车道线此时输出的概率已经比较高（0.8以上），此时能够比较合适的设置阈值。
    ll_idx_points = torch.stack(torch.where(ll_pre_temp < 0.9), dim=1).cpu().numpy()  
    if ll_idx_points.size == 0:
        return score
    tree = spt.KDTree(data=ll_idx_points)

    # 获取所有非零点的索引
    da_idx_points = torch.stack(torch.where(da_gt_temp == 0), dim=1).cpu().numpy()

    # 在 KD 树中查询所有最近邻
    _, indices = tree.query(da_idx_points, k=1)

    # 计算损失
    sss = torch.sum(1 - ll_pre_temp[ll_idx_points[indices][:, 0], ll_idx_points[indices][:, 1], ll_idx_points[indices][:, 2]])
    # import pdb; pdb.set_trace();

    sss /= (384 * 640)
    score += sss
    
    import pdb; pdb.set_trace()

    return score

# 用以计算drivable area的预测值和真值之间的iou损失
def boundary_aware_loss_for_da(da_pre, ll_gt):
    score = 0


    return score

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
